import math
import warnings
from functools import partial

import flax.linen.attention
import jax
from chex import Array
from fjformer import with_sharding_constraint
from flax.linen import dot_product_attention_weights
from flax.linen.dtypes import promote_dtype
from jax import numpy as jnp, lax, random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, Mesh
from fjformer.pallas_operations.flash_attention import gpu as flash_attn_gpu
from fjformer.pallas_operations.flash_attention import tpu as flash_attn_tpu
from fjformer.pallas_operations.ring_attention import (
    ring_flash_attention_tpu,
    ring_attention_standard,
    ring_attention
)
from fjformer.pallas_operations.splash_attention import (
    make_splash_mha,
    make_splash_mqa,
    SegmentIds,
    BlockSizes as SplashBlockSizes,
    CausalMask,
    MultiHeadMask
)
from typing import Tuple, Callable, Type, Any, Optional, Literal, Union
from dataclasses import dataclass

from .flax_modelling_utils import get_gradient_checkpoint_policy


@dataclass
class AttentionOutput:
    attention_weights: Optional[Array] = None
    attention_outputs: Optional[Array] = None


def attention_production(
        query_states: jax.Array,
        key_states: jax.Array,
        value_states: jax.Array,
        attention_bias: jax.Array | None = None,
        deterministic: bool = True,
        dropout_rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
        dropout_rate: float = 0.0
):
    batch, q_sequence_length, q_num_head, head_dim = query_states.shape
    _, kv_sequence_length, kv_num_head, _ = key_states.shape
    assert q_num_head % kv_num_head == 0, (
        f"`query_states` {q_num_head} must be a multiple of `key_states` "
        f"and `value_states` heads {kv_num_head}"
    )
    query_states = jnp.reshape(query_states,
                               (batch, q_sequence_length, kv_num_head, q_num_head // kv_num_head, head_dim))
    attention_score = jnp.einsum(
        "...thHd,...Thd->...hHtT",
        query_states,
        key_states
    ).astype(
        jnp.float32
    )
    attention_score *= 1 / math.sqrt(head_dim)
    max_attention_value = jnp.array(30.0, dtype=attention_score.dtype)
    attention_score = max_attention_value * jnp.tanh(attention_score / max_attention_value)
    attention_score = attention_score + attention_bias[:, :, None, :, :]
    attention_weights = jax.nn.softmax(attention_score).astype(query_states.dtype)
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weights.shape[-2:]
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        multiplier = keep.astype(query_states.dtype) / jnp.asarray(keep_prob, dtype=query_states.dtype)
        attention_weights = attention_weights * multiplier

    attention = jnp.einsum("...hHtT,...Thd->...thHd", attention_weights, value_states).reshape(
        batch, q_sequence_length, q_num_head, head_dim
    )
    return attention


def static_sharded_attention_production(
        query_states: jax.Array,
        key_states: jax.Array,
        value_states: jax.Array,
        attention_bias: jax.Array | None = None,
        deterministic: bool = True,
        dropout_rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
        dropout_rate: float = 0.0
):
    assert key_states.shape[1] == value_states.shape[1], "miss match on key_states and value_states sequence length"
    is_generating = query_states.shape[1] == 1 or query_states.shape[1] != key_states.shape[1]
    sequence_sharding_axis_name = None if is_generating else "sp"
    tensor_sharding_axis_name = "sp" if is_generating else "tp"
    query_states = with_sharding_constraint(
        query_states,
        PartitionSpec(
            ("dp", "fsdp"),
            sequence_sharding_axis_name,
            tensor_sharding_axis_name,
            None
        )
    )
    key_states = with_sharding_constraint(
        key_states,
        PartitionSpec(
            ("dp", "fsdp"),
            sequence_sharding_axis_name,
            tensor_sharding_axis_name,
            None
        )
    )
    value_states = with_sharding_constraint(
        value_states,
        PartitionSpec(
            ("dp", "fsdp"),
            sequence_sharding_axis_name,
            tensor_sharding_axis_name,
            None
        )
    )

    batch, q_sequence_length, q_num_head, head_dim = query_states.shape
    _, kv_sequence_length, kv_num_head, _ = key_states.shape

    assert q_num_head % kv_num_head == 0, (
        f"`query_states` {q_num_head} must be a multiple of"
        f" `key_states` and `value_states` heads {kv_num_head}"
    )

    query_states = jnp.reshape(
        query_states,
        (batch, q_sequence_length, kv_num_head, q_num_head // kv_num_head, head_dim)
    )

    query_states = with_sharding_constraint(
        query_states, PartitionSpec(
            ("dp", "fsdp"),
            sequence_sharding_axis_name,
            tensor_sharding_axis_name,
            None,
            None
        )
    )

    attention_score = jnp.einsum(
        "...thHd,...Thd->...hHtT",
        query_states, key_states
    ).astype(jnp.float32)

    attention_score *= 1 / math.sqrt(head_dim)

    max_attention_value = jnp.array(30.0, dtype=attention_score.dtype)
    attention_score = max_attention_value * jnp.tanh(attention_score / max_attention_value)
    attention_score = attention_score + attention_bias[:, :, None, :, :]

    attention_weights = jax.nn.softmax(attention_score).astype(query_states.dtype)
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weights.shape[-2:]
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        multiplier = keep.astype(query_states.dtype) / jnp.asarray(keep_prob, dtype=query_states.dtype)
        attention_weights = attention_weights * multiplier

    attention = jnp.einsum("...hHtT,...Thd->...thHd", attention_weights, value_states).reshape(
        batch, q_sequence_length, q_num_head, head_dim
    )

    attention = with_sharding_constraint(
        attention,
        PartitionSpec(
            ("dp", "fsdp"),
            sequence_sharding_axis_name,
            tensor_sharding_axis_name,
            None,
        )
    )

    return attention


def static_sharded_dot_product_attention(
        query_states: Array,
        key_states: Array,
        value_states: Array,
        bias: Optional[Array] = None,
        mask: Optional[Array] = None,
        broadcast_dropout: bool = True,
        dropout_rng: Optional[jax.random.PRNGKey] = None,
        dropout_rate: float = 0.0,
        deterministic: bool = False,
        dtype: Optional[jnp.dtype] = jnp.float32,
        precision: Optional[Union[str, lax.Precision]] = None,
        shard_attention_computation: bool = True
):
    assert key_states.shape[1] == value_states.shape[1], "miss match on key_states and value_states sequence length"
    assert query_states.ndim == key_states.ndim, "q, k must have same rank."
    assert query_states.shape[:-3] == key_states.shape[:-3], "q, k batch dims must match."
    assert query_states.shape[-2] == key_states.shape[-2], "q, k num_heads must match."
    assert query_states.shape[-1] == key_states.shape[-1], "q, k depths must match."

    query_states, key_states, value_states = promote_dtype(query_states, key_states, value_states, dtype=dtype)

    if query_states.shape[1] == 1:
        sequence_sharding_axis_name = None
        tensor_sharding_axis_name = "sp"
    elif query_states.shape[1] != key_states.shape[1]:
        sequence_sharding_axis_name = None
        tensor_sharding_axis_name = None
    else:
        sequence_sharding_axis_name = "sp"
        tensor_sharding_axis_name = "tp"

    if shard_attention_computation:
        query_states = with_sharding_constraint(
            query_states, PartitionSpec(
                ("dp", "fsdp"),
                sequence_sharding_axis_name,
                tensor_sharding_axis_name,
                None
            )
        )

        key_states = with_sharding_constraint(
            key_states, PartitionSpec(
                ("dp", "fsdp"),
                sequence_sharding_axis_name,
                tensor_sharding_axis_name,
                None
            )
        )

        value_states = with_sharding_constraint(
            value_states, PartitionSpec(
                ("dp", "fsdp"),
                sequence_sharding_axis_name,
                tensor_sharding_axis_name,
                None
            )
        )

    depth = query_states.shape[-1]
    query_states = query_states / jnp.sqrt(depth).astype(dtype)
    attention_weight = jnp.einsum(
        "...qhd,...khd->...hqk",
        query_states, key_states, precision=precision
    )
    if shard_attention_computation:
        attention_weight = with_sharding_constraint(
            attention_weight, PartitionSpec(
                ("dp", "fsdp"),
                None,
                sequence_sharding_axis_name,
                None
            )
        )
        if bias is not None:
            bias = with_sharding_constraint(
                bias, PartitionSpec(
                    ("dp", "fsdp"),
                    None,
                    sequence_sharding_axis_name,
                    None
                )
            )
    if bias is not None:
        attention_weight = attention_weight + bias
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attention_weight = jnp.where(mask, attention_weight, big_neg)
    attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attention_weight.shape)  # type: ignore
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attention_weight = attention_weight * multiplier
    attention = jnp.einsum(
        "...hqk,...khd->...qhd",
        attention_weight, value_states, precision=precision
    )
    if shard_attention_computation:
        attention = with_sharding_constraint(
            attention, PartitionSpec(
                ("dp", "fsdp"),
                sequence_sharding_axis_name,
                tensor_sharding_axis_name,
                None
            )
        )
    return attention


def get_flash_attention() -> Tuple[Callable, bool, bool]:
    """
    return: FlashAttention FN, Upcast Needed to float32,do_shard_map
    """
    platform = jax.lib.xla_bridge.get_backend().platform
    if platform == "gpu":
        float32_logits = False
        ring_attention_fn = flash_attn_gpu.mha
        do_shard_map = True
    elif platform == "tpu":
        float32_logits = True
        ring_attention_fn = flash_attn_tpu.flash_attention
        do_shard_map = False
    else:
        raise ValueError(f"Unsupported platform {platform}")

    return ring_attention_fn, float32_logits, do_shard_map


class EasyAttention:
    def __init__(
            self,
            mesh: Mesh,
            attn_mechanism: Literal[
                "normal",
                "flash",
                "splash",
                "ring",
                "cudnn"
            ],
            num_attention_heads: int,
            head_dims: int,
            block_k: int = 512,
            block_q: int = 512,
            block_b: int = 512,
            block_k_major: int = 512,
            block_q_major_dkv: int = 512,
            block_k_major_dkv: int = 512,
            block_k_dkv: int = 512,
            block_q_dkv: int = 512,
            block_k_major_dq: int = 512,
            block_k_dq: int = 512,
            block_q_dq: int = 512,
            sm_scale: Optional[float] = None,
            query_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            generation_query_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, "tp", None),
            key_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            value_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            bias_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, "sp", None),
            generation_bias_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, None, None),
            attention_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            generation_attention_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, "tp", None),
            scan_ring_attention: bool = False,
            scan_attention_layers: bool = False,
            attention_dropout: float = 0.0,
            dtype: jnp.dtype = jnp.float32,
            precision: lax.Precision = lax.Precision("fastest"),
            force_float32_tpu: bool = True,
            shard_attention_computation: bool = True,
            use_sharding_constraint: Optional[bool] = False,
    ):
        platform = jax.lib.xla_bridge.get_backend().platform
        if attn_mechanism == "splash":
            raise NotImplementedError("Splash Attention is not Supported YET !")
        if attn_mechanism == "flash" and platform not in ["gpu", "tpu"]:
            raise NotImplementedError("Flash Attention is only supported for GPU/TPU.")
        self.platform = platform
        self.attn_mechanism = attn_mechanism
        self.block_k = block_k
        self.block_q = block_q
        self.block_b = block_b
        self.block_k_major = block_k_major
        self.block_q_major_dkv = block_q_major_dkv
        self.block_k_major_dkv = block_k_major_dkv
        self.block_k_dkv = block_k_dkv
        self.block_q_dkv = block_q_dkv
        self.block_k_major_dq = block_k_major_dq
        self.block_k_dq = block_k_dq
        self.block_q_dq = block_q_dq
        self.num_attention_heads = num_attention_heads
        self.head_dims = head_dims
        self.sm_scale = sm_scale
        self.mesh = mesh
        self.query_partition_spec = query_partition_spec
        self.key_partition_spec = key_partition_spec
        self.value_partition_spec = value_partition_spec
        self.bias_partition_spec = bias_partition_spec
        self.attention_partition_spec = attention_partition_spec
        self.attention_dropout = attention_dropout
        self.dtype = dtype
        self.precision = precision
        self.force_float32_tpu = force_float32_tpu
        self.shard_attention_computation = shard_attention_computation
        self.use_sharding_constraint = use_sharding_constraint
        self.scan_ring_attention = scan_ring_attention
        self.scan_attention_layers = scan_attention_layers
        self.generation_query_partition_spec = generation_query_partition_spec
        self.generation_bias_partition_spec = generation_bias_partition_spec
        self.generation_attention_partition_spec = generation_attention_partition_spec
        self.assertion_mkv_err = f"""
query_states, key_states, value_states and bias shapes must be like
query_states Shape : [batch_size, q_seq_len , {self.num_attention_heads=}, {self.head_dims=}]
key_states   Shape : [batch_size, kv_seq_len, {self.num_attention_heads=}, {self.head_dims=}]
value_states Shape : [batch_size, kv_seq_len, {self.num_attention_heads=}, {self.head_dims=}]
bias         Shape : [batch_size, {self.num_attention_heads=}, q_seq_len , kv_seq_len]
    """

    def _qkv_check(
            self,
            query_states: Array,
            key_states: Array,
            value_states: Array,
            query_sequence_length: int,
            key_value_sequence_length: int,
    ):
        ...

    def __call__(
            self,
            query_states: Array,
            key_states: Array,
            value_states: Array,
            query_sequence_length: int,
            key_value_sequence_length: int,
            bias: Optional[Array] = None,
            attention_mask: Optional[Array] = None,
            segment_ids: Optional[Array] = None,
            causal: bool = False,
            deterministic: bool = False,
            dropout_rng: Optional[random.PRNGKey] = None,
            uses_cache: bool = False
    ):
        with self.mesh:
            batch_size = query_states.shape[0]
            assert batch_size == key_states.shape[0] == value_states.shape[0], "Batch Size for q,k,v wont match"
            k_v_req_shape = (
                batch_size,
                key_value_sequence_length,
                self.num_attention_heads,
                self.head_dims
            )
            q_shape = (
                batch_size,
                query_sequence_length,
                self.num_attention_heads,
                self.head_dims
            )
            assert query_states.shape == q_shape, self.assertion_mkv_err + (
                f"\nMiss Match {query_states.shape} and "
                f"required Shape {q_shape}"
            )
            assert key_states.shape == k_v_req_shape, self.assertion_mkv_err + (
                f"\nMiss Match {key_states.shape} and "
                f"required Shape {k_v_req_shape}"
            )
            assert value_states.shape == k_v_req_shape, self.assertion_mkv_err + (
                f"\nMiss Match {value_states.shape} and "
                f"required Shape {k_v_req_shape}"
            )

            if self.attn_mechanism == "flash":

                attentions = self._qkv_normal_flash_op(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    bias=bias,
                    causal=causal,
                    key_value_sequence_length=key_value_sequence_length,
                )

            elif self.attn_mechanism == "normal":

                attentions = self._qkv_normal_op(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    bias=bias,
                    dropout_rng=dropout_rng,
                    deterministic=deterministic,
                )
            elif self.attn_mechanism == "ring":
                attentions = self._qkv_ring_op(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    bias=bias,
                    dropout_rng=dropout_rng,
                    deterministic=deterministic,
                    query_sequence_length=query_sequence_length,
                    segment_ids=segment_ids,
                    attention_mask=attention_mask
                )

            elif self.attn_mechanism == "splash":
                attentions = self._qkv_normal_splash_op(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    segment_ids=segment_ids,
                )
            elif self.attn_mechanism == "cudnn":
                attentions = self._qkv_normal_cudnn_flash_op(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    bias=bias,
                    causal=causal,
                    deterministic=deterministic,
                    key_value_sequence_length=key_value_sequence_length
                )
            else:
                raise ValueError(f"Unknown Attention mechanism of {self.attn_mechanism}")
            return attentions

    def _qkv_ring_op(
            self,
            *,  # it's Kwarg Only
            query_states: Array,
            key_states: Array,
            value_states: Array,
            query_sequence_length: int,
            bias: Optional[Array] = None,
            attention_mask: Optional[Array] = None,
            deterministic: bool = False,
            dropout_rng: Optional[random.PRNGKey] = None,
            segment_ids: Optional[Array] = None
    ):
        if segment_ids is None:
            segment_ids = jnp.zeros((query_states.shape[0], query_sequence_length), dtype="i4")
        if self.scan_ring_attention and query_states.shape[1] > max(
                self.block_q,
                self.block_k
        ):
            if self.platform == "tpu":
                ring_attention_fn = ring_flash_attention_tpu
            else:
                ring_attention_fn = ring_attention
            ring_attention_sharded = shard_map(
                partial(
                    ring_attention_fn,
                    axis_name="sp",
                    float32_logits=True,
                    blockwise_kwargs=dict(
                        deterministic=deterministic,
                        dropout_rng=dropout_rng,
                        attn_pdrop=self.attention_dropout,
                        causal=True,
                        query_chunk_size=self.block_q,
                        key_chunk_size=self.block_k,
                        dtype=self.dtype,
                        policy=get_gradient_checkpoint_policy("nothing_saveable"),
                        precision=self.precision,
                        prevent_cse=not self.scan_attention_layers,
                    )
                ),
                mesh=self.mesh,
                in_specs=(
                    self.query_partition_spec,
                    self.key_partition_spec,
                    self.value_partition_spec,
                    self.bias_partition_spec,
                    PartitionSpec(("dp", "fsdp"), None),
                ),
                out_specs=self.attention_partition_spec,
                check_rep=False
            )
            attn_output = ring_attention_sharded(query_states, key_states, value_states, bias, segment_ids)
            attn_output = with_sharding_constraint(attn_output, self.attention_partition_spec)
        else:
            if self.platform != "tpu":
                warnings.warn(
                    "Using Ring attention on CPUs or GPUs are not recommended due to miss computations at the moment. "
                    "please refer to other types of attention mechanism.your are bing fell back on "
                    "`ring_attention_sharded`"
                    f" Usage conditions was\nscan_ring_attention = {self.scan_ring_attention} [MUST BE TRUE]"
                    f"\nquery_states.shape[1]({query_states.shape[1]}) > max({self.block_q},{self.block_k})"
                    f"({max(self.block_q, self.block_k)})"
                )
            query_sequence_partition = None if query_states.shape[1] == 1 else "sp"
            ring_attention_sharded = shard_map(
                partial(ring_attention_standard, axis_name="sp"),
                mesh=self.mesh,
                in_specs=(
                    PartitionSpec(("dp", "fsdp"), query_sequence_partition, "tp", None),
                    PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                    PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                    PartitionSpec(("dp", "fsdp"), None, query_sequence_partition, None)
                ),
                out_specs=PartitionSpec(("dp", "fsdp"), query_sequence_partition, "tp", None),
                check_rep=False
            )
            attn_output = ring_attention_sharded(
                query_states, key_states, value_states, attention_mask
            )
        return AttentionOutput(
            attention_weights=None,
            attention_outputs=attn_output
        )

    def _qkv_normal_op(
            self,
            *,  # it's Kwarg Only
            query_states: Array,
            key_states: Array,
            value_states: Array,
            bias: Optional[Array] = None,
            deterministic: bool = False,
            dropout_rng: Optional[random.PRNGKey] = None,
    ) -> AttentionOutput:
        dtype = jnp.promote_types(self.dtype, jnp.float32)
        with self.mesh:
            assert key_states.shape[1] == value_states.shape[1], (
                "miss match on key_states and value_states sequence length"
            )
            assert query_states.ndim == key_states.ndim, "q, k must have same rank."
            assert query_states.shape[:-3] == key_states.shape[:-3], "q, k batch dims must match."
            assert query_states.shape[-2] == key_states.shape[-2], "q, k num_heads must match."
            assert query_states.shape[-1] == key_states.shape[-1], "q, k depths must match."

            query_states, key_states, value_states = promote_dtype(query_states, key_states, value_states, dtype=dtype)

            if query_states.shape[1] == 1:
                sequence_sharding_axis_name = None
                tensor_sharding_axis_name = "sp"
            elif query_states.shape[1] != key_states.shape[1]:
                sequence_sharding_axis_name = None
                tensor_sharding_axis_name = None
            else:
                sequence_sharding_axis_name = "sp"
                tensor_sharding_axis_name = "tp"

            if self.shard_attention_computation:
                query_states = with_sharding_constraint(
                    query_states, PartitionSpec(
                        ("dp", "fsdp"),
                        sequence_sharding_axis_name,
                        tensor_sharding_axis_name,
                        None
                    )
                )

                key_states = with_sharding_constraint(
                    key_states, PartitionSpec(
                        ("dp", "fsdp"),
                        sequence_sharding_axis_name,
                        tensor_sharding_axis_name,
                        None
                    )
                )

                value_states = with_sharding_constraint(
                    value_states, PartitionSpec(
                        ("dp", "fsdp"),
                        sequence_sharding_axis_name,
                        tensor_sharding_axis_name,
                        None
                    )
                )

            depth = query_states.shape[-1]
            query_states = query_states / jnp.sqrt(depth).astype(dtype)
            attention_weight = jnp.einsum(
                "...qhd,...khd->...hqk",
                query_states, key_states, precision=self.precision
            )
            if self.shard_attention_computation:
                attention_weight = with_sharding_constraint(
                    attention_weight, PartitionSpec(
                        ("dp", "fsdp"),
                        None,
                        sequence_sharding_axis_name,
                        None
                    )
                )
                if bias is not None:
                    bias = with_sharding_constraint(
                        bias, PartitionSpec(
                            ("dp", "fsdp"),
                            None,
                            sequence_sharding_axis_name,
                            None
                        )
                    )

            if bias is not None:
                attention_weight = attention_weight + bias
            attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
            if not deterministic and self.attention_dropout > 0.0:
                keep_prob = 1.0 - self.attention_dropout
                dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
                keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore

                multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
                attention_weight = attention_weight * multiplier
            attention = jnp.einsum(
                "...hqk,...khd->...qhd",
                attention_weight, value_states, precision=self.precision
            )
            if self.shard_attention_computation:
                attention = with_sharding_constraint(
                    attention,
                    PartitionSpec(("dp", "fsdp"), sequence_sharding_axis_name, tensor_sharding_axis_name, None)
                )
        return AttentionOutput(
            attention_outputs=attention,
            attention_weights=attention_weight
        )

    def _qkv_normal_flash_op(
            self,
            *,  # it's Kwarg Only
            query_states: Array,
            key_states: Array,
            value_states: Array,
            key_value_sequence_length: int,
            bias: Optional[Array] = None,
            causal: bool = False,
    ) -> AttentionOutput:

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        batch_size, num_attention_heads, query_sequence_length, head_dims = query_states.shape
        if bias is not None:
            if bias.shape[1] != num_attention_heads:
                es = bias.shape
                bias = bias.repeat(
                    num_attention_heads, 1,
                )
        assert bias.shape == (
            batch_size,
            self.num_attention_heads,
            query_sequence_length,
            key_value_sequence_length
        ), self.assertion_mkv_err
        flash_func, float32_logits, _ = get_flash_attention()
        if float32_logits:
            query_states, key_states, value_states = map(
                lambda s: s.astype(jnp.float32),
                (query_states, key_states, value_states)
            )
        is_generating = query_states.shape[2] == 1
        query_sequence_partition = self.generation_query_partition_spec if is_generating else self.query_partition_spec
        bias_partition_spec = self.generation_bias_partition_spec if is_generating else self.bias_partition_spec
        block_q = 1 if is_generating else self.block_q
        block_q_major_dkv = 1 if is_generating else self.block_q_major_dkv
        block_q_dkv = 1 if is_generating else self.block_q_dkv
        block_q_dq = 1 if is_generating else self.block_q_dq
        if self.sm_scale is None:
            self.sm_scale = 1 / math.sqrt(query_states[-1])
        attention_o = shard_map(
            partial(
                flash_func,
                causal=causal,
                sm_scale=self.sm_scale,
                block_sizes=flash_attn_tpu.BlockSizes(
                    block_b=self.block_b,
                    block_k=self.block_k,
                    block_q=block_q,
                    block_k_major=self.block_k_major,
                    block_k_dq=self.block_k_dq,
                    block_q_dq=block_q_dq,
                    block_k_dkv=self.block_k_dkv,
                    block_q_dkv=block_q_dkv,
                    block_k_major_dq=self.block_k_major_dq,
                    block_k_major_dkv=self.block_k_major_dkv,
                    block_q_major_dkv=block_q_major_dkv,
                ),
                debug=False
            ),
            in_specs=(
                query_sequence_partition,
                self.key_partition_spec,
                self.value_partition_spec,
                bias_partition_spec
            ),
            out_specs=(
                self.attention_partition_spec
            ),
            mesh=self.mesh,
            check_rep=False,
        )(
            query_states,
            key_states,
            value_states,
            bias,
        )

        attention_o = attention_o.transpose(0, 2, 1, 3)
        return AttentionOutput(
            attention_outputs=attention_o,
            attention_weights=None
        )

    def _qkv_normal_splash_op(
            self,
            query_states: Array,
            key_states: Array,
            value_states: Array,
            segment_ids: Optional[Array] = None,
    ) -> AttentionOutput:

        if self.platform != "tpu":
            raise OSError(f"Splash attention only supports on TPUs and right now we don't support {self.platform}")

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        query_states, key_states, value_states = map(
            lambda s: s.astype(jnp.float32),
            (query_states, key_states, value_states)
        )
        is_generating = query_states.shape[2] == 1
        query_sequence_partition = self.generation_query_partition_spec if is_generating else self.query_partition_spec
        bias_partition_spec = self.generation_bias_partition_spec if is_generating else self.bias_partition_spec
        block_q = 1 if is_generating else self.block_q
        block_q_major_dkv = 1 if is_generating else self.block_q_major_dkv
        block_q_dkv = 1 if is_generating else self.block_q_dkv
        block_q_dq = 1 if is_generating else self.block_q_dq

        def sfa(
                in_query_states,
                in_key_states,
                in_value_states,
                in_segment_ids,
        ):
            block_sizes = flash_attn_tpu.BlockSizes(
                block_b=self.block_b,
                block_k=self.block_k,
                block_q=block_q,
                block_k_major=self.block_k_major,
                block_k_dq=self.block_k_dq,
                block_q_dq=block_q_dq,
                block_k_dkv=self.block_k_dkv,
                block_q_dkv=block_q_dkv,
                block_k_major_dq=self.block_k_major_dq,
                block_k_major_dkv=self.block_k_major_dkv,
                block_q_major_dkv=block_q_major_dkv,
            )
            multi_head_mask = MultiHeadMask(
                masks=[
                    CausalMask(
                        shape=(in_query_states.shape[2], in_query_states.shape[2])
                    ) for _ in range(
                        in_query_states.shape[1]
                    )
                ]
            )
            splash_kernel = make_splash_mha(
                mask=multi_head_mask,
                head_shards=1,
                q_seq_shards=1,
                block_sizes=block_sizes
            )

            return jax.vmap(splash_kernel)(
                in_query_states,
                in_key_states,
                in_value_states,
                segment_ids=in_segment_ids
            )

        attention_o = shard_map(
            sfa,
            in_specs=(
                query_sequence_partition,
                self.key_partition_spec,
                self.value_partition_spec,
                bias_partition_spec
            ),
            out_specs=(
                self.attention_partition_spec
            ),
            mesh=self.mesh,
            check_rep=False,
        )(
            query_states,
            key_states,
            value_states,
            segment_ids,
        )

        attention_o = attention_o.transpose(0, 2, 1, 3)
        return AttentionOutput(
            attention_outputs=attention_o,
            attention_weights=None
        )

    def _qkv_normal_cudnn_flash_op(
            self,
            *,  # it's Kwarg Only
            query_states: Array,
            key_states: Array,
            value_states: Array,
            key_value_sequence_length: int,
            bias: Optional[Array] = None,
            causal: bool = False,
            deterministic: bool = True
    ) -> AttentionOutput:
        """CUDNN Flash Attention with Transformer Engine."""
        try:
            import transformer_engine.jax.fused_attn as fused_attn
            from transformer_engine.jax.fused_attn import AttnBiasType, AttnMaskType, QKVLayout
            from transformer_engine.jax.fused_attn import is_fused_attn_kernel_available
        except (ModuleNotFoundError, ImportError) as err:
            raise RuntimeError(
                "Please install transformer_engine first. you can install that by running "
                f"`pip install git+https://github.com/NVIDIA/transformer_engine`"
                f"\nhere's extra information on error\n{err}"
            )
        batch, query_sequence_length, num_attention_heads, head_dim = query_states.shape

        qkv_layout = QKVLayout.BS3HD
        attn_mask_type = AttnMaskType.CAUSAL_MASK
        attn_bias_type = AttnBiasType.NO_BIAS

        if self.sm_scale is None:
            self.sm_scale = 1 / math.sqrt(head_dim)
        has_fused_attn_kernel = is_fused_attn_kernel_available(
            self.dtype, self.dtype, qkv_layout,
            attn_bias_type,
            attn_mask_type,
            self.attention_dropout,
            self.num_attention_heads,
            key_states.shape[2],
            query_sequence_length,
            key_value_sequence_length,
            head_dim
        )

        if not has_fused_attn_kernel:
            raise ValueError(
                "Flash attention kernel is not supported for current requested arrays"
                " for details check this repo https://github.com/NVIDIA/TransformerEngine/"
            )

        return AttentionOutput(
            attention_weights=None,
            attention_outputs=fused_attn.self_fused_attn(
                qkv=jnp.concatenate(
                    (
                        jnp.reshape(query_states, (*query_states.shape[:2], 1, *query_states.shape[-2:])),
                        jnp.reshape(key_states, (*query_states.shape[:2], 1, *query_states.shape[-2:])),
                        jnp.reshape(value_states, (*query_states.shape[:2], 1, *query_states.shape[-2:]))
                    ),
                    axis=2
                ),
                bias=bias,
                mask=jnp.zeros((batch, 1, query_sequence_length, key_value_sequence_length)) if causal else None,
                seed=None,
                attn_bias_type=attn_bias_type,
                attn_mask_type=attn_mask_type,
                scaling_factor=self.sm_scale,
                dropout_probability=self.attention_dropout,
                is_training=deterministic
            )
        )
