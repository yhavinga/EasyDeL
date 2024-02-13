import copy

import os


os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import jax

from jax import numpy as jnp
from transformers import MistralForCausalLM
import torch
import numpy as np

np.set_printoptions(edgeitems=3, precision=8, linewidth=175)

try:
    from lib.python.EasyDel import MistralConfig, FlaxMistralForCausalLM
    from lib.python.EasyDel.transform import (
        mistral_convert_hf_to_flax,
        easystate_to_huggingface_model,
    )
    from lib.python.EasyDel import AutoEasyDelConfig
    from lib.python.EasyDel import EasyDelState
    from lib.python.EasyDel.etils.auto_tx import get_optimizer_and_scheduler

except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().joinpath("..").__str__()
    sys.path.append(cp)
    from lib.python.EasyDel import MistralConfig, FlaxMistralForCausalLM
    from lib.python.EasyDel.transform import (
        mistral_convert_hf_to_flax,
        easystate_to_huggingface_model,
    )
    from lib.python.EasyDel import AutoEasyDelConfig
    from lib.python.EasyDel import EasyDelState
    from lib.python.EasyDel.etils.auto_tx import get_optimizer_and_scheduler


def test_dtype_conversion(jax_dtype: jnp.dtype):
    print(f"================== {jax_dtype} conversions ==================")
    dtype_mapping = {
        jnp.float16: torch.float16,
        jnp.float32: torch.float32,
        jnp.bfloat16: torch.bfloat16,
        # Add more dtypes as needed
    }
    torch_dtype = dtype_mapping.get(jax_dtype)

    torch.manual_seed(42)
    seq_len = 128
    config = MistralConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=4,
        intermediate_size=384,
        gradient_checkpointing="",
        max_position_embeddings=seq_len,
        use_bfloat16=True if torch_dtype == torch.bfloat16 else False,
        torch_dtype=torch_dtype,
    )
    batch_size = len(jax.devices())

    # We initialize a random HF Mistral torch model
    hf_torch_model_1 = MistralForCausalLM(config=copy.deepcopy(config))
    if torch_dtype == torch.bfloat16:
        hf_torch_model_1.bfloat16()  # hf model is not bfloat16 by default
    elif torch_dtype == torch.float16:
        hf_torch_model_1.half()
    print("Convert HF Torch model to params")
    params = {
        "params": mistral_convert_hf_to_flax(
            hf_torch_model_1.state_dict(), config, jax.devices("cpu")[0]
        )
    }

    # Initialize a flax model
    tx_init = dict(optimizer="adamw", scheduler="none", learning_rate=1e-5, steps=5000)
    flax_model = FlaxMistralForCausalLM(
        config=config,
        dtype=jax_dtype,
        param_dtype=jax_dtype,
        _do_init=False,
        input_shape=(batch_size, seq_len),
    )

    print("Initialize EasyState with config and params")
    state = EasyDelState.create(
        module_config=config,
        params=params,
        tx_init=tx_init,
        apply_fn=flax_model.__call__,
        tx=get_optimizer_and_scheduler(**tx_init)[0],
        hyperparameters=EasyDelState.create_hyperparameters(
            model_type=config.model_type
        ),
        module=flax_model,
        module_config_args=None,
    )

    # Save flax state with parameters
    state.save_state(filename="state.easy", verbose=True)

    print("Convert EasyState to HF Torch model")
    hf_torch_model_2 = easystate_to_huggingface_model(
        state=EasyDelState.load_state(
            "state.easy", init_optimizer_state=False, verbose=True
        ),
        base_huggingface_module=MistralForCausalLM,
        base_huggingface_module_kwarguments={},
        dtype=jax_dtype,
        config=config,
    )

    # Model type checks
    if torch_dtype == torch.bfloat16:
        hf_torch_model_2.bfloat16()  # hf model is not bfloat16 by default
    elif torch_dtype == torch.float16:
        hf_torch_model_2.half()

    assert hf_torch_model_2.config.torch_dtype == torch_dtype
    assert hf_torch_model_2.config.use_bfloat16 == (torch_dtype == torch.bfloat16)
    for name, param in hf_torch_model_1.named_parameters():
        assert "norm" in name or param.dtype == torch_dtype
    for name, param in hf_torch_model_2.named_parameters():
        assert "norm" in name or param.dtype == torch_dtype

    print("Generate random input_ids and compare predictions")
    np.random.seed(42)
    np_random_input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    input_ids = (
        torch.from_numpy(np_random_input_ids).reshape(batch_size, -1).to(torch.long)
    )
    flax_input_ids = jnp.asarray(np_random_input_ids, dtype=jnp.int32).reshape(
        batch_size, -1
    )

    hf_model_1_output = hf_torch_model_1(input_ids=input_ids)
    hf_model_2_output = hf_torch_model_2(input_ids=input_ids)
    if not torch.allclose(
        hf_model_1_output.logits, hf_model_2_output.logits, rtol=1e-4, atol=1e-6
    ):
        print(f"Logits between torch HF models do not match for dtype {torch_dtype}")
        error = torch.mean(
            hf_model_1_output.logits.cpu().detach().numpy()
            - hf_model_2_output.logits.cpu().detach().numpy()
        )
        print(f"Mean Error : {error}")

    config.add_jax_args()
    config.add_basic_configurations(use_shard_map=True)
    flax_output = flax_model(
        input_ids=flax_input_ids,
        params=params,
    )
    res = jnp.allclose(
        hf_model_1_output.logits.cpu().detach().numpy(),
        flax_output.logits,
        rtol=1e-1,
        atol=1e-1,
    )
    if not res:
        print(
            f"Logits between flax and torch HF models do not match for dtype {jax_dtype}"
        )
    error = jnp.mean(
        hf_model_1_output.logits.cpu().detach().numpy() - flax_output.logits
    )
    print(f"Mean Error : {error}")
    print(f"Flax {jax_dtype=} Predictions :\n", flax_output.logits)
    print(f"Torch {torch_dtype=} Predictions :\n", hf_model_1_output.logits.cpu().detach().numpy())


if __name__ == "__main__":
    test_dtype_conversion(jnp.float32)
    test_dtype_conversion(jnp.bfloat16)
    # test_dtype_conversion(jnp.float16)  not on cpu
