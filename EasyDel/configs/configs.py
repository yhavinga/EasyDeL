llama_configs = {
    '3b': {
        'vocab_size': 32000,
        'hidden_size': 3200,
        'intermediate_size': 8640,
        'num_hidden_layers': 26,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '7b': {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '13b': {
        'vocab_size': 32000,
        'hidden_size': 5120,
        'intermediate_size': 13824,
        'num_hidden_layers': 40,
        'num_attention_heads': 40,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '30b': {
        'vocab_size': 32000,
        'hidden_size': 6656,
        'intermediate_size': 17920,
        'num_hidden_layers': 60,
        'num_attention_heads': 52,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '65b': {
        'vocab_size': 32000,
        'hidden_size': 8192,
        'intermediate_size': 22016,
        'num_hidden_layers': 80,
        'num_attention_heads': 64,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
    }
}

mpt_configs = {
    "1b": {
        "alibi": True,
        "alibi_bias_max": 8,
        "attn_clip_qkv": None,
        "attn_impl": "torch",
        "attn_pdrop": 0,
        "attn_qk_ln": True,
        "attn_uses_sequence_id": False,
        "d_model": 2048,
        "emb_init_std": None,
        "emb_init_uniform_lim": None,
        "emb_pdrop": 0,
        "embedding_fraction": 1.0,
        "fan_mode": "fan_in",
        "init_device": "cpu",
        "init_div_is_residual": True,
        "init_gain": 0,
        "init_nonlinearity": "relu",
        "init_std": 0.02,
        "logit_scale": None,
        "low_precision_layernorm": True,
        "max_seq_len": 2048,
        "mlp_ratio": 4,
        "model_type": "mosaic_gpt",
        "n_heads": 16,
        "n_layers": 24,
        "no_bias": True,
        "param_init_fn": "kaiming_normal_",
        "prefix_lm": False,
        "resid_pdrop": 0,
        "softmax_scale": None,
        "tokenizer_name": "EleutherAI/gpt-neox-20b",
        "torch_dtype": "float16",
        "use_cache": False,
        "verbose": 0,
        "vocab_size": 50432
    },
    "7b": {
        "act_fn": "gelu",
        "alibi": True,
        "d_model": 4096,
        "emb_prob_drop": 0.0,
        "embedding_fraction": 1.0,
        "expansion_ratio": 4,
        "learned_pos_emb": True,
        "logit_scale": None,
        "max_seq_len": 2048,
        "model_type": "mpt",
        "n_heads": 32,
        "n_layers": 32,
        "no_bias": True,
        "qk_ln": False,
        "resid_prob_drop": 0.0,
        "use_bias": False,
        "use_cache": False,
        "use_lm_head": False,
        "use_norm_bias": False,
        "verbose": 0,
        "vocab_size": 50432
    },
    "30b": {
        "act_fn": "gelu",
        "alibi": True,
        "d_model": 7168,
        "emb_prob_drop": 0.0,
        "embedding_fraction": 1.0,
        "expansion_ratio": 4,
        "learned_pos_emb": True,
        "logit_scale": None,
        "max_seq_len": 8192,
        "model_type": "mpt",
        "n_heads": 64,
        "n_layers": 48,
        "no_bias": True,
        "qk_ln": False,
        "resid_prob_drop": 0.0,
        "use_bias": False,
        "use_cache": False,
        "use_lm_head": False,
        "use_norm_bias": False,
        "verbose": 0,
        "vocab_size": 50432
    }
}

gptj_configs = {
    '6b': {
        "vocab_size": 50400,
        "n_positions": 2048,
        "n_embd": 4096,
        "n_layer": 28,
        "n_head": 16,
        "rotary_dim": 64,
        "n_inner": None,
        "activation_function": "gelu_new",
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "scale_attn_weights": True,
        "use_cache": True,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "tie_word_embeddings": False,
        "n_real_tokens": 50257,
    }
}

falcon_configs = {
    '7b': {
        "alibi": False,
        "apply_residual_connection_post_layernorm": False,
        "attention_dropout": 0.0,
        "bias": False,
        "bos_token_id": 11,
        "eos_token_id": 11,
        "hidden_dropout": 0.0,
        "hidden_size": 4544,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "max_seq_len": 2048,
        "model_type": "falcon",
        "multi_query": True,
        "n_head": 71,
        "n_layer": 32,
        "parallel_attn": True,
        "use_cache": False,
        "vocab_size": 65024
    },
    '40b': {
        "bias": False,
        "bos_token_id": 11,
        "eos_token_id": 11,
        "hidden_dropout": 0.0,
        "hidden_size": 8192,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "RefinedWeb",
        "n_head": 128,
        "n_head_kv": 8,
        "n_layer": 60,
        "parallel_attn": True,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 65024
    }
}
