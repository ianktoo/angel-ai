"""Model/tokenizer/LoRA setup shared by training and evaluation."""

from __future__ import annotations

from typing import Any

from angel_ai.backends.base import Backend
from angel_ai.errors import UnsupportedConfigError

# Minimal ChatML-style fallback for models/tokenizers that don't ship a
# chat_template (common on small "tiny-random-*" test checkpoints and some
# base, non-instruct models). Real instruct models (Qwen2.5-Instruct, etc.)
# already define their own and this is never used for them.
_FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def load_tokenizer(model_cfg: Any):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name, trust_remote_code=model_cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE
    return tokenizer


def load_base_model(model_cfg: Any, backend: Backend, *, attn_implementation: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, model_cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        torch_dtype=dtype,
        trust_remote_code=model_cfg.trust_remote_code,
        attn_implementation=attn_implementation,
    )
    return backend.prepare_model(model, model_cfg)


def apply_lora(model, training_cfg: Any, backend: Backend):
    from peft import LoraConfig, get_peft_model

    quantization = training_cfg.get("quantization")
    if quantization and not backend.supports_quantization(quantization):
        raise UnsupportedConfigError(
            f"Backend '{backend.name}' does not support quantization scheme "
            f"'{quantization}'.",
            field="training.quantization",
            suggestion=(
                "Use training=lora (no quantization) on this backend, or switch "
                "to backend=cuda for QLoRA."
            ),
        )

    lora_config = LoraConfig(
        r=training_cfg.lora.r,
        lora_alpha=training_cfg.lora.alpha,
        lora_dropout=training_cfg.lora.dropout,
        target_modules=list(training_cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if training_cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    return model
