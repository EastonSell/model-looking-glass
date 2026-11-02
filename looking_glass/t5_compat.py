"""Validate T5 checkpoint semantics across Transformers config migrations.

Transformers 5.22 records decoder scaling separately and forces the config tie
flag true, even when its loader correctly retains a distinct checkpoint head.
Never repair or replace weights here: verify their identities before normalizing
the declarative flag.
"""


def validate_t5_checkpoint_contract(model, raw):
    from transformers import T5ForConditionalGeneration
    if type(model) is not T5ForConditionalGeneration or type(raw) is not dict:
        raise ValueError("expected standard T5 model and original configuration")
    tied = raw.get("tie_word_embeddings", True)
    scale = raw.get("scale_decoder_outputs", tied)
    if type(tied) is not bool or type(scale) is not bool:
        raise ValueError("T5 embedding and decoder-scaling flags must be booleans")
    if (model.encoder.embed_tokens.weight is not model.shared.weight
            or model.decoder.embed_tokens.weight is not model.shared.weight
            or (model.lm_head.weight is model.shared.weight) != tied
            or model.config.scale_decoder_outputs is not scale):
        raise ValueError("loaded T5 embedding identities or decoder scaling differ from checkpoint")
    model.config.tie_word_embeddings = tied
    return model
