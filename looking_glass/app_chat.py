"""Bounded cached text generation without activation tracing or persistence."""
import inspect
import torch

MAX_CONTEXT = 32768
MAX_OUTPUT = 1024
MAX_MESSAGES = 1025
MAX_CHARACTERS = 1024 * 1024


def context_limit(model):
    config = model.config
    value = getattr(config, 'max_position_embeddings', None) or getattr(config, 'n_positions', None)
    if type(value) is not int or value < 2:
        raise ValueError('Chat requires a known positive model context limit')
    return min(value, MAX_CONTEXT)


def default_context(model, device):
    return min(context_limit(model), 8192 if device.type == 'cuda' else 4096)


def _positive(value, name, maximum):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f'{name} must be a positive integer <= {maximum}')
    return value


def prepare_chat(model, tokenizer, messages, settings, device):
    if type(settings) is not dict or set(settings) - {'max_new_tokens', 'context_tokens'}:
        raise ValueError('Chat settings must contain only max_new_tokens and context_tokens')
    context = _positive(settings.get('context_tokens', default_context(model, device)),
        'context_tokens', context_limit(model))
    # Tiny test/demo models cannot reserve a normal 512-token reply.
    output = _positive(settings.get('max_new_tokens', min(512, max(1, context // 4))),
        'max_new_tokens', MAX_OUTPUT)
    if output >= context:
        raise ValueError('Reply allowance must leave room for input within the model context')
    if type(messages) is not list or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError('Chat requires 1 to 1025 messages')
    total = 0
    for item in messages:
        if (type(item) is not dict or set(item) != {'role', 'content'}
                or type(item['role']) is not str or item['role'] not in ('system', 'user', 'assistant')
                or type(item['content']) is not str
                or (item['role'] != 'assistant' and not item['content'].strip())):
            raise ValueError('Each message must contain a supported role and nonempty content only')
        total += len(item['content'])
    if total > MAX_CHARACTERS:
        raise ValueError('Chat messages exceed the one-million-character request limit')
    start = int(messages[0]['role'] == 'system')
    conversation = messages[start:]
    if (not conversation or len(conversation) % 2 != 1
            or any(item['role'] != ('user' if i % 2 == 0 else 'assistant')
                   for i, item in enumerate(conversation))):
        raise ValueError('Chat must alternate user/assistant messages and end with the latest user message')
    template = bool(getattr(tokenizer, 'chat_template', None))
    if not template and (start or len(messages) != 1):
        raise ValueError('This tokenizer has no chat template. Use a single raw prompt or load an Instruct model with its matching chat template.')
    retained = [dict(item) for item in messages]
    dropped = 0
    while True:
        if template:
            # Render once, then tokenize with no extra BOS/EOS inserted.
            prompt = tokenizer.apply_chat_template(retained, tokenize=False, add_generation_prompt=True)
            if type(prompt) is not str or len(prompt) > 2 * MAX_CHARACTERS:
                raise ValueError('Rendered chat template exceeds the bounded prompt limit')
        else:
            prompt = retained[0]['content']
        encoded = tokenizer(prompt, return_tensors='pt', truncation=False,
            add_special_tokens=not template, return_attention_mask=True)
        ids = encoded['input_ids']
        if ids.ndim != 2 or ids.shape[0] != 1 or not ids.numel():
            raise ValueError('Tokenizer must produce one nonempty input sequence')
        if bool((ids < 0).any()) or bool((ids >= model.config.vocab_size).any()):
            raise ValueError('Tokenizer IDs exceed checkpoint vocabulary')
        if ids.shape[1] + output <= context:
            break
        if not template or len(retained) - start <= 1:
            raise ValueError('The system and latest user message exceed the context allowance; shorten them or raise the context limit. They were not truncated.')
        del retained[start:start + 2]  # oldest whole user + assistant exchange
        dropped += 2
    return {'input_ids': ids.to(torch.int64),
        'attention_mask': encoded.get('attention_mask', torch.ones_like(ids)).to(torch.int64),
        'max_new_tokens': output, 'context_tokens': context, 'dropped_messages': dropped,
        'format': 'chat_template' if template else 'raw_completion'}


def estimated_chat_bytes(model, prepared):
    """KV cache plus conservative visible workspace, not an exact allocator peak."""
    config = model.config
    context = prepared['context_tokens']
    width = getattr(config, 'hidden_size', None) or config.n_embd
    heads = getattr(config, 'num_attention_heads', None) or config.n_head
    layers = getattr(config, 'num_hidden_layers', None) or config.n_layer
    kv_heads = getattr(config, 'num_key_value_heads', heads)
    head_dim = getattr(config, 'head_dim', width // heads)
    size = next(model.parameters()).element_size()
    kv = 2 * layers * kv_heads * head_dim * context * size
    # Single response logits where supported. Legacy models can emit full prompt logits.
    keep_last = 'logits_to_keep' in inspect.signature(model.forward).parameters
    logit_rows = 1 if keep_last else int(prepared['input_ids'].shape[1])
    logits = logit_rows * config.vocab_size * 4
    intermediate = getattr(config, 'intermediate_size', None) or getattr(config, 'n_inner', None) or 4 * width
    workspace = context * (6 * intermediate + 8 * width) * max(size, 4)
    # SDPA may choose a math kernel on a given GPU, so reserve quadratic attention too.
    attention = 3 * heads * int(prepared['input_ids'].shape[1])**2 * max(size, 4)
    return kv + logits + workspace + attention + 64 * 1024**2


def generate_chat(model, tokenizer, prepared, device):
    ids = prepared['input_ids'].to(device)
    mask = prepared['attention_mask'].to(device)
    from transformers import GenerationConfig
    # Never inherit checkpoint beams, static caches, minimum lengths, processors,
    # returned scores, or remote custom-generation settings. Only special IDs carry over.
    special = {}
    for name in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        value = getattr(model.generation_config, name, None)
        values = value if type(value) is list else [value]
        if value is not None and (not values or any(type(token) is not int or not 0 <= token < model.config.vocab_size for token in values)):
            raise ValueError('Generation special token IDs exceed checkpoint vocabulary')
        special[name] = value
    if tokenizer.pad_token_id is not None:
        special['pad_token_id'] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        special['pad_token_id'] = tokenizer.eos_token_id
    policy = GenerationConfig(**special, do_sample=False, num_beams=1,
        num_return_sequences=1, use_cache=True, cache_implementation='dynamic',
        max_new_tokens=prepared['max_new_tokens'], min_length=0, min_new_tokens=0,
        return_dict_in_generate=False, output_scores=False, output_logits=False,
        output_attentions=False, output_hidden_states=False)
    kwargs = dict(input_ids=ids, attention_mask=mask, generation_config=policy)
    if 'logits_to_keep' in inspect.signature(model.forward).parameters:
        kwargs['logits_to_keep'] = 1
    # Transformers fills unspecified policy fields from model.generation_config.
    # The exclusive runtime lock permits a temporary clean default as well; restore
    # the original object even on failure, without editing checkpoint files.
    previous_policy = model.generation_config
    try:
        model.generation_config = policy
        with torch.inference_mode():
            output = model.generate(**kwargs)
    finally:
        model.generation_config = previous_policy
    new_ids = output[0, ids.shape[1]:].detach().cpu().tolist()
    return {'answer': tokenizer.decode(new_ids, skip_special_tokens=True),
        'generated_token_ids': new_ids, 'input_tokens': int(ids.shape[1]), 'output_tokens': len(new_ids),
        'context_tokens': prepared['context_tokens'], 'dropped_messages': prepared['dropped_messages'],
        'format': prepared['format'], 'device': str(device),
        'dtype': str(next(model.parameters()).dtype).removeprefix('torch.'),
        'note': 'Cached generation; no activation capture. History is supplied each turn; KV cache is released after each response. Older complete turns may be dropped to fit the displayed context; system/latest-user messages are never silently truncated.'}
