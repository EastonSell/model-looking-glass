"""Model-call and strict generation rules for the FLAN row-edit study."""

import math
import unicodedata

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

import flan_weight_edit_numerical as science


def numeric_probe():
    tiny = torch.tensor([2.0 ** -149], dtype=torch.float32)
    bits = int((tiny * 1.0).view(torch.int32)[0])
    midpoint_even = float(np.float32(1.0 + 2.0 ** -24))
    midpoint_odd = float(np.float32(1.0 + 3.0 * 2.0 ** -24))
    if bits != 1 or midpoint_even != 1.0 or midpoint_odd != 1.0 + 2.0 ** -22:
        raise ValueError("float32 denormal or rounding mode differs")
    return {"subnormal_bits": bits, "midpoint_even": midpoint_even, "midpoint_odd": midpoint_odd}


def _checked(journal, stage, function, *, context=None):
    try:
        context = {} if context is None else context
        journal.record("numeric_probe", {**context, "stage": stage, "when": "before", **numeric_probe()})
        result = function()
        journal.record("numeric_probe", {**context, "stage": stage, "when": "after", **numeric_probe()})
        return result
    except BaseException:
        journal.fence()
        raise


def _score_terms(logits, tokens):
    values = logits.detach().numpy()
    if values.ndim != 2 or values.shape[0] != len(tokens) or not np.isfinite(values).all():
        raise ValueError("teacher-forced logits or target length differ")
    result = []
    for row, token in zip(values, tokens):
        if type(token) is not int or not 0 <= token < len(row):
            raise ValueError("teacher-forced token is out of range")
        maximum = float(np.max(row))
        logsum = math.log(math.fsum(math.exp(float(value) - maximum) for value in row))
        result.append((float(row[token]), maximum, logsum))
    return result


def _normalized(text):
    return unicodedata.normalize("NFC", text).strip().casefold()


def resource_after(journal, gate):
    try:
        gate("after")
    except BaseException:
        journal.fence()
        raise


def evaluate_case(model, tokenizer, case, journal, arm, *, labels, validate=None, resource_gate=None):
    if validate is None:
        validate = lambda: None
    if not callable(validate):
        raise ValueError("invalid topology validator")
    if resource_gate is None:
        resource_gate = lambda stage: None
    if not callable(resource_gate):
        raise ValueError("invalid resource gate")
    ids, mask = case["input_ids"], case["attention_mask"]
    if (not ids or len(ids) > 64 or ids[-1] != 1 or 1 in ids[:-1]
            or mask != [1] * len(ids)
            or labels != {"positive": [1465, 1], "negative": [2841, 1]}):
        raise ValueError("case input, mask or fixed labels differ")
    source = torch.tensor([ids], dtype=torch.long)
    attention = torch.tensor([mask], dtype=torch.long)
    meta = {"arm": arm}
    context = {"arm": arm, "case_id": case["id"]}
    def checked_forward(function):
        validate()
        resource_gate("before")
        result = function()
        validate()
        return result
    with torch.inference_mode():
        encoded = _checked(journal, "encoder", lambda:
                           journal.call(case["id"], "encoder",
                                        lambda: checked_forward(lambda: model.encoder(input_ids=source, attention_mask=attention,
                                                              return_dict=True).last_hidden_state),
                                        save_logits=False, metadata=meta), context=context)
        resource_after(journal, resource_gate)
        if (not isinstance(encoded, torch.Tensor) or encoded.device.type != "cpu"
                or encoded.dtype != torch.float32 or encoded.ndim != 3
                or encoded.shape[0] != 1 or encoded.shape[1] != len(ids)
                or not bool(torch.isfinite(encoded).all())):
            raise ValueError("encoder result differs from expected CPU float32 shape")

        def decode(prefix, stage, last_only):
            decoder_ids = torch.tensor([prefix], dtype=torch.long)
            decoder_mask = torch.ones_like(decoder_ids)
            def invoke():
                all_logits = model(encoder_outputs=BaseModelOutput(last_hidden_state=encoded),
                               attention_mask=attention, decoder_input_ids=decoder_ids,
                               decoder_attention_mask=decoder_mask, use_cache=False,
                               return_dict=True).logits
                if (not isinstance(all_logits, torch.Tensor) or all_logits.device.type != "cpu"
                        or all_logits.dtype != torch.float32 or tuple(all_logits.shape) != (1, len(prefix), 32128)
                        or not bool(torch.isfinite(all_logits).all())):
                    raise ValueError("decoder returned nonfinite or invalid full logits")
                logits = all_logits[0]
                return logits[-1].contiguous() if last_only else logits.contiguous()
            result = _checked(journal, stage, lambda:
                              journal.call(case["id"], stage, lambda: checked_forward(invoke), save_logits=True,
                                           metadata={"arm": arm, "decoder_prefix": list(prefix)}), context=context)
            resource_after(journal, resource_gate)
            return result, journal.last_record.copy()

        prefix = [0]
        emitted, tops, greedy_spans = [], [], []
        for step in range(4):
            logits, span = decode(prefix, f"greedy-{step}", True)
            row = logits.detach().numpy()
            tied = np.flatnonzero(row == np.max(row)).tolist()
            token = int(tied[0])
            emitted.append(token)
            tops.append({"representative": token, "tied_ids": tied, "unique": len(tied) == 1})
            greedy_spans.append(span)
            prefix.append(token)
            if token == 1:
                break

        terminated = bool(emitted) and emitted[-1] == 1
        content = emitted[:-1] if terminated else emitted
        decoded = tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        strict_output = (terminated and all(item["unique"] for item in tops)
                         and not any(token in tokenizer.all_special_ids for token in content))
        strict = strict_output and _normalized(decoded) == _normalized(case["expected"])
        normalized = _normalized(decoded)
        strict_label = normalized if strict_output and normalized in labels else None
        terms, spans = {}, {}
        for name, target in labels.items():
            teacher_prefix = [0, *target[:-1]]
            logits, span = decode(teacher_prefix, f"teacher-{name}", False)
            terms[name] = _checked(journal, f"score-{name}", lambda: _score_terms(logits, target), context=context)
            spans[name] = span
        candidate_terms, candidate_span = None, None
        if case["expected"] not in labels:
            target = case["expected_ids"]
            logits, candidate_span = decode([0, *target[:-1]], "teacher-candidate", False)
            candidate_terms = _checked(journal, "score-candidate", lambda: _score_terms(logits, target), context=context)
        margin, negative_score = _checked(journal, "score-summary", lambda:
                                          (science.sequence_effect(terms["negative"], terms["positive"]),
                                           science.sequence_score(terms["negative"])), context=context)
        return {"case_id": case["id"], "arm": arm, "greedy_ids": emitted,
                "decoded": decoded, "terminated": terminated, "strict_correct": bool(strict),
                "strict_label": strict_label,
                "tops": tops, "greedy_spans": greedy_spans, "label_terms": terms,
                "label_spans": spans, "candidate_terms": candidate_terms,
                "candidate_span": candidate_span,
                "margin": margin, "negative_score": negative_score}
