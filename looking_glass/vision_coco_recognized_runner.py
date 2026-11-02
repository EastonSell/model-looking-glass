"""Bounded clean screening with byte-first retained model outputs."""
from __future__ import annotations

import torch

from .vision_coco_recognized_screen import check_rounding, screen_score, select_pairs
from .vision_coco_recognized_handoff import validate_candidates
from .vision_coco_study import _bits, check_denormals


def run_clean_screen(model, images, pixels, evidence):
    """Screen each fixed image once, then pair only from retained logits."""
    validate_candidates(images)
    if set(pixels) != {row['image_id'] for row in images}:
        raise ValueError('screen candidate identity/order differs')
    observed = []
    for image in images:
        image_id = image['image_id']
        label = f'screen:{image_id}'
        evidence.reserve(label)
        check_denormals()
        check_rounding()
        value = pixels[image_id].detach().clone()
        input_bytes = _bits(value)
        output, saved = None, False
        try:
            with torch.no_grad():
                output = model(pixel_values=value)
            logits = output.logits.detach().cpu().contiguous()
            row = evidence.save_forward_bytes(label, {'logits': logits})
            saved = True
            evidence.active['phase'] = 'validation'
            check_denormals()
            check_rounding()
            if tuple(logits.shape) != (1, 1000):
                raise ValueError('screen logit shape differs')
            result = screen_score(_bits(logits), image['category_id'])
            if _bits(value) != input_bytes:
                raise ValueError('screen model changed input pixels')
            evidence.finish_forward(row, {'kind': 'screen', 'image_id': image_id}, result)
            observed.append({'image_id': image_id, **result})
        except BaseException as original:
            if not saved and output is not None:
                try:
                    value = output.logits
                    if isinstance(value, torch.Tensor) and value.dtype == torch.float32 and 0 < value.numel() <= 1000:
                        evidence.save_forward_bytes(label, {'logits': value.detach().cpu()})
                except BaseException as salvage_error:
                    if isinstance(original, Exception) and not isinstance(salvage_error, Exception):
                        raise
            evidence.active['return_observed'] = output is not None
            raise
    pairs, summary = select_pairs(images, observed)
    return observed, pairs, summary
