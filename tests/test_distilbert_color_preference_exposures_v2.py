import pytest

from scripts.check_distilbert_color_preference_exposures_v2 import related, source_rows, valid_input


def test_shorter_and_longer_inputs_are_both_rejected_after_sep_removal():
    short = [101, 42, 103, 102]
    longer = [101, 42, 103, 17, 102]
    assert not related(short, longer)
    assert related(short[:-1], longer[:-1])
    assert related(longer[:-1], short[:-1])


@pytest.mark.parametrize('field,value', [
    ('input_ids', [101, True, 103, 102]),
    ('attention_mask', [1, 1, True, 1]),
    ('input_ids', [101, 42, 102, 102]),
])
def test_input_identity_requires_real_integer_ids_and_mask(field, value):
    row = {'input_ids': [101, 42, 103, 102], 'attention_mask': [1, 1, 1, 1]}
    row[field] = value
    with pytest.raises(ValueError):
        valid_input(row)


def test_variant_and_nested_bert_sources_include_all_declared_rows():
    row = {'input_ids': [101, 42, 103, 102], 'attention_mask': [1, 1, 1, 1]}
    other = {'input_ids': [101, 43, 103, 102], 'attention_mask': [1, 1, 1, 1]}
    assert source_rows({'cases': [{'rows': {'b': other, 'a': row}}]}, 'variants') == [row['input_ids'], other['input_ids']]
    assert source_rows({'cases': [{'bert': row}, {'bert': other}]}, 'nested-bert') == [row['input_ids'], other['input_ids']]
