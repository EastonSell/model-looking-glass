"""A planned reservation preserves every proposed input and identity."""
import importlib
import importlib.util


def test_all_planned_records_keep_the_input_fields():
    name = 'scripts.build_distilbert_antonym_exposures_v1'
    assert importlib.util.find_spec(name) is not None, 'antonym reservation builder is not implemented'
    module = importlib.import_module(name)
    rows = [{'id': f'case-{i}', 'object': None, 'role': 'locality',
             'input_ids': [101, i + 110, 103, 102], 'attention_mask': [1, 1, 1, 1],
             'mask_index': 2} for i in range(55)]
    result = module.project(rows)
    assert len(result) == 55
    for row, record in zip(rows, result):
        assert record['case_id'] == row['id'] and record['status'] == 'planned'
        assert all(record[key] == row[key] for key in ('object', 'role', 'input_ids', 'attention_mask', 'mask_index'))
