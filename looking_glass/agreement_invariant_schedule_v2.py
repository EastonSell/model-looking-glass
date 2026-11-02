"""Fixed v2 mapping of the independently reviewed invariant v1 action order."""
from . import agreement_invariant_schedule_v1 as original

BASE = original.BASE
SPLITS = original.SPLITS

def _rename(case_id):
    return 'v2_' + case_id if case_id.startswith('discovery_') else case_id

def cases():
    return [(_rename(case_id), arms) for case_id, arms in original.cases()]

def actions():
    return [(_rename(case_id), action) for case_id, action in original.actions()]
