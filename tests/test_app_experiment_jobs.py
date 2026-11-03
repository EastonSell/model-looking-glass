"""Job cancellation is a targeted signal, never a model interruption."""
import importlib
import re
import threading

import pytest


CAPTURE = "c" * 32


def registry_class():
    return importlib.import_module("looking_glass.app_experiment_jobs").ExperimentJobs


def test_reserved_job_is_claimed_once_cancelled_and_released():
    jobs = registry_class()()
    reservation = jobs.reserve("sensitivity", CAPTURE)
    job_id = reservation["job_id"]
    assert re.fullmatch(r"[0-9a-f]{48}", job_id)
    assert reservation["status"] == "reserved"
    lease = jobs.claim(job_id, "sensitivity", CAPTURE)
    assert not lease.cancel_event.is_set()
    assert jobs.status(job_id)["status"] == "queued"
    assert jobs.mark_started(lease) is True
    assert jobs.status(job_id)["status"] == "running"
    assert jobs.cancel(job_id)["status"] == "stopping"
    assert lease.cancel_event.is_set()
    assert jobs.release(lease) is True
    with pytest.raises(ValueError):
        jobs.status(job_id)


def test_public_action_set_is_the_allowed_reservation_scope():
    module = importlib.import_module("looking_glass.app_experiment_jobs")
    assert module.EXPERIMENT_ACTIONS == frozenset({"sensitivity", "sensitivity-repeat", "patch-cross-prompt"})


def test_cancel_before_claim_is_observed_by_the_exact_job():
    jobs = registry_class()()
    job_id = jobs.reserve("sensitivity-repeat", CAPTURE)["job_id"]
    assert jobs.cancel(job_id)["status"] == "stopping"
    assert jobs.cancel(job_id)["status"] == "stopping"
    assert jobs.status(job_id)["status"] == "stopping"
    lease = jobs.claim(job_id, "sensitivity-repeat", CAPTURE)
    assert lease.cancel_event.is_set()
    assert jobs.mark_started(lease) is True
    assert jobs.status(job_id)["status"] == "stopping"
    assert jobs.release(lease)


def test_wrong_binding_replay_and_overlapping_reservation_are_rejected():
    jobs = registry_class()()
    job_id = jobs.reserve("sensitivity", CAPTURE)["job_id"]
    with pytest.raises(ValueError): jobs.claim(job_id, "sensitivity-repeat", CAPTURE)
    with pytest.raises(ValueError): jobs.claim(job_id, "sensitivity", "another-capture")
    with pytest.raises(ValueError): jobs.cancel("0" * 48 if job_id != "0" * 48 else "1" * 48)
    with pytest.raises(ValueError): jobs.reserve("patch-cross-prompt", CAPTURE)
    assert jobs.status(job_id)["status"] == "reserved"
    lease = jobs.claim(job_id, "sensitivity", CAPTURE)
    with pytest.raises(ValueError): jobs.claim(job_id, "sensitivity", CAPTURE)
    with pytest.raises(ValueError): jobs.reserve("sensitivity", CAPTURE)
    assert jobs.release(lease)
    with pytest.raises(ValueError): jobs.claim(job_id, "sensitivity", CAPTURE)


def test_unclaimed_expiry_does_not_expire_running_jobs():
    now = [100.0]
    jobs = registry_class()(reservation_ttl=30, clock=lambda: now[0])
    expired = jobs.reserve("sensitivity", CAPTURE)["job_id"]
    now[0] += 30
    with pytest.raises(ValueError): jobs.claim(expired, "sensitivity", CAPTURE)
    current = jobs.reserve("sensitivity", CAPTURE)["job_id"]
    lease = jobs.claim(current, "sensitivity", CAPTURE)
    now[0] += 10000
    assert jobs.status(current)["status"] == "queued"
    assert jobs.mark_started(lease)
    assert jobs.status(current)["status"] == "running"
    assert jobs.cancel(current)["status"] == "stopping"
    assert lease.cancel_event.is_set()
    assert jobs.release(lease)


def test_stale_or_forged_lease_cannot_release_a_new_job():
    module = importlib.import_module("looking_glass.app_experiment_jobs")
    jobs = module.ExperimentJobs()
    old_id = jobs.reserve("sensitivity", CAPTURE)["job_id"]
    old = jobs.claim(old_id, "sensitivity", CAPTURE)
    assert jobs.release(old)
    new_id = jobs.reserve("sensitivity", CAPTURE)["job_id"]
    new = jobs.claim(new_id, "sensitivity", CAPTURE)
    with pytest.raises(ValueError): jobs.cancel(old_id)
    forged = module.ExperimentLease(new_id, "sensitivity", CAPTURE, new.cancel_event)
    assert jobs.mark_started(forged) is False
    assert jobs.mark_started(old) is False
    assert jobs.mark_started(new) is True
    assert jobs.mark_started(new) is False
    assert jobs.release(forged) is False
    assert jobs.release(old) is False
    old.cancel_event.set()
    assert not new.cancel_event.is_set()
    assert jobs.status(new_id)["status"] == "running"
    assert jobs.release(new)


@pytest.mark.parametrize("action,capture", [(None, CAPTURE), ("run", CAPTURE), ("sensitivity", ""),
                                             ("sensitivity", None), ("sensitivity", "x" * 65)])
def test_invalid_reservation_never_occupies_the_slot(action, capture):
    jobs = registry_class()()
    with pytest.raises(ValueError): jobs.reserve(action, capture)
    assert jobs.reserve("sensitivity", CAPTURE)["status"] == "reserved"


@pytest.mark.parametrize("job_id", [None, "", "a" * 47, "A" * 48, "x" * 48, "a" * 49, 1])
def test_invalid_job_ids_are_rejected_without_changing_the_current_job(job_id):
    jobs = registry_class()()
    actual = jobs.reserve("patch-cross-prompt", CAPTURE)["job_id"]
    with pytest.raises(ValueError): jobs.cancel(job_id)
    assert jobs.status(actual)["status"] == "reserved"


def test_concurrent_reservations_have_one_winner():
    jobs = registry_class()()
    barrier = threading.Barrier(8)
    answers, rejected = [], []
    def reserve():
        barrier.wait(timeout=5)
        try: answers.append(jobs.reserve("sensitivity", CAPTURE))
        except ValueError: rejected.append(True)
    threads = [threading.Thread(target=reserve) for _ in range(8)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert len(answers) == 1 and len(rejected) == 7


@pytest.mark.parametrize("_iteration", range(10))
def test_cancel_and_claim_race_signals_exactly_the_claimed_lease(_iteration):
    jobs = registry_class()()
    job_id = jobs.reserve("sensitivity", CAPTURE)["job_id"]
    barrier, leases, outcomes = threading.Barrier(2), [], []
    def claim():
        barrier.wait(timeout=5)
        leases.append(jobs.claim(job_id, "sensitivity", CAPTURE))
    def cancel():
        barrier.wait(timeout=5)
        outcomes.append(jobs.cancel(job_id))
    threads = [threading.Thread(target=claim), threading.Thread(target=cancel)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert len(leases) == len(outcomes) == 1 and leases[0].cancel_event.is_set()
    assert jobs.release(leases[0])


@pytest.mark.parametrize("lifetime", [0, -1, True, float("nan"), float("inf"), 301, "30"])
def test_invalid_reservation_lifetime_is_rejected(lifetime):
    with pytest.raises(ValueError): registry_class()(reservation_ttl=lifetime)


@pytest.mark.parametrize("_iteration", range(5))
def test_concurrent_claims_have_one_winner(_iteration):
    jobs = registry_class()()
    job_id = jobs.reserve("sensitivity", CAPTURE)["job_id"]
    barrier = threading.Barrier(8)
    leases, rejected = [], []
    def claim():
        barrier.wait(timeout=5)
        try: leases.append(jobs.claim(job_id, "sensitivity", CAPTURE))
        except ValueError: rejected.append(True)
    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert len(leases) == 1 and len(rejected) == 7
    assert jobs.status(job_id)["status"] == "queued"
    assert jobs.release(leases[0])


@pytest.mark.parametrize("_iteration", range(10))
def test_cancel_racing_with_completion_cannot_affect_a_later_job(_iteration):
    jobs = registry_class()()
    job_id = jobs.reserve("sensitivity", CAPTURE)["job_id"]
    lease = jobs.claim(job_id, "sensitivity", CAPTURE)
    assert jobs.mark_started(lease)
    barrier, outcomes = threading.Barrier(2), []
    def finish():
        barrier.wait(timeout=5)
        outcomes.append(("released", jobs.release(lease)))
    def cancel():
        barrier.wait(timeout=5)
        try: outcomes.append(("cancel", jobs.cancel(job_id)["status"]))
        except ValueError: outcomes.append(("cancel", "already-finished"))
    threads = [threading.Thread(target=finish), threading.Thread(target=cancel)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert ("released", True) in outcomes
    assert ("cancel", "stopping") in outcomes or ("cancel", "already-finished") in outcomes
    next_id = jobs.reserve("sensitivity", CAPTURE)["job_id"]
    next_lease = jobs.claim(next_id, "sensitivity", CAPTURE)
    assert not next_lease.cancel_event.is_set()
    assert jobs.release(next_lease)
