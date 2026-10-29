"""A deterministic, bounded-by-policy queue for independent teams."""

from collections import OrderedDict, deque
from dataclasses import dataclass
from queue import Empty, Full
from threading import Lock
from typing import Any, Hashable


class QueueFull(Full):
    """A global or per-team pending-job limit has been reached."""


class QueueEmpty(Empty):
    """No job is pending."""


def _claim_id(value):
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError('claim_id must be a nonempty plain string of at most 128 characters')


@dataclass(frozen=True)
class QueueItem:
    team_id: Hashable
    job_id: Hashable
    payload: Any
    compatibility_key: bytes | None = None


class FairTeamQueue:
    """Nonblocking per-team FIFO, with round-robin dispatch among active teams."""

    def __init__(self, max_pending: int, max_pending_per_team: int):
        for value in (max_pending, max_pending_per_team):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('queue capacities must be positive integers')
        self._max_pending = max_pending
        self._max_pending_per_team = max_pending_per_team
        self._lock = Lock()
        self._teams = {}
        self._rotation = deque()
        self._job_teams = {}
        self._size = 0
        self._claims = {}
        self._claimed_jobs = set()
        self._claimed_size = 0

    def put(self, team_id: Hashable, job_id: Hashable, payload: Any, *,
            compatibility_key: bytes | None = None) -> None:
        if compatibility_key is not None and (type(compatibility_key) is not bytes or len(compatibility_key) != 32):
            raise ValueError('compatibility key must be exactly 32 immutable bytes or None')
        with self._lock:
            if job_id in self._job_teams or job_id in self._claimed_jobs:
                raise ValueError('job_id is already pending or claimed')
            if self._size >= self._max_pending:
                raise QueueFull('global pending-job limit reached')
            team = self._teams.get(team_id)
            if len(team or ()) >= self._max_pending_per_team:
                raise QueueFull('team pending-job limit reached')
            new_team = team is None
            if new_team:
                team = OrderedDict()
            next_size = self._size + 1
            item = QueueItem(team_id, job_id, payload, compatibility_key)
            try:
                team[job_id] = item
                self._job_teams[job_id] = team_id
                if new_team:
                    self._teams[team_id] = team
                    self._rotation.append(team_id)
            except BaseException:
                if new_team:
                    if self._rotation and self._rotation[-1] is team_id:
                        self._rotation.pop()
                    self._teams.pop(team_id, None)
                self._job_teams.pop(job_id, None)
                team.pop(job_id, None)
                raise
            self._size = next_size

    def peek_compatibility(self):
        """Inspect the current fair head's key; it may change before a claim."""
        with self._lock:
            if not self._rotation:
                raise QueueEmpty()
            key = self._teams[self._rotation[0]][
                next(iter(self._teams[self._rotation[0]]))].compatibility_key
        return key

    def peek_head_id(self):
        """Return the present fair head's job ID without claiming its payload.

        The result may become stale immediately. A coordinator needing a
        stable peek/claim must serialize its queue mutations separately; a
        successful pop still needs the recoverable cohort-claim ledger. For
        custody, use broker-owned immutable job IDs without payload references.
        """
        with self._lock:
            if not self._rotation:
                raise QueueEmpty()
            job_id = next(iter(self._teams[self._rotation[0]]))
        return job_id

    def pop_compatible(self, compatibility_key, *, claim_id, allowed_sizes, max_per_team):
        """Atomically claim the longest allowed compatible round-robin prefix.

        A stale or nonbatchable head, incompatible next head, or team cap ends
        the prefix without skipping it. Return an empty tuple if no requested
        size is available. This method does not wait for arrivals, validate
        model compatibility, authenticate keys, or execute payload callbacks.
        A successful claim remains recoverable by claim_id until acknowledged.
        """
        _claim_id(claim_id)
        if type(compatibility_key) is not bytes or len(compatibility_key) != 32:
            raise ValueError('compatibility key must be exactly 32 immutable bytes')
        if (type(allowed_sizes) is not tuple or not allowed_sizes
                or any(type(n) is not int or n <= 0 or n > self._max_pending for n in allowed_sizes)
                or any(left >= right for left, right in zip(allowed_sizes, allowed_sizes[1:]))):
            raise ValueError('allowed sizes must be increasing positive queue-bounded integers')
        if type(max_per_team) is not int or not 1 <= max_per_team <= self._max_pending:
            raise ValueError('max_per_team must be a positive queue-bounded integer')
        with self._lock:
            chosen = self._pop_compatible_locked(compatibility_key, claim_id, allowed_sizes, max_per_team)
        return chosen

    def _pop_compatible_locked(self, compatibility_key, claim_id, allowed_sizes, max_per_team):
        if claim_id in self._claims:
            raise ValueError('claim_id is already outstanding')
        if not self._rotation:
            raise QueueEmpty()
        # Inspect a virtual round-robin, without altering any pending item.
        rotation = deque(self._rotation)
        iterators, counts, candidates = {}, {}, []
        while rotation and len(candidates) < allowed_sizes[-1]:
            team_id = rotation.popleft()
            count = counts.get(team_id, 0)
            if count >= max_per_team:
                break
            team = self._teams[team_id]
            iterator = iterators.get(team_id)
            if iterator is None:
                iterator = iter(team.values())
                iterators[team_id] = iterator
            item = next(iterator)
            if item.compatibility_key != compatibility_key:
                break
            candidates.append(item)
            counts[team_id] = count + 1
            if count + 1 < len(team):
                rotation.append(team_id)
        size = next((n for n in reversed(allowed_sizes) if n <= len(candidates)), 0)
        if not size:
            return ()
        chosen = tuple(candidates[:size])
        if self._claimed_size + size > self._max_pending:
            raise QueueFull('outstanding cohort-claim limit reached')
        # Build complete shadow state before committing; no payload is
        # dropped while the lock is held because chosen retains each item.
        teams = self._teams.copy()
        rotation = deque(self._rotation)
        job_teams = self._job_teams.copy()
        claims = self._claims.copy()
        claimed_jobs = self._claimed_jobs.copy()
        claims[claim_id] = chosen
        copied = set()
        for item in chosen:
            team_id = rotation[0]
            if team_id not in copied:
                teams[team_id] = teams[team_id].copy()
                copied.add(team_id)
            team = teams[team_id]
            del team[item.job_id]
            del job_teams[item.job_id]
            claimed_jobs.add(item.job_id)
            if team:
                rotation.rotate(-1)
            else:
                rotation.popleft()
                del teams[team_id]
        size_after = self._size - size
        claimed_size_after = self._claimed_size + size
        previous = (self._teams, self._rotation, self._job_teams, self._size,
                    self._claims, self._claimed_jobs, self._claimed_size)
        try:
            self._teams = teams
            self._rotation = rotation
            self._job_teams = job_teams
            self._size = size_after
            self._claims = claims
            self._claimed_jobs = claimed_jobs
            self._claimed_size = claimed_size_after
        except BaseException:
            (self._teams, self._rotation, self._job_teams, self._size,
             self._claims, self._claimed_jobs, self._claimed_size) = previous
            raise
        return chosen

    def get_claim(self, claim_id):
        """Recover an unacknowledged claim; None means no such claim exists."""
        _claim_id(claim_id)
        with self._lock:
            claim = self._claims.get(claim_id)
        return claim

    def acknowledge_claim(self, claim_id):
        """Release a claim only after the trusted caller has taken ownership."""
        _claim_id(claim_id)
        with self._lock:
            selected = self._claims.get(claim_id)
            acknowledged = selected is not None
            if acknowledged:
                claims = self._claims.copy()
                jobs = self._claimed_jobs.copy()
                for item in selected:
                    jobs.remove(item.job_id)
                del claims[claim_id]
                size = self._claimed_size - len(selected)
                previous = self._claims, self._claimed_jobs, self._claimed_size
                try:
                    self._claims = claims
                    self._claimed_jobs = jobs
                    self._claimed_size = size
                except BaseException:
                    self._claims, self._claimed_jobs, self._claimed_size = previous
                    raise
        return acknowledged

    def pop(self) -> QueueItem:
        with self._lock:
            if not self._rotation:
                raise QueueEmpty()
            team_id = self._rotation[0]
            team = self._teams[team_id]
            next_size = self._size - 1
            job_id = next(iter(team))
            item = team[job_id]
            will_rotate = len(team) > 1
            if will_rotate:
                self._rotation.append(team_id)
            try:
                del team[job_id]
            except BaseException:
                if will_rotate:
                    self._rotation.pop()
                raise
            self._rotation.popleft()
            del self._job_teams[job_id]
            if not team:
                del self._teams[team_id]
            self._size = next_size
            return item

    def cancel(self, job_id: Hashable) -> bool:
        """Remove a pending job by ID; return false if it is no longer queued."""
        with self._lock:
            if job_id not in self._job_teams:
                return False
            next_size = self._size - 1
            team_id = self._job_teams[job_id]
            team = self._teams[team_id]
            # A payload finalizer may call queue methods; release it outside the lock.
            removed = team[job_id]
            del team[job_id]
            del self._job_teams[job_id]
            if not team:
                del self._teams[team_id]
                self._rotation.remove(team_id)
            self._size = next_size
        del removed
        return True

    def __len__(self) -> int:
        with self._lock:
            return self._size
