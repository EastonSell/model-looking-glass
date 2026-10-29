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


@dataclass(frozen=True)
class QueueItem:
    team_id: Hashable
    job_id: Hashable
    payload: Any


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

    def put(self, team_id: Hashable, job_id: Hashable, payload: Any) -> None:
        with self._lock:
            if job_id in self._job_teams:
                raise ValueError('job_id is already pending')
            if self._size >= self._max_pending:
                raise QueueFull('global pending-job limit reached')
            team = self._teams.get(team_id)
            if len(team or ()) >= self._max_pending_per_team:
                raise QueueFull('team pending-job limit reached')
            new_team = team is None
            if new_team:
                team = OrderedDict()
            next_size = self._size + 1
            item = QueueItem(team_id, job_id, payload)
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
