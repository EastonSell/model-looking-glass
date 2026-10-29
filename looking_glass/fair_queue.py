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
            if len(self._teams.get(team_id, ())) >= self._max_pending_per_team:
                raise QueueFull('team pending-job limit reached')
            if team_id not in self._teams:
                self._teams[team_id] = OrderedDict()
                self._rotation.append(team_id)
            self._teams[team_id][job_id] = QueueItem(team_id, job_id, payload)
            self._job_teams[job_id] = team_id
            self._size += 1

    def pop(self) -> QueueItem:
        with self._lock:
            if not self._rotation:
                raise QueueEmpty()
            team_id = self._rotation.popleft()
            team = self._teams[team_id]
            job_id, item = team.popitem(last=False)
            del self._job_teams[job_id]
            if team:
                self._rotation.append(team_id)
            else:
                del self._teams[team_id]
            self._size -= 1
            return item

    def cancel(self, job_id: Hashable) -> bool:
        """Remove a pending job by ID; return false if it is no longer queued."""
        with self._lock:
            if job_id not in self._job_teams:
                return False
            team_id = self._job_teams.pop(job_id)
            team = self._teams[team_id]
            # A payload finalizer may call queue methods; release it outside the lock.
            removed = team.pop(job_id)
            if not team:
                del self._teams[team_id]
                self._rotation.remove(team_id)
            self._size -= 1
        del removed
        return True

    def __len__(self) -> int:
        with self._lock:
            return self._size
