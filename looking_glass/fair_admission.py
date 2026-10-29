"""Bounded, per-team fair waiters for a broker-owned admission step."""

from collections import OrderedDict, deque
from dataclasses import dataclass
from queue import Empty, Full
from threading import Lock
from typing import Any, Mapping


class AdmissionFull(Full):
    """No waiter position can be admitted without violating a limit."""


class AdmissionEmpty(Empty):
    """No currently eligible waiter can be claimed."""


class UnknownPrincipal(PermissionError):
    """The authenticated principal is absent from the trusted configuration."""


@dataclass(frozen=True)
class AdmissionRequest:
    ticket_id: int
    principal_id: str
    team_id: str
    payload: Any


class FairAdmissionQueue:
    """A broker-facing admission queue with protected per-team waiter slots."""

    def __init__(self, principal_teams: Mapping[str, str], max_waiters: int,
                 max_waiters_per_team: int):
        if not isinstance(principal_teams, Mapping):
            raise ValueError('principal-to-team mapping must be nonempty')
        self._principals = dict(principal_teams)
        if not self._principals:
            raise ValueError('principal-to-team mapping must be nonempty')
        if any(type(principal) is not str or not principal or
               type(team) is not str or not team
               for principal, team in self._principals.items()):
            raise ValueError('principal and team identifiers must be nonempty strings')
        for value in (max_waiters, max_waiters_per_team):
            if type(value) is not int or value <= 0:
                raise ValueError('waiter limits must be positive integers')
        self._team_ids = frozenset(self._principals.values())
        if max_waiters < len(self._team_ids):
            raise ValueError('max_waiters must provide one slot per configured team')
        self._max_waiters = max_waiters
        self._max_waiters_per_team = max_waiters_per_team
        self._lock = Lock()
        self._teams = {}
        self._rotation = deque()
        self._ticket_teams = {}
        self._size = 0
        self._next_ticket = 1

    def submit(self, principal_id: str, payload: Any) -> int:
        if type(principal_id) is not str or principal_id not in self._principals:
            raise UnknownPrincipal('principal is not configured')
        team_id = self._principals[principal_id]
        with self._lock:
            team = self._teams.get(team_id)
            empty_teams = len(self._team_ids) - len(self._teams)
            new_team = team is None
            if (len(team or ()) >= self._max_waiters_per_team or
                    self._size + 1 + empty_teams - int(new_team) > self._max_waiters):
                raise AdmissionFull()
            if new_team:
                team = OrderedDict()
                self._teams[team_id] = team
                self._rotation.append(team_id)
            ticket_id = self._next_ticket
            self._next_ticket += 1
            team[ticket_id] = AdmissionRequest(ticket_id, principal_id, team_id, payload)
            self._ticket_teams[ticket_id] = team_id
            self._size += 1
            return ticket_id

    def claim_next(self, eligible_teams=None) -> AdmissionRequest:
        if eligible_teams is None:
            eligible = self._team_ids
        else:
            if isinstance(eligible_teams, str):
                raise ValueError('eligible_teams must be a collection of team IDs')
            try:
                eligible = frozenset(eligible_teams)
            except TypeError as exc:
                raise ValueError('eligible_teams must be a collection of team IDs') from exc
            if any(type(team) is not str for team in eligible) or not eligible <= self._team_ids:
                raise ValueError('eligible_teams contains an unconfigured team')
        with self._lock:
            team_id = next((team for team in self._rotation if team in eligible), None)
            if team_id is None:
                raise AdmissionEmpty()
            self._rotation.remove(team_id)
            team = self._teams[team_id]
            ticket_id, request = team.popitem(last=False)
            del self._ticket_teams[ticket_id]
            if team:
                self._rotation.append(team_id)
            else:
                del self._teams[team_id]
            self._size -= 1
            return request

    def cancel(self, ticket_id: int) -> bool:
        """Cancel a waiting ticket; claimed requests belong to the broker."""
        if type(ticket_id) is not int or ticket_id <= 0:
            raise ValueError('ticket_id must be a positive integer')
        with self._lock:
            if ticket_id not in self._ticket_teams:
                return False
            team_id = self._ticket_teams.pop(ticket_id)
            team = self._teams[team_id]
            removed = team.pop(ticket_id)
            if not team:
                del self._teams[team_id]
                self._rotation.remove(team_id)
            self._size -= 1
        # Finalizers must not run under the internal queue lock.
        del removed
        return True

    def __len__(self) -> int:
        with self._lock:
            return self._size
