"""Typed identity for an optional trusted IPC custody notification.

The server epoch is supplied by its trusted process owner, which must ensure
it is not reused across relevant lifetimes. The generation is monotonic only
within one server. This key is not a wire credential or a disposal receipt.
"""
from dataclasses import dataclass
import re


_EPOCH = re.compile(r"[0-9a-f]{32}\Z")


class IPCCustodyCapacity(RuntimeError):
    """Trusted guardian denied reservation before any request state exists.

    Only the exact type is recognized. Raising it asserts that no record,
    claim, pending registration, or request-owned payload was created for
    the key. Violating that contract breaks the custody guarantee.
    """


@dataclass(frozen=True, slots=True)
class IPCCustodyKey:
    server_epoch: str
    connection_generation: int

    def __post_init__(self):
        if (type(self.server_epoch) is not str or not _EPOCH.fullmatch(self.server_epoch)
                or type(self.connection_generation) is not int
                or not 0 < self.connection_generation < 2**63):
            raise ValueError("invalid IPC custody identity")
