"""Fence a single baseline call, including an exceptional Python return.

This is not a source, exposure, resource or process supervisor. The enclosing
owner must keep the model and output namespace exclusive, observe direct-child
exit and cleanup, and never treat the provisional files alone as completion.
"""
from pathlib import Path

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_v2_owner as custody


class OwnerBoundaryError(RuntimeError):
    """The baseline call cannot be classified as a successful single use."""


class OneShot:
    def __init__(self, guard, screen, invoke):
        self.guard = guard
        self.screen = Path(screen)
        self.invoke = invoke
        self.used = False
        if (not self.screen.is_absolute() or not self.screen.parent.is_dir()
                or self.screen.exists() or not callable(invoke)):
            raise OwnerBoundaryError("a fresh screen and callable are required")

    def run(self):
        if self.used:
            raise OwnerBoundaryError("the one-shot baseline was already consumed")
        self.used = True
        try:
            self.guard.verify()
            result = self.invoke()
            if type(result) is not dict or type(result.get("identity")) is not dict:
                raise OwnerBoundaryError("baseline return differs")
            completed_raw = (self.screen / "completed.json").read_bytes()
            if len(completed_raw) > 100_000:
                raise OwnerBoundaryError("completion record exceeds its bound")
            completed = v1._decode_document(completed_raw)
            if (set(completed) != {"identity", "provisional_sha256", "status", "cleanup"}
                    or v1._canonical(completed["identity"]) != v1._canonical(result["identity"])
                    or completed["status"] != "provisional-complete" or completed["cleanup"] != "verified"
                    or completed["provisional_sha256"] != v1._file_sha(self.screen / "provisional.json")):
                raise OwnerBoundaryError("provisional completion differs")
            self.guard.verify()
            custody._json_new(self.screen.parent / "owner-return.json", {
                "identity": result["identity"], "completed_sha256": v1._sha(completed_raw),
                "status": "returned", "scope": "direct-process return only; exit and cleanup remain external"})
            return result
        except BaseException as error:
            try:
                custody._json_new(self.screen.parent / "owner-incomplete.json", {
                    "error_type": type(error).__name__, "status": "incomplete",
                    "scope": "preserve all provisional files; never retry an uncertain call"})
            except BaseException as recording:
                error.add_note("owner incomplete record could not be written: " + type(recording).__name__)
            raise
        finally:
            # A caller must never use this model for another forward after the
            # one-shot call, even when an exception arrives during return.
            self.guard.poisoned = True
