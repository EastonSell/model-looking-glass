"""Isolated entry for the fixed CPU fresh-context launcher and owner.

The interpreter, its standard library, this entry file and the selected virtual
environment are trusted. Source binding occurs in the launched module; this is
not a sandbox or an attestation of the installed dependencies.
"""

import runpy
import sys
from pathlib import Path


def main():
    if sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.dont_write_bytecode:
        raise RuntimeError("use the isolated -I -S -B entry")
    if len(sys.argv) < 2 or sys.argv[1] not in ("launcher", "owner"):
        raise ValueError("choose the fixed launcher or owner entry")
    root = Path(__file__).resolve(strict=True).parents[1]
    executable = Path(sys.executable).resolve(strict=True)
    site = (executable.parents[1] / "Lib" / "site-packages").resolve(strict=True)
    if not site.is_dir() or root == site:
        raise RuntimeError("the interpreter's virtual-environment site is unavailable")
    # Isolated/no-site startup places the interpreter's standard library first.
    # Do not run .pth files or inherit PYTHONPATH, the current directory, or the
    # user site; append only the trusted venv packages and this checkout.
    sys.path.extend((str(site), str(root)))
    role = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    module = ("looking_glass.agreement_head_fresh_launcher_v4" if role == "launcher" else
              "looking_glass.agreement_head_fresh_owner_v4")
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
