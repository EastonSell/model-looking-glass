"""Helpers for nonvacuous opcode interruption tests across CPython versions."""
import dis
import sys


def offsets_at_line(method, line, opname):
    # PEP 657 positions exist only in 3.11+. On 3.10, use the instruction
    # line table and require the caller to verify that the match is unique.
    current_line = None
    offsets = []
    for instruction in dis.get_instructions(method):
        if instruction.starts_line is not None:
            current_line = instruction.starts_line
        positions = getattr(instruction, "positions", None)
        precise = getattr(positions, "lineno", None)
        instruction_line = precise if precise is not None else current_line
        if instruction_line == line and instruction.opname == opname:
            offsets.append(instruction.offset)
    return offsets


def run_opcode_trace(tracer, call):
    """Warm opcode events in a separate session, then trace the actual call."""
    previous = sys.gettrace()

    def warmup():
        return None

    try:
        # CPython 3.12 can wait until a later trace session to emit opcodes.
        sys.settrace(tracer)
        warmup()
    finally:
        sys.settrace(previous)
    try:
        sys.settrace(tracer)
        call()
    finally:
        sys.settrace(previous)
