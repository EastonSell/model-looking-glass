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

    observed = []

    def warmup_trace(frame, event, arg):
        if frame.f_code is warmup.__code__:
            if event == "call":
                frame.f_trace_opcodes = True
            elif event == "opcode":
                observed.append(frame.f_lasti)
        return warmup_trace

    # CPython 3.12 requires an already-active frame to opt into opcode
    # events before settrace(); repeated unseeded sessions are insufficient.
    trace_frame = sys._getframe()
    previous_opcodes = trace_frame.f_trace_opcodes
    trace_frame.f_trace_opcodes = True
    try:
        # CPython 3.12 can wait until a later trace session to emit opcodes. Use a
        # separate tracer, because the supplied tracer may select only its target.
        for _ in range(3):
            try:
                sys.settrace(warmup_trace)
                warmup()
            finally:
                sys.settrace(previous)
            if observed:
                break
        if not observed:
            raise RuntimeError("interpreter did not emit opcode trace events")
        try:
            sys.settrace(tracer)
            call()
        finally:
            sys.settrace(previous)
    finally:
        sys.settrace(previous)
        trace_frame.f_trace_opcodes = previous_opcodes
        del trace_frame
