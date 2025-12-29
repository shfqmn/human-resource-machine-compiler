"""Minimal HRM assembler / simulator.

Supported instructions (text form):
- INBOX
- OUTBOX
- COPYFROM <i>
- COPYTO <i>
- ADD <i>
- SUB <i>
- BUMPUP <i>
- BUMPDN <i>
- JUMP <label>
- JUMPZ <label>
- JUMPN <label>
- LABEL <name>

Labels are declared with `LABEL name` lines. Blank lines and lines
starting with `#` are ignored.
"""
from typing import List, Dict, Optional, cast, Any
from collections import defaultdict


# Public set of supported textual HRM instructions
SUPPORTED_OPS = {
    "INBOX",
    "OUTBOX",
    "COPYFROM",
    "COPYTO",
    "ADD",
    "SUB",
    "BUMPUP",
    "BUMPDN",
    "JUMP",
    "JUMPZ",
    "JUMPN",
}


class Simulator:
    def __init__(self, allowed_ops: Optional[set] = None, tiles: Optional[int] = None) -> None:
        self.program: List[str] = []
        self.labels: Dict[str, int] = {}
        # memory is initialized at run-time; preserve field here
        self.memory: Dict[int, Optional[Any]] = {}
        self.acc: Optional[Any] = None
        # optional tile limit; None => unbounded
        self.tiles: Optional[int] = tiles
        # allowed_ops: None means permit all SUPPORTED_OPS; otherwise use provided set
        if allowed_ops is None:
            self.allowed_ops = SUPPORTED_OPS
        else:
            # normalize to upper-case strings
            self.allowed_ops = {op.upper() for op in allowed_ops}

    def _validate_tile_index(self, idx: int) -> None:
        if self.tiles is not None:
            if idx < 0 or idx >= self.tiles:
                raise RuntimeError(f"Tile index {idx} out of bounds (tiles={self.tiles})")

    def _read_tile(self, idx: int) -> Any:
        """Read a tile's value, raising if the tile is uninitialized (None)."""
        self._validate_tile_index(idx)
        v = self.memory.get(idx, None)
        if v is None:
            raise RuntimeError(f"Empty tile: cannot read tile {idx}")
        return v

    def _require_int_value(self, value: Any, context: str) -> int:
        if not isinstance(value, int):
            raise RuntimeError(f"{context} requires a number, got {value!r}")
        return value

    def _require_numeric_value(self, value: Any, context: str) -> int:
        """Accept ints or single alphabet chars and return their numeric value."""
        if isinstance(value, int):
            return value
        if isinstance(value, str) and len(value) == 1 and value.isalpha():
            return ord(value.upper()) - ord("A") + 1
        raise RuntimeError(f"{context} requires a number or letter, got {value!r}")

    def load_program(self, lines: List[str]) -> None:
        # Keep original lines but drop pure comments; resolution will handle
        # LABEL declarations in both `LABEL name` and `name:` forms, plus
        # ignore `COMMENT` and `DEFINE` blocks produced by some HRM exports.
        self.program = [ln.rstrip() for ln in lines]
        self._resolve_labels()

    def _resolve_labels(self) -> None:
        self.labels = {}
        newprog: List[str] = []
        skip_define = False
        for i, line in enumerate(self.program):
            s = line.strip()
            if not s:
                # blank line resets DEFINE skipping
                skip_define = False
                continue

            if skip_define:
                continue

            up = s.upper()
            if up.startswith("DEFINE "):
                # skip the DEFINE block until a blank line
                skip_define = True
                continue

            if up.startswith("COMMENT"):
                # ignore COMMENT metadata lines
                continue

            # label form: `name:`
            if s.endswith(":"):
                name = s[:-1].strip()
                if not name:
                    raise RuntimeError(f"Invalid label declaration: {line}")
                self.labels[name] = len(newprog)
                continue

            parts = s.split()
            if parts and parts[0].upper() == "LABEL":
                if len(parts) < 2:
                    raise RuntimeError(f"Invalid LABEL declaration: {line}")
                name = parts[1]
                self.labels[name] = len(newprog)
                continue

            # ignore pure comment lines starting with # or HRM header markers
            if s.startswith("#") or s.startswith("--"):
                continue

            # only accept known ops; otherwise ignore (covers binary DEFINE blobs)
            valid_ops = SUPPORTED_OPS
            if not parts:
                continue
            tok = parts[0]
            if tok.upper() not in valid_ops:
                continue

            # If the opcode token is lowercased, it was produced by the
            # compiler to indicate a disallowed operation. Treat such
            # lowered tokens as inert text (ignore them) so they are not
            # executed at runtime. This preserves the compiler's intent
            # of marking disallowed ops without letting them run.
            if tok != tok.upper():
                continue

            newprog.append(s)
        self.program = newprog
        # After resolving labels, validate operations are permitted.
        # If an opcode token in the source is lowercased, treat it as a
        # deliberately-lowered (disallowed) form and do not raise an error.
        bad_ops = []
        for ln in self.program:
            parts = ln.split()
            if not parts:
                continue
            token = parts[0]
            op = token.upper()
            # If the op is not in the allowed set, only treat it as a
            # bad op when the source used the uppercase form. Lowercased
            # tokens are generated by the compiler to indicate a lowered
            # (disallowed) operation and should be accepted as inert text.
            if op not in self.allowed_ops:
                if token != token.upper():
                    # lowered token -> accept (do not report as bad)
                    continue
                bad_ops.append(op)
        if bad_ops:
            bad_list = ", ".join(sorted(set(bad_ops)))
            raise RuntimeError(f"Program uses unsupported operations: {bad_list}")

    def run(self, inbox: List[Any], initial_memory: Optional[Dict[int, Any]] = None) -> List[Any]:
        pc = 0
        outbox: List[int] = []
        # initialize memory; when `tiles` is set, prefill allowed indices
        # with 0 and enforce bounds. When `tiles` is omitted, treat the
        # address space as 'all empty' by using a defaultdict that returns
        # 0 for any accessed index.
        if self.tiles is not None:
            # Represent uninitialized tiles as None to mirror the game's
            # 'empty' tile semantics. Only header-declared inits will
            # populate concrete values; reads from None will raise.
            self.memory = cast(Dict[int, Optional[Any]], {i: None for i in range(self.tiles)})
        else:
            self.memory = defaultdict(int)

        # apply any provided initial memory values (from compiled header)
        if initial_memory:
            for k, v in initial_memory.items():
                # validate tile bounds when tiles is set
                if self.tiles is not None and (k < 0 or k >= self.tiles):
                    raise RuntimeError(f"Initial memory index {k} out of bounds (tiles={self.tiles})")
                self.memory[k] = v
        self.acc = None

        while pc < len(self.program):
            line = self.program[pc]
            parts = line.split()
            op = parts[0].upper()

            if op == "INBOX":
                if not inbox:
                    # End program gracefully when inbox is exhausted
                    return outbox
                self.acc = inbox.pop(0)
                pc += 1
                continue

            if op == "OUTBOX":
                if self.acc is None:
                    raise RuntimeError("No value in hand for OUTBOX")
                outbox.append(self.acc)
                self.acc = None
                pc += 1
                continue

            if op == "COPYFROM":
                arg = parts[1]
                # support indirect: COPYFROM [i]
                if arg.startswith("[") and arg.endswith("]"):
                    idx = int(arg[1:-1])
                    # read the pointer stored in tile `idx`
                    addr = self._require_int_value(self._read_tile(idx), "COPYFROM indirect address")
                    self._validate_tile_index(addr)
                    # read the value at the pointed-to address
                    self.acc = self._read_tile(addr)
                else:
                    idx = int(arg)
                    self._validate_tile_index(idx)
                    self.acc = self._read_tile(idx)
                pc += 1
                continue

            if op == "COPYTO":
                arg = parts[1]
                if self.acc is None:
                    raise RuntimeError("No value in hand for COPYTO")
                # support indirect: COPYTO [i]
                if arg.startswith("[") and arg.endswith("]"):
                    idx = int(arg[1:-1])
                    addr = self._require_int_value(self._read_tile(idx), "COPYTO indirect address")
                    self._validate_tile_index(addr)
                    self.memory[addr] = self.acc
                else:
                    idx = int(arg)
                    self._validate_tile_index(idx)
                    self.memory[idx] = self.acc
                pc += 1
                continue

            if op == "ADD":
                if len(parts) < 2:
                    raise RuntimeError("Missing argument for ADD")
                arg = parts[1]
                if self.acc is None:
                    raise RuntimeError("No value in hand for ADD")
                acc_val = self._require_int_value(self.acc, "ADD accumulator")
                # support indirect: ADD [i] means use memory[i] as address
                if arg.startswith("[") and arg.endswith("]"):
                    idx = int(arg[1:-1])
                    addr = self._require_int_value(self._read_tile(idx), "ADD indirect address")
                    self._validate_tile_index(addr)
                    addend = self._require_int_value(self._read_tile(addr), "ADD operand")
                else:
                    idx = int(arg)
                    self._validate_tile_index(idx)
                    addend = self._require_int_value(self._read_tile(idx), "ADD operand")

                # add tile value into hand; do NOT modify memory
                self.acc = acc_val + addend
                pc += 1
                continue

            if op == "SUB":
                idx = int(parts[1])
                self._validate_tile_index(idx)
                if self.acc is None:
                    raise RuntimeError("No value in hand for SUB")

                acc_raw = self.acc
                operand_raw = self._read_tile(idx)

                def _is_letter(v: Any) -> bool:
                    return isinstance(v, str) and len(v) == 1 and v.isalpha()

                # HRM forbids mixing letters with numbers for SUB; both operands
                # must be numbers or both must be letters. Convert letters to their
                # numeric ordering only after validating the types match.
                if isinstance(acc_raw, int) and isinstance(operand_raw, int):
                    acc_val = acc_raw
                    operand_val = operand_raw
                elif _is_letter(acc_raw) and _is_letter(operand_raw):
                    acc_val = self._require_numeric_value(acc_raw, "SUB accumulator")
                    operand_val = self._require_numeric_value(operand_raw, "SUB operand")
                else:
                    raise RuntimeError("SUB requires both operands to be numbers or both letters")

                self.acc = acc_val - operand_val
                pc += 1
                continue

            if op == "BUMPUP":
                idx = int(parts[1])
                self._validate_tile_index(idx)
                v = self._require_int_value(self._read_tile(idx), "BUMPUP operand") + 1
                self.memory[idx] = v
                self.acc = v
                pc += 1
                continue

            if op == "BUMPDN":
                idx = int(parts[1])
                self._validate_tile_index(idx)
                v = self._require_int_value(self._read_tile(idx), "BUMPDN operand") - 1
                self.memory[idx] = v
                self.acc = v
                pc += 1
                continue

            if op in ("JUMP", "JUMPZ", "JUMPN"):
                if len(parts) < 2:
                    raise RuntimeError(f"Missing label for {op}")
                label = parts[1]
                if label not in self.labels:
                    raise RuntimeError(f"Unknown label: {label}")

                if op == "JUMP":
                    pc = self.labels[label]
                    continue

                if self.acc is None:
                    raise RuntimeError(f"No value in hand for {op}")

                # HRM allows branching on inbox characters by treating any
                # non-integer as non-zero and non-negative. Only pure ints
                # participate in numeric zero/negative checks.
                if isinstance(self.acc, int):
                    acc_val = self.acc
                    if op == "JUMPZ":
                        if acc_val == 0:
                            pc = self.labels[label]
                            continue
                        pc += 1
                        continue

                    if op == "JUMPN":
                        if acc_val < 0:
                            pc = self.labels[label]
                            continue
                        pc += 1
                        continue
                else:
                    # Non-integer values are treated as truthy and not
                    # negative, so conditional jumps fall through.
                    pc += 1
                    continue

            raise RuntimeError(f"Unknown instruction: {line}")

        return outbox

    def trace_run(self, inbox: List[Any], initial_memory: Optional[Dict[int, Any]] = None, max_steps: int = 100000) -> tuple[List[Any], List[str]]:
        """Run the program while recording a step-by-step trace.

        Returns (outbox, logs). Logs contain one entry per executed instruction
        with PC, instruction text, ACC, and a snapshot of referenced memory.
        On runtime error the method raises the same RuntimeError after
        appending the recent trace entries to the exception message for
        diagnosis.
        """
        pc = 0
        outbox: List[int] = []
        logs: List[str] = []

        # initialize memory like `run`
        if self.tiles is not None:
            self.memory = cast(Dict[int, Optional[Any]], {i: None for i in range(self.tiles)})
        else:
            self.memory = defaultdict(int)

        if initial_memory:
            for k, v in initial_memory.items():
                if self.tiles is not None and (k < 0 or k >= self.tiles):
                    raise RuntimeError(f"Initial memory index {k} out of bounds (tiles={self.tiles})")
                self.memory[k] = v

        self.acc = None
        step = 0
        try:
            while pc < len(self.program):
                if step >= max_steps:
                    raise RuntimeError("Trace exceeded max_steps")
                line = self.program[pc]
                sline = line.strip()
                parts = sline.split()
                op = parts[0].upper() if parts else ""

                # collect referenced tile indices for small snapshot
                refs = []
                if parts and len(parts) > 1:
                    arg = parts[1]
                    if arg.startswith("[") and arg.endswith("]"):
                        try:
                            refs.append(int(arg[1:-1]))
                        except Exception:
                            pass
                    else:
                        try:
                            refs.append(int(arg))
                        except Exception:
                            pass

                mem_snapshot = {r: self.memory.get(r, None) for r in refs}
                logs.append(f"pc={pc} instr='{sline}' acc={self.acc!r} refs={mem_snapshot}")

                # Execute one step of the instruction (reuse run's logic)
                if op == "INBOX":
                    if not inbox:
                        return outbox, logs
                    self.acc = inbox.pop(0)
                    pc += 1
                    step += 1
                    continue

                if op == "OUTBOX":
                    if self.acc is None:
                        raise RuntimeError("No value in hand for OUTBOX")
                    outbox.append(self.acc)
                    self.acc = None
                    pc += 1
                    step += 1
                    continue

                if op == "COPYFROM":
                    arg = parts[1]
                    if arg.startswith("[") and arg.endswith("]"):
                        idx = int(arg[1:-1])
                        addr = self._require_int_value(self._read_tile(idx), "COPYFROM indirect address")
                        self._validate_tile_index(addr)
                        self.acc = self._read_tile(addr)
                    else:
                        idx = int(arg)
                        self._validate_tile_index(idx)
                        self.acc = self._read_tile(idx)
                    pc += 1
                    step += 1
                    continue

                if op == "COPYTO":
                    arg = parts[1]
                    if self.acc is None:
                        raise RuntimeError("No value in hand for COPYTO")
                    if arg.startswith("[") and arg.endswith("]"):
                        idx = int(arg[1:-1])
                        addr = self._require_int_value(self._read_tile(idx), "COPYTO indirect address")
                        self._validate_tile_index(addr)
                        self.memory[addr] = self.acc
                    else:
                        idx = int(arg)
                        self._validate_tile_index(idx)
                        self.memory[idx] = self.acc
                    pc += 1
                    step += 1
                    continue

                if op == "ADD":
                    if len(parts) < 2:
                        raise RuntimeError("Missing argument for ADD")
                    arg = parts[1]
                    if self.acc is None:
                        raise RuntimeError("No value in hand for ADD")
                    acc_val = self._require_int_value(self.acc, "ADD accumulator")
                    if arg.startswith("[") and arg.endswith("]"):
                        idx = int(arg[1:-1])
                        addr = self._require_int_value(self._read_tile(idx), "ADD indirect address")
                        self._validate_tile_index(addr)
                        addend = self._require_int_value(self._read_tile(addr), "ADD operand")
                    else:
                        idx = int(arg)
                        self._validate_tile_index(idx)
                        addend = self._require_int_value(self._read_tile(idx), "ADD operand")
                    self.acc = acc_val + addend
                    pc += 1
                    step += 1
                    continue

                if op == "SUB":
                    idx = int(parts[1])
                    self._validate_tile_index(idx)
                    if self.acc is None:
                        raise RuntimeError("No value in hand for SUB")
                    acc_val = self._require_numeric_value(self.acc, "SUB accumulator")
                    operand = self._require_numeric_value(self._read_tile(idx), "SUB operand")
                    self.acc = acc_val - operand
                    pc += 1
                    step += 1
                    continue

                if op == "BUMPUP":
                    idx = int(parts[1])
                    self._validate_tile_index(idx)
                    v = self._require_int_value(self._read_tile(idx), "BUMPUP operand") + 1
                    self.memory[idx] = v
                    self.acc = v
                    pc += 1
                    step += 1
                    continue

                if op == "BUMPDN":
                    idx = int(parts[1])
                    self._validate_tile_index(idx)
                    v = self._require_int_value(self._read_tile(idx), "BUMPDN operand") - 1
                    self.memory[idx] = v
                    self.acc = v
                    pc += 1
                    step += 1
                    continue

                if op in ("JUMP", "JUMPZ", "JUMPN"):
                    if len(parts) < 2:
                        raise RuntimeError(f"Missing label for {op}")
                    label = parts[1]
                    if label not in self.labels:
                        raise RuntimeError(f"Unknown label: {label}")

                    if op == "JUMP":
                        pc = self.labels[label]
                        step += 1
                        continue

                    if self.acc is None:
                        raise RuntimeError(f"No value in hand for {op}")
                    if isinstance(self.acc, int):
                        acc_val = self.acc

                        if op == "JUMPZ":
                            if acc_val == 0:
                                pc = self.labels[label]
                                step += 1
                                continue
                            pc += 1
                            step += 1
                            continue

                        if op == "JUMPN":
                            if acc_val < 0:
                                pc = self.labels[label]
                                step += 1
                                continue
                            pc += 1
                            step += 1
                            continue

                    # Non-integer ACC values are treated as truthy and
                    # non-negative for branching, mirroring the main run loop.
                    pc += 1
                    step += 1
                    continue

                raise RuntimeError(f"Unknown instruction: {line}")

            return outbox, logs
        except RuntimeError as e:
            msg = str(e)
            tail = "\n--- TRACE (last 20 entries) ---\n" + "\n".join(logs[-20:])
            raise RuntimeError(msg + tail) from e
