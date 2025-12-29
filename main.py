import argparse
import os
import tempfile
from hrm.compiler import compile_from_python
from hrm.assembler import Simulator, SUPPORTED_OPS
from hrm import optimizer as hrm_optimizer


def cli():
    p = argparse.ArgumentParser(description="Human Resource Machine - simple Python→HRM tool")
    sub = p.add_subparsers(dest="cmd")

    c_compile = sub.add_parser("compile", help="Compile a Python file containing `commands` to .hrm text")
    c_compile.add_argument("src", help="Source Python file")
    c_compile.add_argument("out", nargs="?", help="Output HRM file (optional). Defaults to <src>.hrm next to source")
    

    c_run = sub.add_parser("run", help="Run a .hrm program with required inbox values")
    c_run.add_argument("program", help="HRM program file")
    c_run.add_argument("--inbox", "-i", nargs="+", required=True, help="Provide inbox values (eg. --inbox 1 2 A B)")
    

    c_avail = sub.add_parser("available", help="Show available HRM commands supported by the tool")

    args = p.parse_args()
    if args.cmd == "compile":
        out_path = args.out if args.out else os.path.splitext(args.src)[0] + ".hrm"
        compile_from_python(args.src, out_path)
        print(f"Wrote HRM program to {out_path}")
    elif args.cmd == "available":
        print("Supported HRM instructions:")
        for op in sorted(SUPPORTED_OPS):
            print("-", op)
    elif args.cmd == "run":
        # --inbox is required (argparse enforces this). Use provided values.
        inbox_tokens = args.inbox or []
        converted_inbox = []
        for tok in inbox_tokens:
            if tok.lstrip('-').isdigit():
                converted_inbox.append(int(tok))
            else:
                # treat multi-char tokens as sequence of literal characters (no ord conversion)
                converted_inbox.extend(list(tok))

        # Read program and detect compiler-emitted header for tiles/init
        header_tiles = None
        header_inits = {}
        with open(args.program, "r", encoding="utf-8") as f:
            raw_lines = [l.rstrip() for l in f.readlines()]

        for ln in raw_lines:
            s = (ln or "").strip()
            if s.upper().startswith("-- TILES:"):
                try:
                    header_tiles = int(s.split(":", 1)[1].strip())
                except Exception:
                    header_tiles = None
                continue
            if s.upper().startswith("-- INIT"):
                # format: -- INIT <idx> <value>
                parts = s.split()
                if len(parts) >= 4 and parts[0] == "--" and parts[1].upper() == "INIT":
                    try:
                        idx = int(parts[2])
                        raw_val = " ".join(parts[3:])
                        try:
                            val = int(raw_val)
                        except Exception:
                            val = raw_val
                        header_inits[idx] = val
                    except Exception:
                        pass

        # Use header-declared tiles only; CLI overrides removed
        sim = Simulator(tiles=header_tiles)

        # Preprocess label syntax (convert `name:` -> `LABEL name`) so the
        # optimizer can operate on a consistent `LABEL` form. Keep other
        # lines untouched and then run the optimizer to perform peephole
        # simplifications before simulation.
        prep_lines = []
        for ln in raw_lines:
            s = (ln or "").strip()
            if s.endswith(":") and not s.startswith("#") and not s.startswith("--"):
                name = s[:-1].strip()
                if name:
                    prep_lines.append(f"LABEL {name}")
                    continue
            prep_lines.append(ln)

        try:
            lines = hrm_optimizer.optimize(prep_lines)
        except Exception:
            # Optimization failures should not block running; fall back
            # to the original program lines.
            lines = prep_lines

        sim.load_program(lines)
        out = sim.run(converted_inbox, initial_memory=header_inits if header_inits else None)

        # Prepare human-friendly outbox: show numeric list and printable characters
        printable_chars = []
        for v in out:
            if isinstance(v, str) and len(v) == 1:
                printable_chars.append(v)
            elif isinstance(v, int) and 32 <= v <= 126:
                printable_chars.append(chr(v))
            else:
                printable_chars.append(None)

        print("Outbox (values):", out)
        if any(c is not None for c in printable_chars):
            print("Outbox (chars):", ''.join(c if c is not None else '?' for c in printable_chars))
    else:
        p.print_help()


if __name__ == "__main__":
    cli()
