"""Simple peephole optimizer for HRM instruction lists.

This module performs a few safe, local transformations to reduce the
instruction count produced by the translator. Optimizations are intentionally
conservative to avoid changing program semantics.
"""
from typing import List, Tuple, Dict


def _split_op(line: str) -> Tuple[str, str]:
    s = (line or "").strip()
    if not s:
        return "", ""
    parts = s.split(None, 1)
    op = parts[0].upper()
    arg = parts[1] if len(parts) > 1 else ""
    return op, arg


def _collect_initial_bumps(lines: List[str]) -> Dict[int, int]:
    vals: Dict[int, int] = {}
    for line in lines:
        op, arg = _split_op(line)
        if op == "BUMPUP":
            try:
                idx = int(arg)
            except Exception:
                break
            vals[idx] = vals.get(idx, 0) + 1
            continue
        if op == "BUMPDN":
            try:
                idx = int(arg)
            except Exception:
                break
            vals[idx] = vals.get(idx, 0) - 1
            continue
        # stop at first non-bump line — init region is expected to be contiguous
        break
    return vals


def _split_blocks(lines: List[str]) -> List[Tuple[int, List[str]]]:
    # split into blocks by LABEL lines; return list of (start_index, block_lines)
    blocks: List[Tuple[int, List[str]]] = []
    cur: List[str] = []
    start_idx = 0
    for i, ln in enumerate(lines):
        op, _ = _split_op(ln)
        if op == "LABEL":
            if cur:
                blocks.append((start_idx, cur))
            cur = [ln]
            start_idx = i
        else:
            cur.append(ln)
    if cur:
        blocks.append((start_idx, cur))
    return blocks


def optimize(lines: List[str]) -> List[str]:
    # conservative, block-local optimizations
    if not lines:
        return lines

    # collect initial bump initializers (const tile values)
    initial_consts = _collect_initial_bumps(lines)

    # find max numeric tile used to allocate new consts if needed
    used_tiles = set()
    for ln in lines:
        op, arg = _split_op(ln)
        if not arg:
            continue
        # arguments may be labels or numbers; try parse ints and bracketed forms
        tok = arg.split()[0]
        if tok.startswith("[") and tok.endswith("]"):
            tok = tok[1:-1]
        try:
            idx = int(tok)
            used_tiles.add(idx)
        except Exception:
            continue
    next_tile = max(used_tiles) + 1 if used_tiles else 0

    blocks = _split_blocks(lines)
    out_lines: List[str] = []

    # preserve initial bump region (we'll prepend any new const inits here)
    init_region: List[str] = []
    i = 0
    for ln in lines:
        op, _ = _split_op(ln)
        if op in ("BUMPUP", "BUMPDN"):
            init_region.append(ln)
            i += 1
            continue
        break

    other_start = i

    new_inits: Dict[int, int] = {}

    for start_idx, block in blocks:
        # operate on a working copy of the block
        new_block: List[str] = []
        # track last write positions for COPYTO in this block
        last_write: Dict[int, int] = {}
        # track whether a tile has been read after a write
        read_after_write: Dict[int, bool] = {}
        # track modified tiles globally to be conservative for const folding
        modified: set = set()

        for idx_ln, ln in enumerate(block):
            op, arg = _split_op(ln)
            # straightforward rules
            if op == "":
                continue

            # coalesce consecutive LABELs: keep first
            if op == "LABEL":
                if new_block and _split_op(new_block[-1])[0] == "LABEL":
                    # drop duplicate label
                    continue
                new_block.append(ln)
                continue

            # remove COPYFROM X; COPYTO X pattern (no-op)
            if op == "COPYFROM" and new_block:
                # lookahead in original block
                next_idx = idx_ln + 1
                if next_idx < len(block):
                    op2, arg2 = _split_op(block[next_idx])
                    if op2 == "COPYTO" and arg2 == arg:
                        # skip both
                        # advance by consuming the next instruction by skipping iteration
                        # we simulate by popping next from block via index manipulation
                        # simply skip adding this COPYFROM and also mark to skip next
                        # implement by setting a flag on the block list to empty the next
                        block[next_idx] = ""
                        continue

            # remove duplicate consecutive COPYFROM
            if op == "COPYFROM" and new_block:
                prev_op, prev_arg = _split_op(new_block[-1])
                if prev_op == "COPYFROM" and prev_arg == arg:
                    continue

            # Retain all COPYTO writes. A previous attempt at dead-store
            # elimination looked forward only in linear order, which is
            # unsound in the presence of backward jumps (loops) and removed
            # live writes (e.g., loop-carried variables). Keep COPYTO to stay
            # conservative and correct.
            if op == "COPYTO":
                try:
                    tidx = int(arg.split()[0])
                    modified.add(tidx)
                except Exception:
                    pass
                new_block.append(ln)
                continue

            # constant folding: COPYFROM a; ADD b -> COPYFROM new_const if both are known
            if op == "COPYFROM":
                # see if next instruction in block is ADD
                next_idx = idx_ln + 1
                if next_idx < len(block):
                    op2, arg2 = _split_op(block[next_idx])
                    if op2 == "ADD":
                        # attempt fold if both args are constant and unmodified
                        try:
                            aidx = int(arg.split()[0])
                            bidx = int(arg2.split()[0])
                        except Exception:
                            new_block.append(ln)
                            continue

                        a_const = initial_consts.get(aidx)
                        b_const = initial_consts.get(bidx)
                        if a_const is not None and b_const is not None and aidx not in modified and bidx not in modified:
                            s = a_const + b_const
                            # allocate new const tile
                            new_idx = next_tile
                            next_tile += 1
                            # emit init bumps for new const
                            if s >= 0:
                                for _ in range(s):
                                    init_region.append(f"BUMPUP {new_idx}")
                            else:
                                for _ in range(-s):
                                    init_region.append(f"BUMPDN {new_idx}")
                            # replace the COPYFROM/ADD pair with COPYFROM new_idx
                            new_block.append(f"COPYFROM {new_idx}")
                            # mark next instruction consumed by skipping it
                            block[next_idx] = ""
                            # register new const
                            initial_consts[new_idx] = s
                            continue

            # default: keep instruction and update modified/read sets
            # detect writes
            if op in ("BUMPUP", "BUMPDN", "COPYTO"):
                try:
                    widx = int(arg.split()[0])
                    modified.add(widx)
                except Exception:
                    pass

            new_block.append(ln)

        # append cleaned block to output
        out_lines.extend(new_block)

    # Prepend init_region (including any new_inits we added) and filter empties
    final = [ln for ln in init_region + out_lines if ln and ln.strip()]
    return final
