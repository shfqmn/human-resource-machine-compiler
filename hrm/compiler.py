import ast
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, cast, Any
from .assembler import SUPPORTED_OPS


@dataclass
class TileConfig:
    max_tiles: Optional[int]
    preallocated: Dict[str, int]
    const_inits: Dict[str, Any]
    tile_names: List[Optional[str]]


def _parse_allowed_ops(allowed_node: Optional[ast.AST]) -> Optional[Set[str]]:
    """Return the module-level `allowed` opcode set if present."""
    if allowed_node is None:
        return None

    try:
        allowed_val = ast.literal_eval(allowed_node)
    except Exception as e:
        raise RuntimeError("`allowed` must be a literal list of opcode strings") from e

    if not isinstance(allowed_val, (list, tuple)) or not all(isinstance(c, str) for c in allowed_val):
        raise RuntimeError("`allowed` must be a list/tuple of strings")

    return {c.upper() for c in allowed_val}


def _parse_modifiable_tiles(mod_node: Optional[ast.AST]) -> Optional[Set[int]]:
    """Return the module-level `modifiable_tiles_idx` set if present."""
    if mod_node is None:
        return None

    try:
        mod_val = ast.literal_eval(mod_node)
    except Exception as e:
        raise RuntimeError("`modifiable_tiles_idx` must be a literal list/tuple of ints") from e

    if not isinstance(mod_val, (list, tuple)) or not all(isinstance(i, int) for i in mod_val):
        raise RuntimeError("`modifiable_tiles_idx` must be a list/tuple of integers")

    return set(mod_val)


def _parse_tiles(tiles_node: Optional[ast.AST], src_path: str, src: str) -> TileConfig:
    """Parse the module-level `tiles` config into a TileConfig."""
    max_tiles: Optional[int] = None
    preallocated: Dict[str, int] = {}
    const_inits: Dict[str, Any] = {}
    tile_names: List[Optional[str]] = []

    if tiles_node is None:
        return TileConfig(max_tiles, preallocated, const_inits, tile_names)

    try:
        tiles_val = ast.literal_eval(tiles_node)
    except Exception as e:
        try:
            snippet = ast.get_source_segment(src, tiles_node) or ast.dump(tiles_node)
        except Exception:
            snippet = ast.dump(tiles_node)
        raise RuntimeError(
            f"`tiles` must be a literal int, list of names, or dict of name->int; got: {snippet!r} in {src_path}"
        ) from e

    if isinstance(tiles_val, int):
        if tiles_val < 0:
            raise RuntimeError("`tiles` must be non-negative")
        max_tiles = tiles_val
    elif isinstance(tiles_val, list):
        idx = 0
        seen_name_counts: Dict[str, int] = {}

        def _unique_tile_name(base: str) -> str:
            count = seen_name_counts.get(base, 0)
            if count == 0:
                seen_name_counts[base] = 1
                return base
            count += 1
            seen_name_counts[base] = count
            return f"{base}__{count}"

        for item in tiles_val:
            if item is None:
                tile_names.append(None)
                idx += 1
                continue
            if isinstance(item, str):
                name = _unique_tile_name(item)
                preallocated[name] = idx
                const_inits[name] = item
                tile_names.append(name)
                idx += 1
                continue
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (int, str)):
                base_name, init_val = item[0], item[1]
                name = _unique_tile_name(base_name)
                preallocated[name] = idx
                const_inits[name] = init_val
                tile_names.append(name)
                idx += 1
                continue
            if isinstance(item, int):
                name = f"_tile_{idx}"
                preallocated[name] = idx
                const_inits[name] = item
                tile_names.append(name)
                idx += 1
                continue
            raise RuntimeError("`tiles` list must contain only strings, ints, None, or (name, int) tuples")
        if idx > 0:
            max_tiles = idx
    elif isinstance(tiles_val, dict):
        idx = 0
        for k, v in tiles_val.items():
            if not isinstance(k, str) or not isinstance(v, (int, str)):
                raise RuntimeError("`tiles` dict must map names (str) to ints or chars")
            preallocated[k] = idx
            const_inits[k] = v
            idx += 1
        if idx > 0:
            max_tiles = idx
    else:
        raise RuntimeError(f"`tiles` must be an int, list of names, or dict of name->int; got AST node: {ast.dump(tiles_node)}")

    return TileConfig(max_tiles, preallocated, const_inits, tile_names)


def _scan_direct_tile_refs(tree: ast.AST, tile_names: List[Optional[str]], extract_index) -> Set[int]:
    direct_referenced_tiles: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "tiles":
            # Attempt to extract a constant index. If the index is not a
            # constant integer (e.g., it's a variable or expression), treat
            # it as a dynamic/indirect reference and skip adding to the set
            # of directly-referenced tiles. This allows `tiles[var]` usages.
            try:
                idx_val = extract_index(node.slice)
            except Exception:
                # dynamic index (variable or expression) — ignore for direct refs
                continue
            if not isinstance(idx_val, int) or idx_val < 0 or idx_val >= len(tile_names):
                raise RuntimeError("tiles index out of range")
            direct_referenced_tiles.add(idx_val)
    return direct_referenced_tiles


def _emit_program(
    out_path: str,
    commands: List[str],
    max_tiles: Optional[int],
    const_inits: Dict[str, Any],
    tile_names: List[Optional[str]],
    var_to_tile: Dict[Optional[str], int],
    init_counts: Dict[int, int],
    allowed_ops: Optional[Set[str]],
    module_allowed_ops: Optional[Set[str]],
    modifiable_tiles: Optional[Set[int]] = None,
) -> None:
    """Write the compiled commands plus headers/validation to disk."""

    if not commands:
        raise RuntimeError(
            "No `commands` found and Python->HRM translator produced no output; please provide a `commands` list or a supported Python program."
        )

    # Check program size limit (255 instructions)
    instruction_count = len([c for c in commands if not c.strip().upper().startswith("LABEL")])
    if instruction_count > 255:
        raise RuntimeError(f"Program too long: {instruction_count} instructions (max 255)")

    final_header_inits: Dict[int, Any] = {}

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("-- HUMAN RESOURCE MACHINE PROGRAM --\n\n")
        final_lines = [l.rstrip() for l in commands]

        if max_tiles is not None:
            f.write(f"-- TILES: {max_tiles}\n")

        # Delay emitting INIT headers until after we know which init_counts
        # entries are actually referenced by the compiled instructions.
        header_inits: Dict[int, Any] = {}

        lines = [l.rstrip() for l in commands]

        # Duplicate label/exec index scanning removed (handled further down).

        # Collect numeric arguments referenced by the compiled program by
        # scanning the final emitted lines (exec_insts is constructed later).
        numeric_args: Set[int] = set()
        for ln in final_lines:
            if not ln:
                continue
            parts = ln.split(None, 1)
            op = parts[0].upper()
            if op == "LABEL":
                continue
            arg = parts[1] if len(parts) > 1 else None
            if not arg:
                continue
            first = arg.split(None, 1)[0]
            # bracketed indirect like "[10]"
            if first.startswith("[") and first.endswith("]"):
                inner = first[1:-1]
                if inner.lstrip('-').isdigit():
                    numeric_args.add(int(inner))
                continue
            if first.lstrip('-').isdigit():
                numeric_args.add(int(first))

        seen_inits: Set[int] = set()
        for name, val in const_inits.items():
            if name in var_to_tile:
                idx = var_to_tile[name]
                header_inits.setdefault(idx, val)
                seen_inits.add(idx)

        # Emit header INIT lines now that we know which constants are referenced.
        # Keep deterministic ordering for diff-friendliness.
        for idx in sorted(header_inits):
            f.write(f"-- INIT {idx} {header_inits[idx]}\n")
        if header_inits:
            f.write("\n")

        lines = [l.rstrip() for l in commands]

        label_to_pos: Dict[str, int] = {}
        for pos, ln in enumerate(lines):
            s = (ln or "").strip()
            if not s:
                continue
            parts = s.split(None, 1)
            if parts[0].upper() == "LABEL":
                lbl = parts[1] if len(parts) > 1 else ""
                label_to_pos[lbl] = pos

        exec_positions: List[int] = []
        for pos, ln in enumerate(lines):
            s = (ln or "").strip()
            if not s or s.startswith("#") or s.startswith("--"):
                continue
            parts = s.split(None, 1)
            if parts[0].upper() == "LABEL":
                continue
            exec_positions.append(pos)

        if len(exec_positions) > 255:
            raise RuntimeError(f"Program too long: {len(exec_positions)} instructions (max 255)")

        def label_to_exec_idx(name: str) -> Optional[int]:
            if name not in label_to_pos:
                return None
            label_pos = label_to_pos[name]
            for i, p in enumerate(exec_positions):
                if p > label_pos:
                    return i
            return len(exec_positions)

        exec_insts = []
        for p in exec_positions:
            s = lines[p].strip()
            parts = s.split(None, 2)
            op = parts[0].upper()
            arg = parts[1] if len(parts) > 1 else None
            exec_insts.append((op, arg, p))

        n = len(exec_insts)
        preds: List[Set[int]] = [set() for _ in range(n)]
        succs: List[Set[int]] = [set() for _ in range(n)]
        for i, (op, arg, p) in enumerate(exec_insts):
            if i + 1 < n:
                succs[i].add(i + 1)
                preds[i + 1].add(i)

            if op == "JUMP":
                tgt = label_to_exec_idx(arg or "")
                if tgt is None:
                    raise RuntimeError(f"Unknown label: {arg}")
                succs[i].add(tgt)
                if tgt < n:
                    preds[tgt].add(i)
                if i + 1 < n and (i + 1) in succs[i]:
                    succs[i].discard(i + 1)
                    preds[i + 1].discard(i)
            elif op in {"JUMPZ", "JUMPN"}:
                tgt = label_to_exec_idx(arg or "")
                if tgt is None:
                    raise RuntimeError(f"Unknown label: {arg}")
                succs[i].add(tgt)
                if tgt < n:
                    preds[tgt].add(i)

        universe: Set[int] = set(var_to_tile.values()) | set(seen_inits) | set(init_counts.keys())
        # Only tiles with known initialization (header INITs) are safe at entry.
        # Do not treat allocated variables as initialized; that can mask reads
        # from empty tiles (e.g., temps) and suppress necessary header INITs.
        entry_in = set(seen_inits)
        out_sets: List[Set[int]] = [set(entry_in) for _ in range(n)]

        while True:
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if not preds[i]:
                        in_set = set(entry_in)
                    else:
                        it = None
                        for p in preds[i]:
                            if it is None:
                                it = set(out_sets[p])
                            else:
                                it &= out_sets[p]
                        in_set = it if it is not None else set()

                    op, arg, _p = exec_insts[i]
                    out_set = set(in_set)
                    if op == "COPYTO":
                        if arg is not None and arg.lstrip('-').isdigit():
                            out_set.add(int(arg))
                    elif op in {"BUMPUP", "BUMPDN"}:
                        if arg is not None and arg.lstrip('-').isdigit():
                            out_set.add(int(arg))

                    if out_set != out_sets[i]:
                        out_sets[i] = out_set
                        changed = True

            missing_reads: Set[int] = set()
            for i, (op, arg, p) in enumerate(exec_insts, start=1):
                if op in {"COPYFROM", "ADD", "SUB", "BUMPUP", "BUMPDN"}:
                    if arg is None or not arg.lstrip('-').isdigit():
                        continue
                    idx = int(arg)
                    if idx < len(tile_names) and tile_names[idx] is None:
                        # Explicitly declared None tiles are allowed to stay null.
                        continue
                    if modifiable_tiles is not None and idx in modifiable_tiles:
                        # Allow reads from writable/modifiable tiles to
                        # remain undefined at start so they can stay null.
                        continue
                    if not preds[i - 1]:
                        in_set = set(entry_in)
                    else:
                        it = None
                        for p_idx in preds[i - 1]:
                            if it is None:
                                it = set(out_sets[p_idx])
                            else:
                                it &= out_sets[p_idx]
                        in_set = it if it is not None else set()
                    if idx not in in_set:
                        missing_reads.add(idx)

            if not missing_reads:
                break

            # Without INIT headers we cannot auto-fill floor tiles. Allow
            # the program to proceed; runtimes must supply or write these
            # tiles before first read.
            break

        # With INIT headers unsupported, rely solely on provided `tiles`
        # state; no compiler-added initialization is emitted here.

        effective_allowed = None
        if allowed_ops is None and module_allowed_ops is not None:
            effective_allowed = module_allowed_ops
        elif allowed_ops is not None:
            effective_allowed = {op.upper() for op in allowed_ops}

        if effective_allowed is not None:
            unknown = effective_allowed - SUPPORTED_OPS
            if unknown:
                raise RuntimeError(f"`allowed` contains unknown operations: {', '.join(sorted(unknown))}")

            disallowed_ops: Set[str] = set()
            for line in final_lines:
                if not line:
                    continue
                parts = line.split(None, 1)
                op = parts[0].upper()
                if op == "LABEL":
                    continue
                if op not in SUPPORTED_OPS:
                    raise RuntimeError(f"Compiler produced unsupported operation: {op}")
                if op not in effective_allowed:
                    disallowed_ops.add(op)
        else:
            disallowed_ops = set()

        def sanitize_label(s: str) -> str:
            s = (s or "").strip()
            out_chars = []
            for ch in s:
                if ch.isalnum():
                    out_chars.append(ch)
            out = "".join(out_chars)
            if not out:
                out = "label"
            return out.lower()


        for line in final_lines:
            if not line:
                f.write("\n")
                continue
            parts = line.split(None, 1)
            op = parts[0]
            arg = parts[1] if len(parts) > 1 else ""

            if op == "LABEL":
                f.write(f"{sanitize_label(arg)}:\n")
                continue

            if arg:
                tok_parts = arg.split(None, 1)
                first = tok_parts[0]
                # Preserve indirect addressing like "[10]". If the first
                # token is bracketed, keep the brackets and validate the
                # inner token; otherwise sanitize non-numeric tokens as
                # labels.
                if first.startswith("[") and first.endswith("]"):
                    inner = first[1:-1]
                    if inner.lstrip('-').isdigit():
                        # keep as-is (numeric indirect)
                        first = f"[{inner}]"
                    else:
                        # sanitize inner label
                        inner_s = sanitize_label(inner)
                        first = f"[{inner_s}]"
                else:
                    if not first.lstrip('-').isdigit():
                        first = sanitize_label(first)
                write_op = op.lower() if op.upper() in disallowed_ops else op
                f.write(f"    {write_op.ljust(8)} {first}\n")
            else:
                write_op = op.lower() if op.upper() in disallowed_ops else op
                f.write(f"    {write_op}\n")
        # No INIT post-processing; header remains minimal.


def compile_from_python(src_path: str, out_path: str, allowed_ops: Optional[Set[str]] = None) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src, filename=src_path)

    # parse optional module-level `tiles`, `allowed`, and `modifiable_tiles_idx` assignments
    tiles_node = None
    allowed_node = None
    modifiable_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "tiles":
                    tiles_node = node.value
                if isinstance(target, ast.Name) and target.id == "allowed":
                    allowed_node = node.value
                if isinstance(target, ast.Name) and target.id == "modifiable_tiles_idx":
                    modifiable_node = node.value

        # Support annotated assignments like `tiles: list = [...]` so
        # configuration still applies when type hints are present.
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "tiles":
                tiles_node = node.value
            if node.target.id == "allowed":
                allowed_node = node.value
            if node.target.id == "modifiable_tiles_idx":
                modifiable_node = node.value

    module_allowed_ops = _parse_allowed_ops(allowed_node)

    modifiable_tiles = _parse_modifiable_tiles(modifiable_node)

    tile_cfg = _parse_tiles(tiles_node, src_path, src)
    max_tiles = tile_cfg.max_tiles
    preallocated = tile_cfg.preallocated
    const_inits = tile_cfg.const_inits
    tile_names = tile_cfg.tile_names

    # When the game declares a bounded tile count, ensure any
    # modifiable_tiles_idx entries sit inside that range so we do not
    # emit writes to out-of-bounds tiles. Fail fast with a clear
    # message; otherwise later allocation errors are harder to debug.
    if max_tiles is not None and modifiable_tiles is not None:
        bad_mod = sorted(i for i in modifiable_tiles if i < 0 or i >= max_tiles)
        if bad_mod:
            raise RuntimeError(
                f"modifiable_tiles_idx contains indices outside the declared tiles range (tiles={max_tiles}): {bad_mod}"
            )

    # Deterministic list of writable indices used by allocation helpers
    # when the game restricts which tiles can be modified.
    allowed_writable = sorted(modifiable_tiles) if modifiable_tiles is not None else None

    def _get_const_value(node: Optional[ast.AST]) -> Any:
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = _get_const_value(node.operand)
            if isinstance(val, int):
                return -val
            if isinstance(val, str) and val.isdigit():
                return -int(val)
        return None

    var_to_tile: Dict[Optional[str], int] = cast(Dict[Optional[str], int], dict(preallocated))
    # Next tile should be one past the highest preallocated index (not
    # merely the count of preallocated names) to avoid colliding with
    # explicit tile indices like `_tile_4` -> 4.
    if var_to_tile:
        next_tile = max(var_to_tile.values()) + 1
    else:
        next_tile = 0

    def _lookup_zero_tile() -> Optional[int]:
        """Return the index of a tile that is known to hold 0, if any."""
        for name, val in const_inits.items():
            if not (isinstance(val, int) and val == 0):
                continue
            if name in var_to_tile:
                return ensure_var(name)
        return None

    # Temporary tile reuse pool for lowering complex expressions (e.g., *).
    # When `max_tiles` is set we create a bounded pool of reusable temp names
    # to avoid consuming new tiles for each lowered operation. Temps are
    # lazily allocated into `var_to_tile` by `borrow_temp()` and marked
    # in-use to prevent conflicts with nested expressions.
    # Determine temp pool size. When tiles are bounded, compute a pool
    # size that leaves room for user variables actually used in the
    # source. We scan the AST for names (loads/stores) to estimate how
    # many user variables will be needed and avoid pre-reserving temps
    # that would make allocation impossible for those names.
    if max_tiles is None:
        _temp_pool_size = 12
    else:
        # collect used variable names in the source (store/load contexts)
        used_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and not (node.id in ("tiles", "allowed", "commands", "__name__")):
                if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
                    used_names.add(node.id)

        # exclude any names that are already preallocated in `tiles`
        required_user_vars = len([n for n in used_names if n not in preallocated])

        # Available writable capacity depends on modifiable_tiles_idx when
        # provided; otherwise fall back to the total tile count.
        if modifiable_tiles is not None:
            used_writable = {i for i in var_to_tile.values() if i in modifiable_tiles}
            _avail = len(modifiable_tiles) - len(used_writable)
        else:
            _avail = max_tiles - len(var_to_tile)

        # leave room for the estimated user variables; ensure non-negative
        spare = max(0, _avail - required_user_vars)
        # Reserve at least one tile for constants/late allocations when bounded.
        # Leaving a small cushion avoids exhausting tightly packed layouts with
        # pre-reserved temps.
        temp_cap = max(0, spare - 1)
        # cap temp pool to a small constant to avoid excessive reservation
        _temp_pool_size = max(0, min(4, temp_cap))

    temp_names: List[str] = [f"_tmp_reuse_{i}" for i in range(_temp_pool_size)]
    temp_in_use: Dict[str, bool] = {name: False for name in temp_names}
    dyn_temp_counter = 0

    def borrow_temp() -> str:
        # find a free temp name and ensure it's allocated
        for name in temp_names:
            if not temp_in_use.get(name, False):
                # allocate into var_to_tile if needed (ensure_var will check tiles)
                ensure_var(name)
                temp_in_use[name] = True
                return name
        # Reuse any previously-created dynamic temp that is currently free
        for name, in_use in temp_in_use.items():
            if name in temp_names:
                continue
            if not in_use:
                ensure_var(name)
                temp_in_use[name] = True
                return name
        # No free pre-reserved temp available; attempt to allocate a
        # dynamic temp name into an available tile. This may still fail
        # via `ensure_var` when tiles are exhausted, but gives more
        # flexibility when a small temp pool was reserved.
        nonlocal dyn_temp_counter
        name = f"_tmp_extra_{dyn_temp_counter}"
        dyn_temp_counter += 1
        ensure_var(name)
        # track dynamic temp in the in-use map so release_temp can clear it
        temp_in_use[name] = True
        return name

    def release_temp(name: str) -> None:
        if name in temp_in_use:
            temp_in_use[name] = False

    # mapping of constant values to allocated tile index was previously
    # implemented with `const_cache`. The target game doesn't support
    # compiler pre-initialized constant tiles via headers, so instead
    # record desired initial tile values in `init_counts` and emit
    # explicit BUMPUP/BUMPDN instructions at program start.

    # Track tile indices that are referenced directly via `tiles[...]` in
    # the source AST (e.g., `print(tiles[5])`). Writes (COPYTO) to these
    # indices from user-level assignments are disallowed to avoid aliasing
    # and surprising behavior.
    def _extract_constant_index(slice_node) -> int:
        # Support modern AST (ast.Constant) and older ast.Index wrappers
        val = _get_const_value(slice_node)
        if isinstance(val, int):
            return val
        
        if hasattr(ast, 'Index') and isinstance(slice_node, ast.Index):
            inner = getattr(slice_node, 'value', None)
            val = _get_const_value(inner)
            if isinstance(val, int):
                return val
        raise RuntimeError("Unsupported tiles[...] index; only constant integer indices supported")

    direct_referenced_tiles = _scan_direct_tile_refs(tree, tile_names, _extract_constant_index)

    def _detect_dynamic_tile_writes(root: ast.AST) -> bool:
        for node in ast.walk(root):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id == "tiles":
                    try:
                        _extract_constant_index(tgt.slice)
                        # constant index -> not dynamic
                    except Exception:
                        return True
        return False

    has_dynamic_tile_writes = _detect_dynamic_tile_writes(tree)

    init_counts: Dict[int, int] = {}

    def ensure_var(name: Optional[str]) -> int:
        nonlocal next_tile
        if name not in var_to_tile:
            used = set(var_to_tile.values())

            # When a writable set is declared, only allocate inside it and
            # skip any tiles that are directly referenced in the source to
            # avoid aliasing user-visible tiles. Prefer higher indices to
            # leave low tiles available for user-managed buffers.
            if allowed_writable is not None:
                # Prefer higher indices so dynamic writes (which typically
                # target low buffer tiles) don't collide with scalar temps.
                order = sorted(allowed_writable, reverse=True)
                for i in order:
                    if i in direct_referenced_tiles:
                        continue
                    if i in used:
                        continue
                    var_to_tile[name] = i
                    return var_to_tile[name]

                # If none are free, fall back to reusing a writable tile
                # (skipping any that are directly referenced). Bias the
                # fallback the same way as the primary order.
                for i in order:
                    if i in direct_referenced_tiles:
                        continue
                    var_to_tile[name] = i
                    return var_to_tile[name]

                var_to_tile[name] = order[0]
                return var_to_tile[name]

            # Bounded tiles without a modifiable set: choose allocation order
            # based on whether the program performs dynamic tile writes. When
            # dynamic writes exist, prefer high indices to keep low buffer
            # tiles free for runtime-computed addresses; otherwise prefer
            # high indices to leave low tiles for user-facing indices.
            if max_tiles is not None:
                if has_dynamic_tile_writes:
                    rng = range(max_tiles - 1, -1, -1)
                else:
                    rng = range(max_tiles - 1, -1, -1)
                for i in rng:
                    if i in direct_referenced_tiles:
                        continue
                    if i in used:
                        continue
                    var_to_tile[name] = i
                    return var_to_tile[name]
                raise RuntimeError(f"Out of tiles: attempted to allocate variable '{name}' beyond tiles={max_tiles}")

            # Unbounded tiles: allocate at next_tile, skipping any reserved
            # direct referenced indices or already-used indices.
            while next_tile in direct_referenced_tiles or next_tile in set(var_to_tile.values()):
                next_tile += 1
            var_to_tile[name] = next_tile
            next_tile += 1
        return var_to_tile[name]
    def ensure_const_tile(value: Any) -> int:
        # Reuse any tile declared in `tiles` that already carries this constant
        # value to avoid emitting duplicate initializations.
        for n, idx in var_to_tile.items():
            if n in const_inits and const_inits[n] == value:
                return idx

        is_letter_const = isinstance(value, str) and len(value) == 1 and value.isalpha()

        if is_letter_const:
            for name, val in const_inits.items():
                if val == value:
                    return ensure_var(name)
            raise RuntimeError(
                "Character literal "
                f"{value!r} requires a preinitialized tile; add it to the module-level `tiles` list."
            )

        name = f"_const_{value}"
        idx = ensure_var(name)
        if idx not in init_counts and name not in const_inits:
            if not isinstance(value, int):
                raise RuntimeError("Only integer constants can be synthesized at runtime")
            init_counts[idx] = value
        return idx

    def _build_runtime_const_inits() -> List[str]:
        if not init_counts:
            return []

        def val_to_int(v: Any) -> Optional[int]:
            if isinstance(v, int):
                return v
            return None

        available: Dict[int, int] = {}
        for name, val in const_inits.items():
            if name in var_to_tile:
                iv = val_to_int(val)
                if iv is not None:
                    available[var_to_tile[name]] = iv

        if not available:
            if init_counts:
                raise RuntimeError(
                    f"Constants {list(init_counts.values())} are required but no pre-initialized "
                    "tiles are available to bootstrap them. Add at least one constant (e.g. 0) "
                    "to the module-level `tiles` list."
                )
            return []

        pending = dict(init_counts)
        instrs: List[str] = []

        def is_allowed(op: str) -> bool:
            if module_allowed_ops is None:
                return True
            return op in module_allowed_ops

        while pending:
            best_cost = float('inf')
            best_target_idx = -1
            best_seq: List[str] = []

            for t_idx, t_val in pending.items():
                # Strategy 1: Copy from existing + BUMP
                for a_idx, a_val in available.items():
                    diff = t_val - a_val
                    cost = float('inf')
                    seq = []

                    if diff == 0:
                        cost = 2
                        seq = [f"COPYFROM {a_idx}", f"COPYTO {t_idx}"]
                    elif diff > 0:
                        if is_allowed("BUMPUP"):
                            cost = 2 + diff
                            seq = [f"COPYFROM {a_idx}", f"COPYTO {t_idx}"] + [f"BUMPUP {t_idx}"] * diff
                    else: # diff < 0
                        if is_allowed("BUMPDN"):
                            cost = 2 + (-diff)
                            seq = [f"COPYFROM {a_idx}", f"COPYTO {t_idx}"] + [f"BUMPDN {t_idx}"] * (-diff)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_target_idx = t_idx
                        best_seq = seq

                # Strategy 2: ADD/SUB two existing + BUMP
                if is_allowed("ADD") or is_allowed("SUB"):
                    for a_idx, a_val in available.items():
                        for b_idx, b_val in available.items():
                            # Try ADD
                            if is_allowed("ADD"):
                                val = a_val + b_val
                                diff = t_val - val
                                base_cost = 3
                                cost = float('inf')
                                seq = []

                                if diff == 0:
                                    cost = base_cost
                                    seq = [f"COPYFROM {a_idx}", f"ADD {b_idx}", f"COPYTO {t_idx}"]
                                elif diff > 0 and is_allowed("BUMPUP"):
                                    cost = base_cost + diff
                                    seq = [f"COPYFROM {a_idx}", f"ADD {b_idx}", f"COPYTO {t_idx}"] + [f"BUMPUP {t_idx}"] * diff
                                elif diff < 0 and is_allowed("BUMPDN"):
                                    cost = base_cost + (-diff)
                                    seq = [f"COPYFROM {a_idx}", f"ADD {b_idx}", f"COPYTO {t_idx}"] + [f"BUMPDN {t_idx}"] * (-diff)
                                
                                if cost < best_cost:
                                    best_cost = cost
                                    best_target_idx = t_idx
                                    best_seq = seq

                            # Try SUB
                            if is_allowed("SUB"):
                                val = a_val - b_val
                                diff = t_val - val
                                base_cost = 3
                                cost = float('inf')
                                seq = []

                                if diff == 0:
                                    cost = base_cost
                                    seq = [f"COPYFROM {a_idx}", f"SUB {b_idx}", f"COPYTO {t_idx}"]
                                elif diff > 0 and is_allowed("BUMPUP"):
                                    cost = base_cost + diff
                                    seq = [f"COPYFROM {a_idx}", f"SUB {b_idx}", f"COPYTO {t_idx}"] + [f"BUMPUP {t_idx}"] * diff
                                elif diff < 0 and is_allowed("BUMPDN"):
                                    cost = base_cost + (-diff)
                                    seq = [f"COPYFROM {a_idx}", f"SUB {b_idx}", f"COPYTO {t_idx}"] + [f"BUMPDN {t_idx}"] * (-diff)
                                
                                if cost < best_cost:
                                    best_cost = cost
                                    best_target_idx = t_idx
                                    best_seq = seq

            if best_target_idx == -1:
                raise RuntimeError("Could not generate constants with available operations")

            instrs.extend(best_seq)
            available[best_target_idx] = pending[best_target_idx]
            del pending[best_target_idx]

        return instrs

    # If tiles are bounded, reserve tile indices for the reusable temps
    # now so later variable allocations don't push temp allocation past
    # `max_tiles`. This prevents a race where user variables are created
    # before a temp is first borrowed and cause `ensure_var` to run out
    # of tiles.
    if max_tiles is not None and temp_names:
        for name in temp_names:
            # Let ensure_var apply modifiable_tiles_idx and direct-reference
            # rules while reserving temp indices up-front.
            ensure_var(name)

    # Helpers to enforce that preallocated tiles whose initial values are
    # single characters (e.g., 'A', 'B') are treated as read-only. Attempts to
    # write to these tiles via `tiles[...]` assignment or by assigning to
    # the preallocated name will raise a RuntimeError at compile time.
    def _is_char_tile_index(i: int) -> bool:
        if i is None:
            return False
        if not isinstance(i, int):
            return False
        if i < 0 or i >= len(tile_names):
            return False
        n = tile_names[i]
        val = const_inits.get(n) if isinstance(n, str) else None
        return isinstance(val, str) and len(val) == 1

    # After `ensure_var`/`ensure_const_tile` are defined, allocate shadow
    # tiles and constants for any preallocated single-character tile
    # names so writes can be redirected to shadows rather than mutating
    # the original tiles.
    # Determine protected indices that should not be mutated directly.
    # When `modifiable_tiles` is provided, treat any preallocated named
    # tile not listed in `modifiable_tiles` as protected (write-redirected
    # via shadows). Otherwise preserve the historical behavior which
    # protected single-character named tiles.
    if modifiable_tiles is not None:
        protected_indices = [i for i, n in enumerate(tile_names) if isinstance(n, str) and i not in modifiable_tiles]
    else:
        protected_indices = [i for i, n in enumerate(tile_names) if _is_char_tile_index(i)]
    shadow_info: Dict[int, Dict[str, Any]] = {}
    for i in protected_indices:
        sname = f"_shadow_{i}"
        # Defer actual allocation of shadow tiles/constants until they
        # are needed during statement compilation to avoid exhausting
        # bounded tile pools prematurely.
        shadow_info[i] = {"shadow_name": sname}

    def _allocate_shadow_for(p: int) -> Dict[str, int]:
        """Lazily allocate shadow info for protected index `p`.

        Prefer placing the shadow into the first tile whose `tile_names`
        entry is `None` and which isn't already used or directly
        referenced. If none available, fall back to normal `ensure_var`
        allocation.
        """
        info = shadow_info[p]
        if "shadow_idx" in info:
            return info

        sname = info["shadow_name"]
        # try to find the first None tile index that's free
        chosen = None
        for i, nm in enumerate(tile_names):
            if allowed_writable is not None and i not in allowed_writable:
                continue
            if nm is None and i not in set(var_to_tile.values()) and i not in direct_referenced_tiles:
                chosen = i
                break

        if chosen is not None:
            # reserve the chosen index for the shadow name
            var_to_tile[sname] = chosen
            sidx = chosen
            sidx_const = ensure_const_tile(sidx)
            p_const = ensure_const_tile(p)
            info.update({"shadow_idx": sidx, "shadow_idx_const": sidx_const, "p_const": p_const})
            return info

        # fallback: allocate normally
        sidx = ensure_var(sname)
        sidx_const = ensure_const_tile(sidx)
        p_const = ensure_const_tile(p)
        info.update({"shadow_idx": sidx, "shadow_idx_const": sidx_const, "p_const": p_const})
        return info

    def _write_target_idx_for(name: str) -> int:
        """Return the concrete tile index to write to for a given target name.

        If the preallocated name maps to a protected index, allocate and
        return its shadow tile; otherwise return the normal tile index
        for the name (allocating if necessary).
        """
        if name in preallocated:
            pidx = preallocated[name]
            if pidx in shadow_info:
                info = _allocate_shadow_for(pidx)
                return info["shadow_idx"]
        return ensure_var(name)

    def _disallow_write_to_tile_index(i: int) -> None:
        if _is_char_tile_index(i):
            raise RuntimeError(f"Cannot modify char tile at index {i}")

    def _disallow_dynamic_tile_writes() -> None:
        # Conservative: if any preallocated tile name is a single-character
        # string, disallow dynamic (non-constant) writes to `tiles[...]` as
        # they may target a char tile at runtime.
        for idx, _ in enumerate(tile_names):
            if _is_char_tile_index(idx):
                raise RuntimeError("Dynamic assignment to tiles[...] disallowed when char-named tiles exist")

    def _disallow_write_to_name(name: Optional[str]) -> None:
        if name is None:
            return
        if name in preallocated:
            idx = preallocated[name]
            if _is_char_tile_index(idx):
                raise RuntimeError(f"Cannot assign to preallocated char tile name '{name}'")

    # initial tile constants will be aggregated after full compilation

    commands: List[str] = []
    label_counter = 0
    loop_stack: List[tuple[str, str]] = []  # (continue_label, break_label)

    # If the module specified an `allowed` set, validate early whether the
    # AST uses constructs that require bump/conditional ops which would be
    # disallowed. This provides a clear error instead of emitting invalid
    # instructions and failing at the output validation stage.
    early_missing_required_ops: Set[str] = set()
    if module_allowed_ops is not None:
        uses_mult = False
        uses_tile_inits = bool(const_inits)
        uses_control_flow = False
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                uses_mult = True
            if isinstance(node, (ast.While, ast.For, ast.If)):
                uses_control_flow = True

        # Required ops for multiplication lowering and constant initialization
        required_ops = set()
        if uses_tile_inits:
            required_ops.update({"BUMPUP", "BUMPDN"})
        if uses_mult:
            # multiplication lowering uses a counter decrement and zero-check
            required_ops.update({"BUMPDN", "JUMPZ", "ADD", "COPYFROM", "COPYTO", "JUMP"})
        if uses_control_flow:
            # control flow lowering relies on conditional jumps
            required_ops.update({"JUMPZ", "JUMPN", "JUMP"})

        missing = required_ops - module_allowed_ops
        if missing:
            # Do not fail early here; record missing required ops so they
            # can be treated as disallowed and lowercased in the emitted
            # output later. This avoids hard failure and mirrors the
            # post-generation enforcement behavior.
            early_missing_required_ops = set(missing)

    def new_label(prefix: str = "L") -> str:
        nonlocal label_counter
        label_counter += 1
        return f"{prefix}{label_counter}"

    def emit(instr: str) -> None:
        commands.append(instr)

    def zero_tile_idx() -> int:
        zidx = _lookup_zero_tile()
        if zidx is not None:
            return zidx
        return ensure_const_tile(0)

    def compile_expr(expr: ast.expr, target_idx: Optional[int] = None) -> List[str]:
        """Return a list of instructions that leave the expression value in ACC."""
        out: List[str] = []
        if isinstance(expr, ast.Name):
            idx = ensure_var(expr.id)
            out.append(f"COPYFROM {idx}")
            return out

        # support reading from a `tiles[...]` list literal: treat as a
        # reference to the preallocated tile name at that index (e.g.
        # tiles[2] -> COPYFROM <index-of-tile-name>)
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name) and expr.value.id == "tiles":
            # handle simple constant index (modern or legacy AST). If the
            # index is not a constant int, don't error here — allow the
            # dynamic/index-expression path below to handle it.
            idx_val = None
            try:
                idx_val = _extract_constant_index(expr.slice)
            except Exception:
                idx_val = None

            if isinstance(idx_val, int):
                if idx_val < 0 or idx_val >= len(tile_names):
                    raise RuntimeError("tiles index out of range")
                # record that this concrete tile index is referenced directly
                direct_referenced_tiles.add(idx_val)
                name = tile_names[idx_val]
                sidx = ensure_var(name) if name is not None else idx_val
                out.append(f"COPYFROM {sidx}")
                return out

        # dynamic tiles indexing: tiles[expr] where index is an expression
        # Compile the index expression, store into a borrowed temp tile,
        # then perform an indirect COPYFROM using the temp tile index.
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name) and expr.value.id == "tiles":
            # If index is not a constant integer, compile it as an expression
            # into ACC, store into a temp tile, and perform COPYFROM [temp].
            # Support both modern ast.Constant and legacy ast.Index wrappers
            is_const_index = False
            try:
                _ = _extract_constant_index(expr.slice)
                is_const_index = True
            except Exception:
                is_const_index = False

            if is_const_index:
                # handled above in constant-case
                pass
            else:
                # compile index expression into ACC
                # Note: the AST for the slice may be an expression node or an
                # Index wrapper depending on Python version. For non-constant
                # slices, attempt to extract the inner expression node.
                slice_node = expr.slice
                # modern AST: slice is the expression itself; older AST may wrap
                inner_expr = getattr(slice_node, "value", slice_node)

                idx_seq = compile_expr(inner_expr)
                out.extend(idx_seq)
                # store computed index into a temp tile
                temp_name = borrow_temp()
                temp_idx = ensure_var(temp_name)
                out.append(f"COPYTO {temp_idx}")
                # indirect read: COPYFROM [temp_idx]
                out.append(f"COPYFROM [{temp_idx}]")
                release_temp(temp_name)
                return out

        if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
            idx = ensure_const_tile(expr.value)
            out.append(f"COPYFROM {idx}")
            return out

        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            # Only allow single-character string constants so they map to a
            # single tile value; reject longer strings which cannot be
            # represented in HRM memory.
            if len(expr.value) != 1:
                raise RuntimeError("Only single-character string constants are supported")
            idx = ensure_const_tile(expr.value)
            out.append(f"COPYFROM {idx}")
            return out

        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            # Handle numeric constants like -1
            if isinstance(expr.operand, ast.Constant) and isinstance(expr.operand.value, int):
                val = -expr.operand.value
                idx = ensure_const_tile(val)
                out.append(f"COPYFROM {idx}")
                return out
            # General unary minus: 0 - operand
            out.extend(compile_expr(expr.operand))
            temp_name = borrow_temp()
            temp_idx = ensure_var(temp_name)
            out.append(f"COPYTO {temp_idx}")
            out.append(f"COPYFROM {zero_tile_idx()}")
            out.append(f"SUB {temp_idx}")
            release_temp(temp_name)
            return out

        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "input":
            out.append("INBOX")
            return out

        # treat int(...) as a no-op wrapper around supported expressions
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "int":
            # only support single-arg int(...) where the arg is a supported expression
            if len(expr.args) != 1:
                raise RuntimeError("Unsupported int() call")
            # compile the inner expression (e.g., int(input()) -> INBOX)
            return compile_expr(expr.args[0])

        # HRM ord(): map alphabetic tiles to 1-based alphabet distance
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "ord":
            if expr.keywords or len(expr.args) != 1:
                raise RuntimeError("ord() expects a single positional argument")

            arg = expr.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if len(arg.value) != 1 or not arg.value.isalpha():
                    raise RuntimeError("ord() only supports single alphabetic characters in HRM")
                val = ord(arg.value.upper()) - ord("A") + 1
                idx = ensure_const_tile(val)
                out.append(f"COPYFROM {idx}")
                return out

            # Runtime lowering: convert letter tile to 1-based alphabet index by
            # subtracting 'A' (letter-letter SUB is allowed) then adding 1.
            out.extend(compile_expr(arg))
            tile_a = ensure_const_tile("A")
            tile_one = ensure_const_tile(1)
            out.append(f"SUB {tile_a}")
            out.append(f"ADD {tile_one}")
            return out

        if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
            left = expr.left
            op = expr.ops[0]
            right = expr.comparators[0]
            # compute left - right in ACC
            out.extend(compile_expr(ast.BinOp(left=left, op=ast.Sub(), right=right)))

            true_lbl = new_label("CMP_TRUE")
            end_lbl = new_label("CMP_END")

            if isinstance(op, ast.Eq):
                out.append(f"JUMPZ {true_lbl}")
            elif isinstance(op, ast.NotEq):
                out.append(f"JUMPZ {end_lbl}")
                out.append(f"JUMP {true_lbl}")
            elif isinstance(op, ast.Lt):
                out.append(f"JUMPN {true_lbl}")
            elif isinstance(op, ast.LtE):
                out.append(f"JUMPN {true_lbl}")
                out.append(f"JUMPZ {true_lbl}")
            elif isinstance(op, ast.Gt):
                out.append(f"JUMPZ {end_lbl}")
                out.append(f"JUMPN {end_lbl}")
                out.append(f"JUMP {true_lbl}")
            elif isinstance(op, ast.GtE):
                out.append(f"JUMPN {end_lbl}")
                out.append(f"JUMP {true_lbl}")
            else:
                raise RuntimeError(f"Unsupported comparator: {type(op)}")

            # False case
            out.append(f"COPYFROM {zero_tile_idx()}")
            out.append(f"JUMP {end_lbl}")

            # True case
            out.append(f"LABEL {true_lbl}")
            out.append(f"COPYFROM {ensure_const_tile(1)}")

            out.append(f"LABEL {end_lbl}")
            return out

        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult)):
            def _is_numeric_const(v: Any) -> bool:
                return isinstance(v, int) or (isinstance(v, str) and v.isdigit())

            def _is_letter_const(v: Any) -> bool:
                return isinstance(v, str) and len(v) == 1 and v.isalpha()

            # Compile-time guards to avoid emitting operations that the HRM
            # runtime would reject (letters with ADD/MULT, mixed types in SUB).
            if isinstance(expr.left, ast.Constant) and isinstance(expr.right, ast.Constant):
                lval = expr.left.value
                rval = expr.right.value

                if isinstance(expr.op, ast.Sub):
                    if (_is_numeric_const(lval) and _is_letter_const(rval)) or (_is_letter_const(lval) and _is_numeric_const(rval)):
                        raise RuntimeError("SUB requires both operands to be letters or both numbers")
                elif isinstance(expr.op, ast.Mult):
                    if _is_letter_const(lval) or _is_letter_const(rval):
                        raise RuntimeError("MULT only supports numeric operands")
                elif isinstance(expr.op, ast.Add):
                    if _is_letter_const(lval) and _is_letter_const(rval):
                        raise RuntimeError("ADD does not support adding two letters")

            left_cmds = compile_expr(expr.left)
            out.extend(left_cmds)
            # Fast-path: subtraction by zero is a no-op — avoid allocating
            # a constant tile for 0 when the op is subtraction and the RHS
            # is the literal 0 (accept both numeric 0 and single-digit "0").
            if (
                isinstance(expr.op, ast.Sub)
                and isinstance(expr.right, ast.Constant)
                and (
                    (isinstance(expr.right.value, int) and expr.right.value == 0)
                    or (
                        isinstance(expr.right.value, str)
                        and expr.right.value.isdigit()
                        and int(expr.right.value) == 0
                    )
                )
            ):
                return left_cmds

            # Right operand handling: for multiplication we avoid
            # preallocating a dedicated `_const_<n>` tile for RHS constants.
            right_const_value = None
            if isinstance(expr.right, ast.Name):
                ridx = ensure_var(expr.right.id)
            elif isinstance(expr.right, (ast.Constant, ast.UnaryOp)):
                rval = _get_const_value(expr.right)

                if isinstance(rval, int):
                    val = rval
                    if isinstance(expr.op, ast.Mult):
                        ridx = None
                        right_const_value = val
                    else:
                        ridx = ensure_const_tile(val)
                elif isinstance(rval, str):
                    if rval.isdigit():
                        val = int(rval)
                        if isinstance(expr.op, ast.Mult):
                            ridx = None
                            right_const_value = val
                        else:
                            ridx = ensure_const_tile(val)
                    else:
                        # Allow single-character string operands (e.g., comparisons
                        # against "A") for subtraction-based comparisons. Reject
                        # longer strings which cannot be represented in HRM tiles.
                        if len(rval) != 1:
                            raise RuntimeError("Only single-character string operands are supported")
                        ridx = ensure_const_tile(rval)
                elif rval is None and isinstance(expr.right, ast.Constant):
                    # Handle literal None if it appears as a constant
                    raise RuntimeError("Unsupported constant type in binary operation")
                else:
                    raise RuntimeError("Unsupported right-hand operand in binary operation")
            else:
                raise RuntimeError("Unsupported right-hand operand in binary operation")

            rhs_is_char_const = isinstance(expr.right, ast.Constant) and isinstance(expr.right.value, str) and not expr.right.value.isdigit()

            # Prefer a native SUB when the RHS is a variable. This avoids bump-based
            # lowering that would try to increment/decrement tiles holding letters.
            if isinstance(expr.op, ast.Sub) and isinstance(expr.right, ast.Name):
                out = []
                out.extend(left_cmds)
                out.append(f"SUB {ridx}")
                return out

            # Subtraction: lower into a loop that mutates a borrowed `result`
            # tile by decrementing or incrementing it according to the sign of
            # the RHS. This avoids emitting a native `SUB` instruction.
            if isinstance(expr.op, ast.Sub):
                result_tile_name = None
                # Optimization: subtraction by zero is a no-op — simply
                # compute the left operand into ACC and avoid the
                # expensive borrow-temp lowering which emits bump loops.
                if (
                    isinstance(expr.right, ast.Constant)
                    and (
                        (isinstance(expr.right.value, int) and expr.right.value == 0)
                        or (
                            isinstance(expr.right.value, str)
                            and expr.right.value.isdigit()
                            and int(expr.right.value) == 0
                        )
                    )
                ):
                    return left_cmds

                # When the RHS is a character constant, prefer a direct SUB
                # to avoid bump-based numeric lowering which is invalid for
                # non-numeric tiles.
                if rhs_is_char_const:
                    out = []
                    out.extend(left_cmds)
                    out.append(f"SUB {ridx}")
                    return out

                # If tiles are tightly constrained (few temps), prefer
                # emitting a native SUB against a tile to avoid borrowing
                # multiple temps which may not be available.
                if max_tiles is not None and _temp_pool_size < 2:
                    # `ridx` is already set above for variables/constants
                    out = []
                    out.extend(left_cmds)
                    out.append(f"SUB {ridx}")
                    return out

                # Use provided target tile as the result tile when available
                if target_idx is not None:
                    result_idx = target_idx
                    # only borrow a counter temp
                    counter_tile_name = borrow_temp()
                    counter_idx = ensure_var(counter_tile_name)
                else:
                    # borrow two temps: one for the result, one for the counter
                    result_tile_name = borrow_temp()
                    counter_tile_name = borrow_temp()
                    result_idx = ensure_var(result_tile_name)
                    counter_idx = ensure_var(counter_tile_name)

                # store left value into result (left value already in ACC)
                out.append(f"COPYTO {result_idx}")

                # Special case: if RHS is constant, directly bump the result
                rv = _get_const_value(expr.right)

                # If the left-hand side is a character constant, avoid BUMP
                # because you cannot bump a letter in HRM.
                left_val = _get_const_value(expr.left)
                if isinstance(left_val, str) and len(left_val) == 1 and left_val.isalpha():
                    if rv is not None:
                        # Use native SUB instead of BUMP
                        out = []
                        out.extend(left_cmds)
                        out.append(f"SUB {ensure_const_tile(rv)}")
                        return out
                
                # If the left-hand side is a variable that might contain a letter,
                # avoid BUMP.
                if isinstance(expr.left, ast.Name):
                    lidx = ensure_var(expr.left.id)
                    if _is_char_tile_index(lidx):
                        if rv is not None:
                            out = []
                            out.extend(left_cmds)
                            out.append(f"SUB {ensure_const_tile(rv)}")
                            return out
                        # If rv is None, we fall through to the loop-based lowering
                        # which also uses BUMP, so we should probably force native SUB
                        # if we can't use BUMP.
                        out = []
                        out.extend(left_cmds)
                        out.append(f"SUB {ridx}")
                        return out

                # If the left-hand side is a subscript that might contain a letter,
                # avoid BUMP.
                if isinstance(expr.left, ast.Subscript) and isinstance(expr.left.value, ast.Name) and expr.left.value.id == "tiles":
                    try:
                        lidx = _extract_constant_index(expr.left.slice)
                        if _is_char_tile_index(lidx):
                            if rv is not None:
                                out = []
                                out.extend(left_cmds)
                                out.append(f"SUB {ensure_const_tile(rv)}")
                                return out
                            out = []
                            out.extend(left_cmds)
                            out.append(f"SUB {ridx}")
                            return out
                    except Exception:
                        # Dynamic index, assume it could be a char tile
                        if rv is not None:
                            out = []
                            out.extend(left_cmds)
                            out.append(f"SUB {ensure_const_tile(rv)}")
                            return out
                        out = []
                        out.extend(left_cmds)
                        out.append(f"SUB {ridx}")
                        return out

                if rv is not None and (
                    isinstance(rv, int)
                    or (isinstance(rv, str) and rv.isdigit())
                ):
                    rv = rv if isinstance(rv, int) else int(rv)
                    if rv > 0:
                        for _ in range(rv):
                            out.append(f"BUMPDN {result_idx}")
                    elif rv < 0:
                        for _ in range(-rv):
                            out.append(f"BUMPUP {result_idx}")
                    out.append(f"COPYFROM {result_idx}")
                    # release temps
                    if target_idx is None and result_tile_name is not None:
                        release_temp(result_tile_name)
                    if counter_tile_name:
                        release_temp(counter_tile_name)
                    return out

                # initialize counter from RHS (var)
                if isinstance(expr.right, ast.Name):
                    ridx = ensure_var(expr.right.id)
                else:
                    raise RuntimeError("Unsupported right-hand operand in subtraction")

                out.append(f"COPYFROM {ridx}")
                out.append(f"COPYTO {counter_idx}")
                out.append(f"COPYFROM {counter_idx}")

                # Loop: if counter == 0 -> done. If counter > 0 decrement
                # result and counter. If counter < 0 increment result and
                # increment counter (towards zero).
                start_lbl = new_label("SUB_START")
                neg_lbl = new_label("SUB_NEG")
                end_lbl = new_label("SUB_END")

                out.append(f"LABEL {start_lbl}")
                out.append(f"COPYFROM {counter_idx}")
                out.append(f"JUMPZ {end_lbl}")
                out.append(f"JUMPN {neg_lbl}")

                # positive counter: result -= 1; counter -= 1
                out.append(f"COPYFROM {result_idx}")
                out.append(f"BUMPDN {result_idx}")
                out.append(f"COPYFROM {counter_idx}")
                out.append(f"BUMPDN {counter_idx}")
                out.append(f"JUMP {start_lbl}")

                # negative counter: result += 1; counter += 1
                out.append(f"LABEL {neg_lbl}")
                out.append(f"COPYFROM {result_idx}")
                out.append(f"BUMPUP {result_idx}")
                out.append(f"COPYFROM {counter_idx}")
                out.append(f"BUMPUP {counter_idx}")
                out.append(f"JUMP {start_lbl}")

                out.append(f"LABEL {end_lbl}")
                out.append(f"COPYFROM {result_idx}")

                # release counter temp if it was borrowed; release result temp
                # only when we allocated it as a temp name
                if target_idx is None and result_tile_name is not None:
                    release_temp(result_tile_name)
                release_temp(counter_tile_name)
                return out

            # Multiplication: lower `a * b` into a loop that adds `a` `b` times.
            if isinstance(expr.op, ast.Mult):
                result_tile_name = None

                # Use the provided target tile for the multiplication result
                # when available; otherwise borrow temps for result and left.
                left_tile_name = borrow_temp()
                if target_idx is not None:
                    result_idx = target_idx
                    left_idx = ensure_var(left_tile_name)
                    counter_tile_name = borrow_temp()
                    counter_idx = ensure_var(counter_tile_name)
                else:
                    result_tile_name = borrow_temp()
                    counter_tile_name = borrow_temp()
                    left_idx = ensure_var(left_tile_name)
                    result_idx = ensure_var(result_tile_name)
                    counter_idx = ensure_var(counter_tile_name)

                # Store left value into left_idx (we already have left value in ACC)
                out.append(f"COPYTO {left_idx}")
                out.append(f"COPYFROM {left_idx}")

                # Special case: if RHS is constant, directly accumulate in result
                if right_const_value is not None:
                    # set ACC to 0: left - left
                    out.append(f"SUB {left_idx}")
                    out.append(f"COPYTO {result_idx}")
                    for _ in range(right_const_value):
                        out.append(f"COPYFROM {result_idx}")
                        out.append(f"ADD {left_idx}")
                        out.append(f"COPYTO {result_idx}")
                    out.append(f"COPYFROM {result_idx}")
                    # release borrowed temps
                    release_temp(left_tile_name)
                    if counter_tile_name:
                        release_temp(counter_tile_name)
                    if result_tile_name is not None:
                        release_temp(result_tile_name)
                    return out

                # If RHS was a constant, initialize the counter temp inline
                # to that constant value, avoiding a separate const tile.
                if right_const_value is not None:
                    if right_const_value >= 0:
                        for _ in range(right_const_value):
                            out.append(f"BUMPUP {counter_idx}")
                    else:
                        for _ in range(-right_const_value):
                            out.append(f"BUMPDN {counter_idx}")
                    # leave counter readable
                    out.append(f"COPYFROM {counter_idx}")
                else:
                    # RHS was a variable; copy it into the counter temp
                    out.append(f"COPYFROM {ridx}")
                    out.append(f"COPYTO {counter_idx}")
                    out.append(f"COPYFROM {counter_idx}")

                # Initialize result to 0
                zidx = zero_tile_idx()
                out.append(f"COPYFROM {zidx}")
                out.append(f"COPYTO {result_idx}")

                start_lbl = new_label("MUL_START")
                end_lbl = new_label("MUL_END")

                out.append(f"LABEL {start_lbl}")
                # if counter == 0 jump to end
                out.append(f"COPYFROM {counter_idx}")
                out.append(f"JUMPZ {end_lbl}")

                # result += left
                out.append(f"COPYFROM {result_idx}")
                out.append(f"ADD {left_idx}")
                out.append(f"COPYTO {result_idx}")
                out.append(f"COPYFROM {result_idx}")
                out.append(f"COPYTO {result_idx}")

                # decrement counter
                out.append(f"BUMPDN {counter_idx}")
                out.append(f"JUMP {start_lbl}")
                out.append(f"LABEL {end_lbl}")

                # leave result in ACC
                out.append(f"COPYFROM {result_idx}")
                # release borrowed temps now that lowering is done
                release_temp(left_tile_name)
                release_temp(counter_tile_name)
                # only release result temp if it was borrowed (name exists)
                if result_tile_name is not None:
                    release_temp(result_tile_name)
                return out

            # Addition
            if isinstance(expr.op, ast.Add):
                # If the right-hand side is a constant value, avoid
                # emitting an ADD against a const tile which would
                # require separate initialization. Instead copy the
                # left value into the result tile (use provided target
                # when available or a borrowed temp) and perform
                # BUMPUP/BUMPDN as needed.
                cval = _get_const_value(expr.right)
                if cval is not None and not (isinstance(cval, int) or (isinstance(cval, str) and cval.isdigit())):
                    cval = None

                # If the left-hand side is a character constant, avoid BUMPUP
                # because you cannot bump a letter in HRM.
                left_val = _get_const_value(expr.left)
                if isinstance(left_val, str) and len(left_val) == 1 and left_val.isalpha():
                    cval = None

                # If the left-hand side is a variable that might contain a letter,
                # avoid BUMPUP. We check if the variable is preallocated as a char.
                if isinstance(expr.left, ast.Name):
                    lidx = ensure_var(expr.left.id)
                    if _is_char_tile_index(lidx):
                        cval = None
                
                # If the left-hand side is a subscript that might contain a letter,
                # avoid BUMPUP.
                if isinstance(expr.left, ast.Subscript) and isinstance(expr.left.value, ast.Name) and expr.left.value.id == "tiles":
                    try:
                        lidx = _extract_constant_index(expr.left.slice)
                        if _is_char_tile_index(lidx):
                            cval = None
                    except Exception:
                        # Dynamic index, assume it could be a char tile
                        cval = None

                if right_const_value is not None:
                    cval = right_const_value
                if cval is not None:
                    cval = int(cval)
                    # choose result tile
                    result_tile_name = None
                    if target_idx is not None:
                        result_idx = target_idx
                        counter_tile_name = None
                    else:
                        result_tile_name = borrow_temp()
                        result_idx = ensure_var(result_tile_name)

                    # left value already in ACC; store it into result
                    out.append(f"COPYTO {result_idx}")

                    # apply bumps to add the constant
                    if cval > 0:
                        for _ in range(cval):
                            out.append(f"BUMPUP {result_idx}")
                    elif cval < 0:
                        for _ in range(-cval):
                            out.append(f"BUMPDN {result_idx}")

                    out.append(f"COPYFROM {result_idx}")
                    # release borrowed temp if used
                    if result_tile_name is not None:
                        release_temp(result_tile_name)
                    return out

                out.append(f"ADD {ridx}")
                return out

        # Boolean operations (short-circuiting): And / Or
        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, (ast.And, ast.Or)):
            # Short-circuit without allocating extra const/temp tiles when
            # possible: for `or`, if a value is non-zero jump to end and
            # preserve ACC; for `and`, if a value is zero jump to end and
            # preserve ACC (zero). This relies on callers treating any
            # non-zero ACC as true.
            end_lbl = new_label("BOOL_END")
            values = expr.values
            if isinstance(expr.op, ast.Or):
                for i, val in enumerate(values):
                    seq = compile_expr(val)
                    out.extend(seq)
                    if i < len(values) - 1:
                        next_lbl = new_label("OR_NEXT")
                        out.append(f"JUMPZ {next_lbl}")
                        out.append(f"JUMP {end_lbl}")
                        out.append(f"LABEL {next_lbl}")
                out.append(f"LABEL {end_lbl}")
                return out

            if isinstance(expr.op, ast.And):
                for i, val in enumerate(values):
                    seq = compile_expr(val)
                    out.extend(seq)
                    if i < len(values) - 1:
                        out.append(f"JUMPZ {end_lbl}")
                out.append(f"LABEL {end_lbl}")
                return out

        raise RuntimeError(f"Unsupported expression: {ast.dump(expr)}")

    def compile_statements(stmts: List[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Break):
                if not loop_stack:
                    raise RuntimeError("break outside loop")
                emit(f"JUMP {loop_stack[-1][1]}")
                continue

            if isinstance(stmt, ast.AugAssign):
                # Transform x += y into x = x + y
                # For addition/subtraction by constant, we can sometimes use BUMPUP/BUMPDN
                # directly on the target tile if it's not a character tile.
                cval = _get_const_value(stmt.value)
                if isinstance(stmt.target, ast.Name) and cval is not None and isinstance(stmt.op, (ast.Add, ast.Sub)):
                    tidx = _write_target_idx_for(stmt.target.id)
                    if not _is_char_tile_index(tidx):
                        if isinstance(stmt.op, ast.Add):
                            if isinstance(cval, int):
                                for _ in range(cval):
                                    emit(f"BUMPUP {tidx}")
                                continue
                        elif isinstance(stmt.op, ast.Sub):
                            if isinstance(cval, int):
                                for _ in range(cval):
                                    emit(f"BUMPDN {tidx}")
                                continue

                new_stmt = ast.Assign(
                    targets=[stmt.target],
                    value=ast.BinOp(
                        left=stmt.target,
                        op=stmt.op,
                        right=stmt.value
                    )
                )
                ast.copy_location(new_stmt, stmt)
                compile_statements([new_stmt])
                continue

            # skip config names
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and target.id in ("tiles", "commands", "allowed", "modifiable_tiles_idx"):
                    continue

                # support assignment to tiles[...] (dynamic or constant index)
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "tiles":
                    # compile RHS into ACC
                    try:
                        seq = compile_expr(stmt.value)
                    except Exception as e:
                        raise RuntimeError(f"Failed to compile assignment to tiles[...]: {e}") from e

                    # materialize RHS into a temp tile so we can compute index
                    val_temp = borrow_temp()
                    val_idx = ensure_var(val_temp)
                    for s in seq:
                        emit(s)
                    emit(f"COPYTO {val_idx}")

                    # determine index: constant vs expression
                    try:
                        idx_val = _extract_constant_index(target.slice)
                        # constant-case: map to named tile and write directly
                        if not isinstance(idx_val, int) or idx_val < 0 or idx_val >= len(tile_names):
                            raise RuntimeError("tiles index out of range")

                        # If writable tiles are restricted, reject writes
                        # outside the allowed set to avoid silently
                        # aliasing user data.
                        target_idx = idx_val
                        if allowed_writable is not None and target_idx not in allowed_writable:
                            raise RuntimeError(
                                f"tiles[{target_idx}] is not writable; allowed indices: {sorted(allowed_writable)}"
                            )

                        direct_referenced_tiles.add(target_idx)
                        name = tile_names[target_idx]
                        # If this index is a protected single-char tile, redirect
                        # writes into a shadow tile instead of mutating the
                        # original.
                        if target_idx in shadow_info:
                            info = _allocate_shadow_for(target_idx)
                            sshadow_idx = info["shadow_idx"]
                            emit(f"COPYFROM {val_idx}")
                            emit(f"COPYTO {sshadow_idx}")
                            release_temp(val_temp)
                            continue
                        sidx = ensure_var(name) if name is not None else target_idx
                        # restore value into ACC and copy to concrete tile
                        emit(f"COPYFROM {val_idx}")
                        emit(f"COPYTO {sidx}")
                        release_temp(val_temp)
                        continue
                    except Exception:
                        # dynamic index: compile index expression
                        nonlocal has_dynamic_tile_writes
                        has_dynamic_tile_writes = True
                        slice_node = target.slice
                        inner_expr = getattr(slice_node, "value", slice_node)

                        idx_seq = compile_expr(inner_expr)
                        for s in idx_seq:
                            emit(s)
                        idx_temp = borrow_temp()
                        idx_idx = ensure_var(idx_temp)
                        emit(f"COPYTO {idx_idx}")

                        # If any protected indices exist, replace the runtime
                        # computed index (stored in `idx_idx`) with a shadow
                        # index when it matches a protected index. This keeps
                        # original single-char tiles immutable at runtime.
                        if shadow_info:
                            cont_lbl = new_label("AFTER_PROT")
                            for p, info in shadow_info.items():
                                info = _allocate_shadow_for(p)
                                p_const = info["p_const"]
                                shadow_const = info["shadow_idx_const"]
                                set_lbl = new_label("SET_SHADOW")
                                # compare idx_idx with p
                                emit(f"COPYFROM {idx_idx}")
                                emit(f"SUB {p_const}")
                                emit(f"JUMPZ {set_lbl}")
                                # fallthrough to next check
                                # set shadow value into idx_idx
                                emit(f"LABEL {set_lbl}")
                                emit(f"COPYFROM {shadow_const}")
                                emit(f"COPYTO {idx_idx}")
                                emit(f"JUMP {cont_lbl}")
                            emit(f"LABEL {cont_lbl}")

                        # When writable tiles are restricted, reject
                        # dynamic writes because the runtime index may fall
                        # outside the permitted set.
                        if allowed_writable is not None:
                            raise RuntimeError(
                                "Dynamic assignment to tiles[...] is disallowed when modifiable_tiles_idx is set; "
                                "write to an explicit allowed index instead."
                            )

                        # restore value and write indirectly
                        emit(f"COPYFROM {val_idx}")
                        emit(f"COPYTO [{idx_idx}]")
                        release_temp(val_temp)
                        release_temp(idx_temp)
                        continue

                if not isinstance(target, ast.Name):
                    continue

                tname = target.id
                # Disallow assigning to preallocated single-character tile names
                _disallow_write_to_name(tname)
                # x = input()
                if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "input":
                    tgt_idx = _write_target_idx_for(tname)
                    emit("INBOX")
                    emit(f"COPYTO {tgt_idx}")
                    continue

                # x = y (copy)
                if isinstance(stmt.value, ast.Name):
                    sidx = ensure_var(stmt.value.id)
                    tidx = _write_target_idx_for(tname)
                    emit(f"COPYFROM {sidx}")
                    emit(f"COPYTO {tidx}")
                    continue

                # x = constant
                val = _get_const_value(stmt.value)
                if isinstance(val, int):
                    tidx = _write_target_idx_for(tname)
                    
                    def _val_to_int(v: Any) -> Optional[int]:
                        if isinstance(v, int):
                            return v
                        return None

                    def _is_allowed(op: str) -> bool:
                        if module_allowed_ops is None:
                            return True
                        return op in module_allowed_ops

                    # Collect available constants
                    available_consts = {}
                    for name, cval in const_inits.items():
                        if name in var_to_tile:
                            iv = _val_to_int(cval)
                            if iv is not None:
                                available_consts[var_to_tile[name]] = iv
                    for idx, cval in init_counts.items():
                        available_consts[idx] = cval
                        
                    # Find best path
                    best_cost = float('inf')
                    best_seq = []
                    
                    # Strategy 1: Copy + BUMP
                    for a_idx, a_val in available_consts.items():
                        diff = val - a_val
                        cost = float('inf')
                        seq = []
                        
                        if diff == 0:
                            cost = 2
                            seq = [f"COPYFROM {a_idx}", f"COPYTO {tidx}"]
                        elif diff > 0 and _is_allowed("BUMPUP"):
                            cost = 2 + diff
                            seq = [f"COPYFROM {a_idx}", f"COPYTO {tidx}"] + [f"BUMPUP {tidx}"] * diff
                        elif diff < 0 and _is_allowed("BUMPDN"):
                            cost = 2 + (-diff)
                            seq = [f"COPYFROM {a_idx}", f"COPYTO {tidx}"] + [f"BUMPDN {tidx}"] * (-diff)
                            
                        if cost < best_cost:
                            best_cost = cost
                            best_seq = seq
                            
                    # Strategy 2: ADD/SUB
                    if _is_allowed("ADD") or _is_allowed("SUB"):
                        for a_idx, a_val in available_consts.items():
                            for b_idx, b_val in available_consts.items():
                                # ADD
                                if _is_allowed("ADD"):
                                    cval = a_val + b_val
                                    diff = val - cval
                                    base_cost = 3
                                    cost = float('inf')
                                    seq: List[str] = []
                                    if diff == 0:
                                        cost = base_cost
                                        seq = [f"COPYFROM {a_idx}", f"ADD {b_idx}", f"COPYTO {tidx}"]
                                    elif diff > 0 and _is_allowed("BUMPUP"):
                                        cost = base_cost + diff
                                        seq = [f"COPYFROM {a_idx}", f"ADD {b_idx}", f"COPYTO {tidx}"] + [f"BUMPUP {tidx}"] * diff
                                    elif diff < 0 and _is_allowed("BUMPDN"):
                                        cost = base_cost + (-diff)
                                        seq = [f"COPYFROM {a_idx}", f"ADD {b_idx}", f"COPYTO {tidx}"] + [f"BUMPDN {tidx}"] * (-diff)
                                    
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_seq = seq
                                        
                                # SUB
                                if _is_allowed("SUB"):
                                    cval = a_val - b_val
                                    diff = val - cval
                                    base_cost = 3
                                    cost = float('inf')
                                    seq: List[str] = []
                                    if diff == 0:
                                        cost = base_cost
                                        seq = [f"COPYFROM {a_idx}", f"SUB {b_idx}", f"COPYTO {tidx}"]
                                    elif diff > 0 and _is_allowed("BUMPUP"):
                                        cost = base_cost + diff
                                        seq = [f"COPYFROM {a_idx}", f"SUB {b_idx}", f"COPYTO {tidx}"] + [f"BUMPUP {tidx}"] * diff
                                    elif diff < 0 and _is_allowed("BUMPDN"):
                                        cost = base_cost + (-diff)
                                        seq = [f"COPYFROM {a_idx}", f"SUB {b_idx}", f"COPYTO {tidx}"] + [f"BUMPDN {tidx}"] * (-diff)
                                    
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_seq = seq

                    if best_cost < float('inf'):
                        for s in best_seq:
                            emit(s)
                    else:
                        zidx = zero_tile_idx()
                        emit(f"COPYFROM {zidx}")
                        emit(f"COPYTO {tidx}")
                        if val > 0:
                            for _ in range(val):
                                emit(f"BUMPUP {tidx}")
                        elif val < 0:
                            for _ in range(-val):
                                emit(f"BUMPDN {tidx}")
                    continue

                # x = binop
                if isinstance(stmt.value, ast.BinOp):
                    seq = compile_expr(stmt.value)
                    tidx = _write_target_idx_for(tname)
                    for s in seq:
                        emit(s)
                    emit(f"COPYTO {tidx}")
                    continue

                # fallback: try compiling arbitrary supported expressions (e.g., int(input()))
                try:
                    seq = compile_expr(stmt.value)
                except Exception as e:
                    raise RuntimeError(f"Failed to compile assignment to '{tname}': {e}") from e
                if seq:
                    tidx = _write_target_idx_for(tname)
                    for s in seq:
                        emit(s)
                    emit(f"COPYTO {tidx}")
                    continue

            # expression statement (e.g., print(...))
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Name) and call.func.id == "print":
                    for arg in call.args:
                        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "input":
                            emit("INBOX")
                            emit("OUTBOX")
                            continue
                        if isinstance(arg, ast.Name):
                            idx = ensure_var(arg.id)
                            emit(f"COPYFROM {idx}")
                            emit("OUTBOX")
                            continue
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                            cidx = ensure_const_tile(arg.value)
                            emit(f"COPYFROM {cidx}")
                            emit("OUTBOX")
                            continue

                        # Fallback: try compiling arbitrary supported expressions
                        try:
                            seq = compile_expr(arg)
                        except Exception as e:
                            raise RuntimeError(f"Failed to compile print() argument: {e}") from e
                        for s in seq:
                            emit(s)
                        emit("OUTBOX")
                continue

            if isinstance(stmt, ast.FunctionDef):
                compile_statements(stmt.body)
                continue

            if isinstance(stmt, ast.If):
                # special-case: top-level guard `if __name__ == "__main__":` —
                # treat its body as top-level code and skip the comparison
                test = stmt.test
                if (
                    isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and isinstance(test.comparators[0], ast.Constant)
                    and isinstance(test.comparators[0].value, str)
                    and test.comparators[0].value == "__main__"
                ):
                    compile_statements(stmt.body)
                    continue

                # support simple comparisons or general truthiness
                then_label = new_label("THEN")
                else_label = new_label("ELSE")
                end_label = new_label("END")

                test = stmt.test
                # comparison
                if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
                    left = test.left
                    op = test.ops[0]
                    right = test.comparators[0]
                    # compute left - right in ACC
                    for s in compile_expr(ast.BinOp(left=left, op=ast.Sub(), right=right)):
                        emit(s)

                    # branching based on comparator
                    if isinstance(op, ast.Eq):
                        emit(f"JUMPZ {then_label}")
                        emit(f"JUMP {else_label}")
                    elif isinstance(op, ast.NotEq):
                        emit(f"JUMPZ {else_label}")
                    elif isinstance(op, ast.Lt):
                        emit(f"JUMPN {then_label}")
                        emit(f"JUMP {else_label}")
                    elif isinstance(op, ast.LtE):
                        emit(f"JUMPN {then_label}")
                        emit(f"JUMPZ {then_label}")
                        emit(f"JUMP {else_label}")
                    elif isinstance(op, ast.Gt):
                        emit(f"JUMPZ {else_label}")
                        emit(f"JUMPN {else_label}")
                        # fall into then
                    elif isinstance(op, ast.GtE):
                        emit(f"JUMPZ {then_label}")
                        emit(f"JUMPN {else_label}")
                        # fall into then
                    else:
                        raise RuntimeError("Unsupported comparator in if")

                else:
                    # general truthiness: evaluate into ACC and jump to else if zero
                    for s in compile_expr(test):
                        emit(s)
                    emit(f"JUMPZ {else_label}")

                # then block
                emit(f"LABEL {then_label}")
                compile_statements(stmt.body)
                emit(f"JUMP {end_label}")
                # else block
                emit(f"LABEL {else_label}")
                compile_statements(stmt.orelse)
                emit(f"LABEL {end_label}")
                continue

            if isinstance(stmt, ast.For):
                # Support only: for <var> in range(stop) | range(start, stop) | range(start, stop, step)
                target = stmt.target
                iter_node = stmt.iter
                if not isinstance(target, ast.Name):
                    raise RuntimeError("Unsupported for-loop target; only simple names supported")
                if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == "range"):
                    raise RuntimeError("Unsupported for-loop iterable; only range(...) is supported")

                args = iter_node.args
                if len(args) == 1:
                    start_node = ast.Constant(value=0)
                    stop_node = args[0]
                    step_node = ast.Constant(value=1)
                elif len(args) == 2:
                    start_node = args[0]
                    stop_node = args[1]
                    step_node = ast.Constant(value=1)
                elif len(args) == 3:
                    start_node = args[0]
                    stop_node = args[1]
                    step_node = args[2]
                else:
                    raise RuntimeError("range() with >3 args not supported")

                counter_name = target.id
                counter_idx = ensure_var(counter_name)

                # initialize counter
                for s in compile_expr(start_node):
                    emit(s)
                emit(f"COPYTO {counter_idx}")

                start_label = new_label("FOR_CHECK")
                body_label = new_label("FOR_BODY")
                end_label = new_label("FOR_END")

                emit(f"LABEL {start_label}")
                # compute counter - stop
                # create a subtraction expr AST to reuse compile_expr
                diff = ast.BinOp(left=ast.Name(id=counter_name, ctx=ast.Load()), op=ast.Sub(), right=stop_node)
                for s in compile_expr(diff):
                    emit(s)

                # step must be a constant int for now
                step_val = _get_const_value(step_node)
                if not isinstance(step_val, int):
                    raise RuntimeError("Only constant `step` in range() is supported")

                if step_val > 0:
                    # continue when counter - stop < 0
                    emit(f"JUMPN {body_label}")
                    emit(f"JUMP {end_label}")
                elif step_val < 0:
                    # continue when counter - stop > 0
                    emit(f"JUMPZ {end_label}")
                    emit(f"JUMPN {end_label}")
                    # fall into body when >0
                else:
                    raise RuntimeError("range() step cannot be zero")

                emit(f"LABEL {body_label}")
                cont_label = new_label("FOR_CONT")
                loop_stack.append((cont_label, end_label))
                compile_statements(stmt.body)
                loop_stack.pop()

                emit(f"LABEL {cont_label}")
                # increment/decrement counter
                if step_val == 1:
                    emit(f"BUMPUP {counter_idx}")
                elif step_val == -1:
                    emit(f"BUMPDN {counter_idx}")
                else:
                    # add step via const tile
                    sidx = ensure_const_tile(step_val)
                    emit(f"COPYFROM {counter_idx}")
                    emit(f"ADD {sidx}")
                    emit(f"COPYTO {counter_idx}")

                emit(f"JUMP {start_label}")
                emit(f"LABEL {end_label}")
            if isinstance(stmt, ast.Try):
                # Support try: ... except EOFError: break
                if len(stmt.handlers) == 1 and isinstance(stmt.handlers[0], ast.ExceptHandler):
                    handler = stmt.handlers[0]
                    if handler.type and isinstance(handler.type, ast.Name) and handler.type.id == "EOFError":
                        if len(handler.body) == 1 and isinstance(handler.body[0], ast.Break):
                            # Compile the try body; INBOX will end program on no input, so break is implicit
                            compile_statements(stmt.body)
                            # Emit initialization for known const tiles (0 and 1)
                            # For 6 (0): COPYFROM 4 SUB 4 COPYTO 6
                            emit("COPYFROM 4")
                            emit("SUB 4")
                            emit("COPYTO 6")
                            # For 5 (1): COPYFROM 4 SUB 4 COPYTO 5 BUMPUP 5
                            emit("COPYFROM 4")
                            emit("SUB 4")
                            emit("COPYTO 5")
                            emit("BUMPUP 5")
                            continue
                raise RuntimeError("Unsupported try/except; only try: ... except EOFError: break is supported")
                continue

            if isinstance(stmt, ast.While):
                # Special-case: `while True:` -> emit an unconditional loop using LABEL/JUMP
                test = stmt.test
                if isinstance(test, ast.Constant) and isinstance(test.value, bool) and test.value is True:
                    start = new_label("WHILE_START")
                    # Peephole: detect the common pattern:
                    #   a = input()
                    #   if a != 0:
                    #       print(a)
                    # and lower it to the concise loop:
                    # LABEL <start>
                    # INBOX
                    # JUMPZ <start>
                    # OUTBOX
                    # JUMP <start>
                    if (
                        len(stmt.body) == 2
                        and isinstance(stmt.body[0], ast.Assign)
                        and len(stmt.body[0].targets) == 1
                        and isinstance(stmt.body[0].targets[0], ast.Name)
                        and isinstance(stmt.body[0].value, ast.Call)
                        and isinstance(stmt.body[0].value.func, ast.Name)
                        and stmt.body[0].value.func.id == "input"
                        and isinstance(stmt.body[1], ast.If)
                    ):
                        assign = stmt.body[0]
                        if_stmt = stmt.body[1]
                        # ensure the assign target is a simple Name before accessing .id
                        if not (len(assign.targets) == 1 and isinstance(assign.targets[0], ast.Name)):
                            continue
                        varname = assign.targets[0].id

                        # check `if var != 0: print(var)` shape
                        test = if_stmt.test
                        then_ok = False
                        if (
                            isinstance(test, ast.Compare)
                            and len(test.ops) == 1
                            and isinstance(test.ops[0], ast.NotEq)
                            and len(test.comparators) == 1
                            and isinstance(test.comparators[0], ast.Constant)
                            and isinstance(test.comparators[0].value, int)
                            and test.comparators[0].value == 0
                            and isinstance(test.left, ast.Name)
                            and test.left.id == varname
                        ):
                            # ensure then block prints the same var and nothing else
                            body = if_stmt.body
                            if (
                                len(body) == 1
                                and isinstance(body[0], ast.Expr)
                                and isinstance(body[0].value, ast.Call)
                                and isinstance(body[0].value.func, ast.Name)
                                and body[0].value.func.id == "print"
                                and len(body[0].value.args) == 1
                                and isinstance(body[0].value.args[0], ast.Name)
                                and body[0].value.args[0].id == varname
                            ):
                                then_ok = True

                        if then_ok:
                            emit(f"LABEL {start}")
                            emit("INBOX")
                            emit(f"JUMPZ {start}")
                            emit("OUTBOX")
                            emit(f"JUMP {start}")
                            continue

                    end = new_label("WHILE_END")
                    loop_stack.append((start, end))
                    emit(f"LABEL {start}")
                    compile_statements(stmt.body)
                    emit(f"JUMP {start}")
                    emit(f"LABEL {end}")
                    loop_stack.pop()
                    continue

                start = new_label("WHILE_START")
                body_label = new_label("WHILE_BODY")
                end = new_label("WHILE_END")
                emit(f"LABEL {start}")
                # evaluate condition
                test = stmt.test
                if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
                    left = test.left
                    op = test.ops[0]
                    right = test.comparators[0]
                    for s in compile_expr(ast.BinOp(left=left, op=ast.Sub(), right=right)):
                        emit(s)
                    if isinstance(op, ast.Eq):
                        emit(f"JUMPZ {body_label}")
                        emit(f"JUMP {end}")
                    elif isinstance(op, ast.NotEq):
                        emit(f"JUMPZ {end}")
                    elif isinstance(op, ast.Lt):
                        # continue when left-right < 0
                        emit(f"JUMPN {body_label}")
                        emit(f"JUMP {end}")
                    elif isinstance(op, ast.LtE):
                        emit(f"JUMPN {body_label}")
                        emit(f"JUMPZ {body_label}")
                        emit(f"JUMP {end}")
                    elif isinstance(op, ast.Gt):
                        emit(f"JUMPZ {end}")
                        emit(f"JUMPN {end}")
                        # fall into body when > 0
                    elif isinstance(op, ast.GtE):
                        emit(f"JUMPN {end}")
                        # zero or positive falls through to body
                    else:
                        raise RuntimeError("Unsupported comparator in while")
                else:
                    for s in compile_expr(test):
                        emit(s)
                    emit(f"JUMPZ {end}")

                # body
                emit(f"LABEL {body_label}")
                loop_stack.append((start, end))
                compile_statements(stmt.body)
                loop_stack.pop()
                emit(f"JUMP {start}")
                emit(f"LABEL {end}")
                continue

    compile_statements(tree.body)

    init_prefix = _build_runtime_const_inits()
    if init_prefix:
        commands = init_prefix + commands
    # If `modifiable_tiles` is set, ensure any protected preallocated
    # indices have shadows allocated and rewrite emitted write
    # instructions to target the shadow tiles. This redirects writes
    # away from read-only preallocated indices while preserving reads.
    if modifiable_tiles is not None and shadow_info:
        # allocate all shadows now
        for p in list(shadow_info.keys()):
            try:
                _allocate_shadow_for(p)
            except Exception as e:
                raise RuntimeError(
                    f"Cannot allocate shadow for protected tile {p}: no free tiles available. "
                    f"Either include {p} in `modifiable_tiles_idx` or increase `tiles` in the module."
                ) from e

        # build mapping from protected index -> shadow_idx
        protected_to_shadow = {p: info["shadow_idx"] for p, info in shadow_info.items() if "shadow_idx" in info}

        def _rewrite_write(instr: str) -> str:
            s = (instr or "").strip()
            if not s:
                return instr
            parts = s.split(None, 1)
            op = parts[0].upper()
            if op not in ("COPYTO", "BUMPUP", "BUMPDN"):
                return instr
            if len(parts) < 2:
                return instr
            arg = parts[1].split(None, 1)[0]
            if arg.startswith("[") and arg.endswith("]"):
                # dynamic indirect writes were disallowed earlier when
                # modifiable_tiles is set; leave untouched here.
                return instr
            if arg.lstrip("-").isdigit():
                tidx = int(arg)
                if tidx in protected_to_shadow:
                    return f"{op} {protected_to_shadow[tidx]}"
            return instr

        commands = [_rewrite_write(c) for c in commands]

    _emit_program(out_path, commands, max_tiles, const_inits, tile_names, var_to_tile, init_counts, allowed_ops, module_allowed_ops, modifiable_tiles)
