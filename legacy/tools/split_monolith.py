"""
Task 1 tool: mechanically split the legacy stardust_core.run_stardust() monolith
into one Python module per engine stage, preserving behaviour exactly.

How it works
------------
The legacy engine is a single 6,000-line function whose "modules" share state
through closure variables. For each stage (a contiguous line range of the
function body) we use the AST to find:

  * outputs  - names bound at the stage's top level (published to ``ctx``)
  * inputs   - names the stage reads that an earlier stage published
  * imports  - import-bound names used here but imported upstream (re-imported
               locally instead of being passed through ``ctx``)

Each stage becomes ``def run(ctx): <inputs> <original code verbatim> <publish>``.
The original code keeps its 4-space indentation, which is exactly the
indentation a function body needs, so the logic is copied byte-for-byte.

The tool also reports two hazards that a closure->module split can introduce:
  * forward references - a nested function reads a name only defined later
  * late binding       - a nested function reads a name that a later stage
                         rebinds before the function is called

Run:  python split_monolith.py <path/to/stardust_core.py> <out_package_dir>
"""
from __future__ import annotations

import ast
import builtins
import json
import symtable
import sys
from dataclasses import dataclass, field
from pathlib import Path

# (module_name, first_line, title) - first_line is 1-indexed in stardust_core.py;
# each stage ends where the next begins. Boundaries sit on the legacy
# "# ======" section banners.
STAGES = [
    ("m0_config",          28,   "M0 - Global configuration (+ adaptive SIZE regime detection)"),
    ("m1_data",            553,  "M1 - Market data fetching & cleaning"),
    ("m2a_factor_cov",     757,  "M2 (2A-2E) - Factor construction, factor covariance, baselines, factor-model returns"),
    ("m2b_signals_views",  1142, "M2 (2B.5-2B.7) - Fundamentals, signals, systematic Black-Litterman views"),
    ("m2c_bl_selection",   1674, "M2 (2F-2K) - Black-Litterman returns, method selection, shrinkage, checks"),
    ("m3_optimizer",       1966, "M3 - Portfolio optimizer (max Sharpe / min variance)"),
    ("m4_frontier",        2298, "M4 - Efficient frontier (random-weights Monte Carlo)"),
    ("m5_simulation",      2702, "M5 - Forward simulation (Cholesky t-distribution Monte Carlo)"),
    ("m6_risk_stress",     3119, "M6 - Risk metrics & stress testing"),
    ("m8_backtest",        3610, "M8 - Walk-forward backtest"),
    ("m7_report",          4659, "M7 - Excel report generator"),
]
FUNC_PARAMS = ["params", "tickers", "output_dir", "status_callback"]
BUILTINS = set(dir(builtins))


@dataclass
class StageInfo:
    name: str
    title: str
    start: int
    end: int
    src_lines: list[str]
    bound: set[str] = field(default_factory=set)          # top-level bindings (non-import)
    import_bound: dict[str, str] = field(default_factory=dict)  # name -> import stmt
    loaded: set[str] = field(default_factory=set)         # every Name load, any depth
    needed: set[str] = field(default_factory=set)          # reads that resolve outside the stage
    rebound_after_read: set[str] = field(default_factory=set)  # read from upstream, then rebound here
    func_free: dict[str, set[str]] = field(default_factory=dict)  # nested def -> free names
    calls: set[str] = field(default_factory=set)          # names called anywhere
    first_load: dict[str, int] = field(default_factory=dict)
    first_for: dict[str, int] = field(default_factory=dict)
    first_store: dict[str, int] = field(default_factory=dict)

    def is_carryover(self, n: str) -> bool:
        """Rebound here before it is ever read -> the upstream value is incidental."""
        load = self.first_load.get(n, 10**9)
        return ((n in self.first_for and load >= self.first_for[n])
                or (n in self.first_store and self.first_store[n] < load))


class TopLevelBinder(ast.NodeVisitor):
    """Collect names bound in the stage's own scope (not inside nested defs)."""

    def __init__(self):
        self.bound: set[str] = set()
        self.import_bound: dict[str, str] = {}

    def _bind_target(self, node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                self.bound.add(n.id)

    def visit_Assign(self, node):
        for t in node.targets:
            self._bind_target(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node):
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            nm = a.asname or a.name.split(".")[0]
            stmt = f"import {a.name}" + (f" as {a.asname}" if a.asname else "")
            self.import_bound.setdefault(nm, stmt)

    def visit_ImportFrom(self, node):
        for a in node.names:
            nm = a.asname or a.name
            stmt = f"from {'.' * node.level}{node.module or ''} import {a.name}" + (
                f" as {a.asname}" if a.asname else "")
            self.import_bound.setdefault(nm, stmt)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)      # do not descend: its locals are its own

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.bound.add(node.name)

    def visit_Lambda(self, node):
        pass

    # comprehension variables do not leak in py3 -> skip their targets
    def visit_ListComp(self, node):
        for g in node.generators:
            self.visit(g.iter)

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        for g in node.generators:
            self.visit(g.iter)


def function_free_names(fn: ast.FunctionDef) -> set[str]:
    """Names a nested def reads that it does not bind itself (approximate)."""
    local: set[str] = set()
    a = fn.args
    for arg in a.posonlyargs + a.args + a.kwonlyargs:
        local.add(arg.arg)
    if a.vararg:
        local.add(a.vararg.arg)
    if a.kwarg:
        local.add(a.kwarg.arg)
    binder = TopLevelBinder()
    for stmt in fn.body:
        binder.visit(stmt)
    local |= binder.bound | set(binder.import_bound)
    loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return loads - local - BUILTINS


def free_and_bound(src: str) -> tuple[set[str], set[str]]:
    """Exact scope analysis: (names the stage reads from outside, names it binds).

    symtable knows that `except ... as _e` inside a nested def is local to that
    def, which an AST walk cannot tell from a read of an upstream value.
    """
    st = symtable.symtable(src, "<stage>", "exec")
    bound = {s.get_name() for s in st.get_symbols() if s.is_assigned() or s.is_imported()}
    referenced: set[str] = set()

    def walk(tab):
        for s in tab.get_symbols():
            if s.is_referenced() and s.is_global():   # resolves to the stage's scope
                referenced.add(s.get_name())
        for child in tab.get_children():
            walk(child)

    walk(st)
    return referenced - bound - BUILTINS, bound


class EvalOrder:
    """Record (load|store, name) events at stage scope in EVALUATION order.

    Needed because a name can be both read from upstream and rebound here
    (``output_dir = pathlib.Path(output_dir)``): only the order tells us whether
    the stage consumes the upstream value. Nested def/class bodies are skipped -
    their reads happen at call time, after the definitions have run.
    """

    def __init__(self):
        self.events: list[tuple[str, str]] = []

    def expr(self, node):
        if node is None:
            return
        if isinstance(node, ast.Name):
            self.events.append(("load" if isinstance(node.ctx, ast.Load) else "store", node.id))
            return
        if isinstance(node, ast.Lambda):
            return                          # body runs at call time, not here
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            # AST order puts the element first, but evaluation runs the generators
            # first - otherwise every comprehension variable looks read-before-write
            for gen in node.generators:
                self.expr(gen.iter)
                self.expr(gen.target)
                for cond in gen.ifs:
                    self.expr(cond)
            if isinstance(node, ast.DictComp):
                self.expr(node.key)
                self.expr(node.value)
            else:
                self.expr(node.elt)
            return
        for ch in ast.iter_child_nodes(node):
            self.expr(ch)

    def body(self, stmts):
        for s in stmts:
            self.stmt(s)

    def stmt(self, node):
        if isinstance(node, ast.Assign):
            self.expr(node.value)
            for t in node.targets:
                self.expr(t)
        elif isinstance(node, ast.AnnAssign):
            self.expr(node.value)
            self.expr(node.target)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                self.events.append(("load", node.target.id))
            self.expr(node.value)
            self.expr(node.target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in getattr(node, "decorator_list", []):
                self.expr(d)
            args = getattr(node, "args", None)
            if args is not None:
                for d in list(args.defaults) + [d for d in args.kw_defaults if d]:
                    self.expr(d)
            self.events.append(("store", node.name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                self.events.append(("store", a.asname or a.name.split(".")[0]))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self.expr(node.iter)
            self.expr(node.target)
            self.body(node.body)
            self.body(node.orelse)
        elif isinstance(node, ast.While):
            self.expr(node.test)
            self.body(node.body)
            self.body(node.orelse)
        elif isinstance(node, ast.If):
            self.expr(node.test)
            self.body(node.body)
            self.body(node.orelse)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self.expr(item.context_expr)
                self.expr(item.optional_vars)
            self.body(node.body)
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            self.body(node.body)
            for h in node.handlers:
                self.expr(h.type)
                if h.name:
                    self.events.append(("store", h.name))
                self.body(h.body)
            self.body(node.orelse)
            self.body(node.finalbody)
        else:
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, ast.stmt):
                    self.stmt(ch)
                else:
                    self.expr(ch)


def read_before_write(tree: ast.Module) -> set[str]:
    """Names whose first stage-scope event is a load -> the upstream value is used."""
    ev = EvalOrder()
    ev.body(tree.body)
    first: dict[str, str] = {}
    for kind, name in ev.events:
        first.setdefault(name, kind)
    return {n for n, k in first.items() if k == "load"}


def analyse(stage: StageInfo) -> None:
    dedented = []
    for ln in stage.src_lines:
        if ln.startswith("    "):
            dedented.append(ln[4:])
        elif ln.strip() == "":
            dedented.append("")
        else:
            dedented.append(ln)   # e.g. inside a triple-quoted string
    source = "\n".join(dedented)
    stage.needed, sym_bound = free_and_bound(source)
    tree = ast.parse(source)
    binder = TopLevelBinder()
    for stmt in tree.body:
        binder.visit(stmt)
    stage.bound = binder.bound | (sym_bound - set(binder.import_bound))
    # a name bound here but READ first still consumes the upstream value
    stage.needed |= read_before_write(tree) & stage.bound
    stage.rebound_after_read = read_before_write(tree) & stage.bound
    stage.import_bound = binder.import_bound
    stage.loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    stage.calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for stmt in tree.body:
        for n in ast.walk(stmt):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stage.func_free[n.name] = function_free_names(n)
    # first line where each name is (a) loaded at stage scope, (b) bound as a for-target
    first_load: dict[str, int] = {}
    first_for: dict[str, int] = {}
    first_store: dict[str, int] = {}

    def scan(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_store.setdefault(node.name, node.lineno)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.For):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    first_for.setdefault(t.id, node.lineno)
        if isinstance(node, ast.ExceptHandler) and node.name:
            # `except X as e` binds e, then DELETES it when the block ends, so an
            # upstream stage can never hand it over - it is always carry-over.
            first_store.setdefault(node.name, node.lineno)
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                first_load.setdefault(node.id, node.lineno)
            else:
                first_store.setdefault(node.id, node.lineno)
        for child in ast.iter_child_nodes(node):
            scan(child)

    for stmt in tree.body:
        scan(stmt)
    stage.first_load = first_load
    stage.first_for = first_for
    stage.first_store = first_store


def fmt_tuple(names) -> str:
    names = sorted(names, key=str.lower)
    if not names:
        return "()"
    body = "".join(f"    {n!r},\n" for n in names)
    return "(\n" + body + ")"


def main(core_path: str, out_dir: str) -> None:
    src = Path(core_path).read_text(encoding="utf-8").splitlines()
    total = len(src)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Sanity: the function signature we expect
    assert src[26].startswith("def run_stardust("), src[26]

    stages: list[StageInfo] = []
    for i, (name, start, title) in enumerate(STAGES):
        end = STAGES[i + 1][1] - 1 if i + 1 < len(STAGES) else total
        lines = src[start - 1:end]
        st = StageInfo(name=name, title=title, start=start, end=end, src_lines=lines)
        analyse(st)
        stages.append(st)

    # Pre-pass: which stage produces each value a later stage reads as data?
    # (the latest earlier stage that binds it = the one whose value is seen)
    produced_for_later: dict[int, set[str]] = {i: set() for i in range(len(stages))}
    seen: set[str] = set(FUNC_PARAMS)
    for j, st in enumerate(stages):
        inp = st.needed & seen
        for n in inp:
            for i in range(j - 1, -1, -1):
                if n in stages[i].bound:
                    produced_for_later[i].add(n)
                    break
        seen |= st.bound

    published: dict[str, str] = {p: "run_stardust()" for p in FUNC_PARAMS}   # name -> stage that last bound it
    import_stmts: dict[str, str] = {}
    report = {"stages": [], "forward_refs": [], "late_binding": []}

    for idx, st in enumerate(stages):
        # names read here that an earlier stage (or the signature) published
        inputs = st.needed & set(published)
        # names that are import-bound upstream but not re-imported here
        need_imports = sorted((st.needed & set(import_stmts)) - set(published))
        # forward references: read here but nothing upstream provides them
        unknown = st.needed - set(published) - set(import_stmts)
        for nm in sorted(unknown):
            later = [s.name for s in stages[idx + 1:] if nm in s.bound]
            report["forward_refs"].append({"stage": st.name, "name": nm, "bound_later_in": later})

        # Inputs the stage also rebinds are pulled defensively: their read may sit in
        # a branch that never runs, in which case the monolith would not have needed
        # the upstream value either.
        carryover = inputs & st.rebound_after_read
        data_inputs = inputs - carryover
        # outputs = values a later stage reads as data from THIS stage (the real
        # downstream contract); everything bound here is still published.
        outputs = sorted(produced_for_later[idx], key=str.lower)
        everything = sorted(st.bound)
        st_imports = [import_stmts[n] for n in need_imports]

        # late binding: nested defs here reading X where a later stage rebinds X
        for fn, free in st.func_free.items():
            for nm in free:
                rebinds = [s.name for s in stages[idx + 1:] if nm in s.bound]
                callers = [s.name for s in stages[idx + 1:] if fn in s.loaded]
                if rebinds and callers:
                    report["late_binding"].append({"stage": st.name, "function": fn, "name": nm,
                                                   "rebound_in": rebinds, "called_in": callers})

        # ---- emit module ----
        body = "\n".join(st.src_lines)
        in_lines = "\n".join(f"    {n} = ctx[{n!r}]" for n in sorted(data_inputs, key=str.lower))
        if carryover:
            in_lines += ("\n    # ---- read from upstream, then rebound here (generated) ----\n"
                         + "\n".join(f"    if {n!r} in ctx: {n} = ctx[{n!r}]"
                                     for n in sorted(carryover, key=str.lower)))
        imp_lines = "\n".join(f"    {s}" for s in st_imports)
        text = (
            f'"""\nLegacy Stardust 1.x - {st.title}\n\n'
            f"Mechanically extracted from stardust_core.py lines {st.start}-{st.end} by\n"
            f"legacy/tools/split_monolith.py. The code inside run() is the original,\n"
            f"byte-for-byte; only the input/publish scaffolding around it is generated.\n"
            f"Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.\n"
            f'"""\n'
            f"from ._bootstrap import ensure_bootstrap\n\n"
            f"ensure_bootstrap()\n\n"
            f"#: data this stage reads from earlier stages (its upstream contract)\n"
            f"INPUTS = {fmt_tuple(data_inputs)}\n\n"
            f"#: legacy loop variables that leak in from earlier stages; pulled only so the\n"
            f"#: split behaves exactly like the monolith's single shared scope\n"
            f"CARRYOVER = {fmt_tuple(carryover)}\n\n"
            f"#: data this stage produces that a later stage reads (downstream contract)\n"
            f"OUTPUTS = {fmt_tuple(outputs)}\n\n"
            f"#: everything bound at stage scope; all of it is published to ctx\n"
            f"_PUBLISHED = {tuple(everything)!r}\n\n\n"
            f"def run(ctx: dict) -> None:\n"
            f"    # ---- inputs from upstream stages (generated) ----\n"
            f"{in_lines if in_lines else '    pass'}\n"
            + (f"    # ---- imports made upstream, repeated locally (generated) ----\n{imp_lines}\n" if imp_lines else "")
            + f"    # ---- original code: stardust_core.py lines {st.start}-{st.end} ----\n"
            f"{body}\n"
            f"    # ---- publish (generated) ----\n"
            f"    _scope = locals()\n"
            f"    ctx.update({{k: _scope[k] for k in _PUBLISHED if k in _scope}})\n"
        )
        (out / f"{st.name}.py").write_text(text, encoding="utf-8")

        for n in st.bound:
            published[n] = st.name
        for n, s in st.import_bound.items():
            import_stmts.setdefault(n, s)
        report["stages"].append({"stage": st.name, "lines": f"{st.start}-{st.end}",
                                 "inputs": sorted(data_inputs, key=str.lower),
                                 "carryover": sorted(carryover, key=str.lower),
                                 "outputs": outputs,
                                 "n_published": len(everything),
                                 "imports_repeated": need_imports})

    (out / "_split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
