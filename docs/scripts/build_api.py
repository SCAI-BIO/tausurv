"""Generate API reference MDX from tausurv source.

Walks the packages with griffe, parses Numpy-style docstrings, and emits one
MDX page per public submodule under ``docs/src/content/docs/api/``.

Run from the docs directory:

    uv run python scripts/build_api.py
"""

from __future__ import annotations

import re
import shutil
import textwrap
from pathlib import Path

import griffe
from griffe import Alias, Class, Function, Module, ParameterKind

DOCS = Path(__file__).resolve().parent.parent
REPO = DOCS.parent
API_OUT = DOCS / "src" / "content" / "docs" / "api"

PACKAGES = ["tausurv"]
SEARCH_PATHS = [str(REPO / "tausurv")]

NUMPY_SECTIONS = (
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Attributes",
    "Notes",
    "Examples",
    "References",
    "See Also",
    "Warnings",
)
ITEM_SECTIONS = {"Parameters", "Returns", "Yields", "Raises", "Attributes"}
CROSS_REF = re.compile(r":(?:func|meth|class|mod|obj|attr|data|exc):`([^`]+)`")


def parse_numpy(text: str) -> tuple[str, list[tuple[str, str]]]:
    text = textwrap.dedent(text).strip("\n")
    lines = text.splitlines()
    headers: list[tuple[str, int]] = []
    for i in range(len(lines) - 1):
        head = lines[i].strip()
        rule = lines[i + 1].strip()
        if (
            head in NUMPY_SECTIONS
            and rule
            and set(rule) == {"-"}
            and len(rule) >= len(head)
        ):
            headers.append((head, i))
    if not headers:
        return text.rstrip(), []
    desc = "\n".join(lines[: headers[0][1]]).rstrip()
    sections: list[tuple[str, str]] = []
    for k, (name, i) in enumerate(headers):
        start = i + 2
        end = headers[k + 1][1] if k + 1 < len(headers) else len(lines)
        body = "\n".join(lines[start:end]).rstrip()
        sections.append((name, textwrap.dedent(body)))
    return desc, sections


def render_items(body: str) -> str:
    out: list[str] = []
    cur: tuple[str, str | None, list[str]] | None = None
    for line in body.splitlines():
        if not line.strip():
            if cur is not None:
                cur[2].append("")
            continue
        indented = line[0] in " \t"
        if not indented:
            if cur is not None:
                out.append(_format_item(*cur))
            name, sep, typ = line.partition(":")
            cur = (name.strip(), typ.strip() if sep else None, [])
        else:
            if cur is None:
                cur = ("", None, [])
            cur[2].append(line.lstrip())
    if cur is not None:
        out.append(_format_item(*cur))
    return "\n".join(out)


def _format_item(name: str, typ: str | None, desc_lines: list[str]) -> str:
    desc = " ".join(part for part in (line.strip() for line in desc_lines) if part)
    pieces: list[str] = []
    if name:
        pieces.append(f"**`{name}`**")
    if typ:
        pieces.append(f"*{typ}*")
    head = " — ".join(pieces)
    if head and desc:
        return f"- {head} — {desc}"
    if head:
        return f"- {head}"
    return f"- {desc}"


def render_docstring(raw: str) -> str:
    raw = CROSS_REF.sub(r"`\1`", raw)
    desc, sections = parse_numpy(raw)
    chunks: list[str] = []
    if desc:
        chunks.append(desc)
    for name, body in sections:
        chunks.append(f"\n#### {name}\n")
        if name in ITEM_SECTIONS:
            chunks.append(render_items(body))
        else:
            chunks.append(body.strip("\n"))
    return "\n".join(chunks).strip()


def format_params(params, drop_self: bool = False) -> str:
    parts: list[str] = []
    seen_kw_only = False
    for p in params:
        if drop_self and p.name == "self":
            continue
        if p.kind == ParameterKind.var_positional:
            parts.append(f"*{p.name}")
            continue
        if p.kind == ParameterKind.var_keyword:
            parts.append(f"**{p.name}")
            continue
        if p.kind == ParameterKind.keyword_only and not seen_kw_only:
            parts.append("*")
            seen_kw_only = True
        piece = p.name
        if p.annotation is not None:
            piece += f": {p.annotation}"
        if p.default is not None:
            piece += f" = {p.default}"
        parts.append(piece)
    return ", ".join(parts)


def signature(obj) -> str:
    if isinstance(obj, Function):
        params = format_params(obj.parameters, drop_self=True)
        ret = f" -> {obj.returns}" if obj.returns is not None else ""
        return f"{obj.name}({params}){ret}"
    if isinstance(obj, Class):
        init = obj.members.get("__init__")
        if isinstance(init, Function):
            params = format_params(init.parameters, drop_self=True)
            return f"class {obj.name}({params})"
        return f"class {obj.name}"
    return obj.name


def public_members(module: Module) -> list[tuple[str, object]]:
    """Locally-defined public classes, functions, and submodules.

    Aliases (re-exports / imports) are skipped — each symbol is documented
    on the page for the module that defines it.
    """
    items: list[tuple[str, object]] = []
    for name, member in module.members.items():
        if name.startswith("_"):
            continue
        if isinstance(member, Alias):
            continue
        if isinstance(member, (Class, Function, Module)):
            items.append((name, member))
    return items


def render_symbol(name: str, obj) -> str:
    kind = "class" if isinstance(obj, Class) else "function"
    chunks: list[str] = [
        f'### `{name}` <Badge text="{kind}" variant="tip" size="small" />',
        "",
        "```python",
        signature(obj).replace(obj.name, name, 1),
        "```",
        "",
    ]
    if obj.docstring is not None and obj.docstring.value.strip():
        chunks.append(render_docstring(obj.docstring.value))
        chunks.append("")
    return "\n".join(chunks)


def render_module(module: Module) -> str:
    title = module.canonical_path
    chunks: list[str] = [
        "---",
        f"title: {title}",
        f"description: API reference for {title}.",
        "---",
        "",
        'import { Badge } from "@astrojs/starlight/components";',
        "",
    ]
    if module.docstring is not None and module.docstring.value.strip():
        chunks.append(render_docstring(module.docstring.value))
        chunks.append("")
    subs = [m for _n, m in public_members(module) if isinstance(m, Module)]
    if subs:
        chunks.append("## Submodules\n")
        for m in subs:
            href = "/api/" + m.canonical_path.replace(".", "/") + "/"
            chunks.append(f"- [`{m.canonical_path}`]({href})")
        chunks.append("")
    syms = [(n, m) for n, m in public_members(module) if not isinstance(m, Module)]
    if syms:
        chunks.append("## Reference\n")
        for n, m in syms:
            chunks.append(render_symbol(n, m))
    return "\n".join(chunks).rstrip() + "\n"


def walk_modules(module: Module, seen: set[str] | None = None):
    seen = seen if seen is not None else set()
    if module.canonical_path in seen:
        return
    seen.add(module.canonical_path)
    yield module
    for _, m in public_members(module):
        if isinstance(m, Module):
            yield from walk_modules(m, seen)


def output_path(module: Module) -> Path:
    parts = module.canonical_path.split(".")
    has_subs = any(isinstance(m, Module) for _, m in public_members(module))
    if has_subs:
        return API_OUT.joinpath(*parts) / "index.mdx"
    if len(parts) == 1:
        return API_OUT / f"{parts[0]}.mdx"
    return API_OUT.joinpath(*parts[:-1]) / f"{parts[-1]}.mdx"


def clean_output() -> None:
    if not API_OUT.exists():
        return
    for child in API_OUT.iterdir():
        if child.name == "index.mdx":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    clean_output()
    API_OUT.mkdir(parents=True, exist_ok=True)
    for package in PACKAGES:
        try:
            root = griffe.load(package, search_paths=SEARCH_PATHS)
        except ModuleNotFoundError:
            print(f"skip: {package} not found on search paths")
            continue
        for module in walk_modules(root):
            content = render_module(module)
            out_path = output_path(module)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content)
            print(f"wrote {out_path.relative_to(DOCS)}")


if __name__ == "__main__":
    main()
