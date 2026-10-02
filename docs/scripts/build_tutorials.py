"""Convert jupytext-format .py notebooks into rendered MDX.

The pipeline is chapter-agnostic: pass a chapter name (default ``tutorials``)
and the script reads ``docs/<chapter>/*.py``, executes each notebook through
the ``tausurv-docs`` kernel, and writes:

- MDX to ``docs/src/content/docs/<chapter>/<slug>.mdx``
- Images to ``docs/public/<chapter>/<slug>/``

Code cells become fenced ``python`` blocks; outputs become image references
or styled ``<pre>`` text blocks depending on their MIME type, so inputs and
outputs are visually distinct in the rendered page.

Run from the docs directory:

    uv run --group docs python scripts/build_tutorials.py            # tutorials
    uv run --group docs python scripts/build_tutorials.py plotting   # plotting
"""

from __future__ import annotations

import base64
import re
import shutil
import sys
from pathlib import Path

import jupytext
from nbclient import NotebookClient

DOCS = Path(__file__).resolve().parent.parent

KERNEL_NAME = "tausurv-docs"
EXECUTE_TIMEOUT = 300

_PLOT_REPR_RE = re.compile(
    r"^(<(matplotlib\.|Axes\b|Figure\b)|[A-Z]\w*Display\()"
)


def _strip_style_blocks(html: str) -> str:
    """Remove ``<style>...</style>`` blocks from HTML cell outputs.

    Polars renders DataFrames as HTML containing a ``<style>`` element whose
    CSS contains literal ``{`` and ``}`` braces. MDX's parser reads those
    braces as JSX-expression delimiters and the page fails to compile.
    The data table itself does not need polars' inline CSS; default browser
    styling (plus any project-level table CSS) renders it cleanly.
    """
    return re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.DOTALL)


def _flatten_html(html: str) -> str:
    """Collapse interior whitespace so an HTML output sits on a single line.

    MDX's block-HTML detection needs the element to be either entirely on
    one line or surrounded by blank lines. Polars emits multi-line HTML
    (a leading ``<div>`` on its own line followed by content), which MDX
    reads as a partial block start and fails to close. Flattening to one
    line makes MDX accept it as a self-contained HTML block.
    """
    return re.sub(r"\s*\n\s*", "", html).strip()


def _yaml_quote(s: str) -> str:
    """Wrap a string in YAML double-quotes so colons and other punctuation
    in the value don't trip the frontmatter parser."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _is_plot_repr(text: str) -> bool:
    """Detect a plot-object repr that should be hidden in tutorial output.

    Plotting functions return Axes / Figure / Display objects; when the
    cell's last expression is one of those, Jupyter renders the repr as a
    text output alongside the actual figure. The repr adds noise without
    value when a figure is also being shown.
    """
    return bool(_PLOT_REPR_RE.match(text.strip()))


def _finished_progress_bars(text: str) -> str:
    """Last frame of each completed tqdm bar in a stderr chunk, or ``""``.

    tqdm redraws a bar in place with carriage returns; the final frame of a
    bar that reached 100% is the one line worth keeping.
    """
    frames = [frame.strip() for line in text.split("\n") for frame in line.split("\r")]
    last_by_bar = {f.split("100%|")[0]: f for f in frames if "100%|" in f}
    return "\n".join(last_by_bar.values())


def _cell_output_html(text: str) -> str:
    """Wrap a text/plain output in a styled <pre> block.

    Bypasses expressive-code's code-block treatment so the result has no
    copy button, no language label, and can be styled independently of
    source-code blocks.
    """
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # MDX reads a bare brace as the start of a JSX expression, even inside <pre>.
    escaped = escaped.replace("{", "&#123;").replace("}", "&#125;")
    return f'<pre class="cell-output">{escaped}</pre>'


def render_output(
    output, slug: str, img_dir: Path, idx: int, chapter: str,
) -> str | None:
    """Render a single cell output to a markdown chunk.

    Returns None when the output should be skipped (empty stream, unknown
    output type).
    """
    otype = output.get("output_type")
    if otype == "stream":
        if output.get("name") == "stderr":
            # Warnings and other stderr noise are not part of the tutorial's
            # intended output and belong in the build log, not the page. The
            # one exception is a finished tqdm progress bar, which is part of
            # what the reader would see: keep its last frame only.
            bars = _finished_progress_bars(output.get("text", ""))
            return _cell_output_html(bars) if bars else None
        text = output.get("text", "").rstrip("\n")
        return _cell_output_html(text) if text else None
    if otype in ("execute_result", "display_data"):
        data = output.get("data", {})
        if "image/svg+xml" in data:
            filename = f"output_{idx:02d}.svg"
            (img_dir / filename).write_text(data["image/svg+xml"])
            return f"![](/{chapter}/{slug}/{filename})"
        if "image/png" in data:
            filename = f"output_{idx:02d}.png"
            (img_dir / filename).write_bytes(base64.b64decode(data["image/png"]))
            return f"![](/{chapter}/{slug}/{filename})"
        # Prefer text/plain over text/html: polars' HTML repr is multi-line
        # with embedded <style> blocks that MDX parses brittlely. The ASCII
        # text repr renders cleanly in our cell-output styling and is robust.
        if "text/plain" in data:
            text = data["text/plain"].rstrip("\n")
            if not text or _is_plot_repr(text):
                return None
            return _cell_output_html(text)
        if "text/html" in data:
            return _flatten_html(_strip_style_blocks(data["text/html"]))
    if otype == "error":
        text = "\n".join(output.get("traceback", []))
        return _cell_output_html(text)
    return None


def render_notebook(nb, slug: str, img_dir: Path, chapter: str) -> str:
    """Render an executed notebook to MDX markdown body."""
    parts: list[str] = []
    output_idx = 0

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            parts.append(cell.source.rstrip())
            parts.append("")
        elif cell.cell_type == "code":
            source = cell.source.strip()
            if not source:
                continue
            parts.append("```python")
            parts.append(source)
            parts.append("```")
            parts.append("")
            for output in _merge_streams(cell.outputs):
                rendered = render_output(
                    output, slug, img_dir, output_idx, chapter,
                )
                if rendered is None:
                    continue
                parts.append(rendered)
                parts.append("")
                output_idx += 1

    return "\n".join(parts).rstrip() + "\n"


def _merge_streams(outputs: list) -> list:
    """Join consecutive chunks of the same stream into one output.

    The kernel flushes stdout and stderr in pieces while a cell runs, so a
    single ``print`` loop or progress bar can arrive as several stream
    messages. Jupyter front-ends concatenate them; the page should too.
    """
    merged: list = []
    for output in outputs:
        previous = merged[-1] if merged else None
        same_stream = (
            previous is not None
            and output.get("output_type") == "stream"
            and previous.get("output_type") == "stream"
            and previous.get("name") == output.get("name")
        )
        if same_stream:
            merged[-1] = {**previous, "text": previous["text"] + output["text"]}
        else:
            merged.append(output)
    return merged


def extract_frontmatter(md: str, fallback_name: str) -> tuple[str, str, str]:
    """Pull the first H1 as title and the paragraph below it as description.

    Strip both from the body so they aren't duplicated in the rendered page.
    """
    lines = md.splitlines()
    title: str | None = None
    description: str | None = None
    body_start = 0

    for i, line in enumerate(lines):
        if not line.startswith("# "):
            continue
        title = line[2:].strip()
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        desc_lines: list[str] = []
        while (
            j < len(lines)
            and lines[j].strip()
            and not lines[j].startswith(("#", "```", "!["))
        ):
            desc_lines.append(lines[j].strip())
            j += 1
        if desc_lines:
            description = " ".join(desc_lines)
        body_start = j
        while body_start < len(lines) and not lines[body_start].strip():
            body_start += 1
        break

    if title is None:
        title = fallback_name.replace("_", " ").capitalize()
    if description is None:
        description = ""

    body = "\n".join(lines[body_start:]).strip()
    return title, description, body


def build_notebook(
    py_path: Path, mdx_dir: Path, img_dir_root: Path, chapter: str,
) -> None:
    name = py_path.stem
    slug = name.replace("_", "-")
    print(f"building {chapter}/{name}")

    nb = jupytext.read(py_path)

    client = NotebookClient(
        nb,
        kernel_name=KERNEL_NAME,
        timeout=EXECUTE_TIMEOUT,
        resources={"metadata": {"path": str(py_path.parent)}},
    )
    client.execute()

    img_dir = img_dir_root / slug
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)

    body_md = render_notebook(nb, slug, img_dir, chapter)
    title, description, body = extract_frontmatter(body_md, name)
    mdx = (
        f"---\n"
        f"title: {_yaml_quote(title)}\n"
        f"description: {_yaml_quote(description)}\n"
        f"---\n\n"
        f"{body}\n"
    )

    mdx_path = mdx_dir / f"{slug}.mdx"
    mdx_path.write_text(mdx)
    print(f"  wrote {mdx_path.relative_to(DOCS)}")


def build_chapter(chapter: str) -> None:
    source_dir = DOCS / chapter
    mdx_dir = DOCS / "src" / "content" / "docs" / chapter
    img_dir = DOCS / "public" / chapter

    if not source_dir.exists():
        print(f"no source directory at {source_dir}")
        return
    mdx_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    py_files = sorted(
        p for p in source_dir.glob("*.py") if not p.name.startswith("_")
    )
    if not py_files:
        print(f"no notebooks in {source_dir}")
        return

    for py_path in py_files:
        try:
            build_notebook(py_path, mdx_dir, img_dir, chapter)
        except Exception as exc:
            print(f"FAILED {py_path.name}: {exc}", file=sys.stderr)
            raise


def main() -> None:
    chapter = sys.argv[1] if len(sys.argv) > 1 else "tutorials"
    build_chapter(chapter)


if __name__ == "__main__":
    main()
