"""MCP server for running memllm evaluation arms on Kaggle's free GPUs.

The loop this closes: build a notebook from kaggle/PASTE_THESE_CELLS.md, push it
to Kaggle, poll until it finishes, pull the result JSON back into results/.
Without it every arm requires a human to copy cells into a browser and download
a zip afterwards.

Auth is never handled here. The Kaggle client reads OAuth credentials cached by
`kaggle auth login`, or a token from $KAGGLE_API_TOKEN / ~/.kaggle/access_token.
This server only reports whether that worked -- it never accepts, logs, or
echoes a token, because anything passed through a tool call is written to the
session transcript.

Register with:
    claude mcp add kaggle -- /Users/ishant/memllm/.venv/bin/python \
        /Users/ishant/memllm/kaggle_mcp/server.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CELLS = REPO / "kaggle" / "PASTE_THESE_CELLS.md"

mcp = MCPServer(
    name="kaggle",
    instructions=(
        "Run memllm evaluation arms on Kaggle GPUs. Typical flow: "
        "kaggle_auth_status -> kaggle_list_cells -> kaggle_push_notebook -> "
        "kaggle_status (poll) -> kaggle_fetch_results. Kernels are pushed "
        "private by default. GPU quota is ~30h/week and is consumed by every "
        "run, so prefer pushing one notebook with several arms over many "
        "small pushes."
    ),
)


# --------------------------------------------------------------------------
# auth / client
# --------------------------------------------------------------------------

_AUTH_HELP = (
    "Not authenticated to Kaggle. Run ONE of these in your own shell "
    "(prefix with ! in Claude Code so it runs in this session):\n"
    "  !kaggle auth login                 # OAuth, nothing to copy around\n"
    "  !export KAGGLE_API_TOKEN=<token>   # from kaggle.com/settings/api\n"
    "Do not paste the token into chat -- the transcript is logged. "
    "The token file at ~/.kaggle/access_token also works."
)


def _api():
    """Authenticate lazily, and treat importing the client as part of that.

    Never at module import: the MCP server is spawned at session start, and an
    unauthenticated import would kill it before any tool could explain why.

    The `import` sits inside the try because the Kaggle client touches the
    filesystem while loading -- an unwritable or missing HOME raises OSError
    from the import line, not from authenticate(). Worse, a half-failed import
    leaves a partially-initialised module behind, so every later call reports a
    bogus "circular import" instead of the real cause. Purging the module on
    failure keeps the next attempt honest.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as e:  # noqa: BLE001 - surfaced to the caller as text
        for name in [m for m in list(sys.modules) if m.startswith("kaggle")]:
            sys.modules.pop(name, None)
        raise RuntimeError(
            f"{_AUTH_HELP}\n\nunderlying error: {type(e).__name__}: {e}"
        )
    return api


def _username(api) -> str:
    """Resolve the account slug for building kernel refs."""
    for attr in ("username",):
        v = getattr(api, attr, None)
        if isinstance(v, str) and v:
            return v
    cfg = getattr(api, "config_values", None)
    if isinstance(cfg, dict) and cfg.get("username"):
        return str(cfg["username"])
    for env in ("KAGGLE_USERNAME",):
        if os.environ.get(env):
            return os.environ[env]
    # Last resort: any kernel I own carries "owner/slug" in its ref.
    try:
        mine = api.kernels_list(mine=True, page_size=1) or []
        if mine and getattr(mine[0], "ref", None):
            return str(mine[0].ref).split("/")[0]
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError(
        "Could not determine your Kaggle username. Pass owner= explicitly, "
        "or set KAGGLE_USERNAME."
    )


def _qualify(slug: str, api) -> str:
    return slug if "/" in slug else f"{_username(api)}/{slug}"


# --------------------------------------------------------------------------
# notebook assembly
# --------------------------------------------------------------------------

_CELL_HEADER = re.compile(r"^##\s+(Cell\s+[^\n]*)$", re.M)
_PY_FENCE = re.compile(r"```python\n(.*?)```", re.S)


def _parse_cells(md: str) -> list[dict[str, str]]:
    """Split the notebook doc into named cells, each with its python source.

    Keyed on '## Cell ...' headings so a caller can push Cell 4b alone rather
    than re-running a 3-hour ladder to get one 45-minute control arm.
    """
    out: list[dict[str, str]] = []
    marks = list(_CELL_HEADER.finditer(md))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(md)
        body = md[m.end():end]
        code = "\n\n".join(f.group(1).rstrip() for f in _PY_FENCE.finditer(body))
        if code.strip():
            out.append({"name": m.group(1).strip(), "code": code})
    return out


def _notebook(sources: list[str]) -> dict[str, Any]:
    return {
        "cells": [
            {
                # nbformat 5.1.4+ warns on missing ids and will hard-error later
                "id": f"cell{i}",
                "cell_type": "code",
                "source": s.splitlines(keepends=True),
                "metadata": {},
                "execution_count": None,
                "outputs": [],
            }
            for i, s in enumerate(sources)
        ],
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


@mcp.tool()
def kaggle_auth_status() -> str:
    """Check whether Kaggle authentication works, without revealing the token.

    Call this first. Every other tool fails with the same setup instructions
    if auth is missing, so this just gets the answer cheaply.
    """
    try:
        api = _api()
    except RuntimeError as e:
        return str(e)
    try:
        user = _username(api)
    except RuntimeError as e:
        return f"Authenticated, but: {e}"
    return f"Authenticated as {user}."


@mcp.tool()
def kaggle_list_cells(cells_file: str = "") -> str:
    """List the named cells available in the notebook source document.

    Returns each '## Cell ...' heading with its line count, so a caller can
    choose which subset to push instead of pushing everything.
    """
    path = Path(cells_file) if cells_file else DEFAULT_CELLS
    if not path.exists():
        return f"no such file: {path}"
    cells = _parse_cells(path.read_text())
    if not cells:
        return f"no ```python cells found under '## Cell' headings in {path}"
    lines = [f"{len(cells)} cells in {path}:"]
    for c in cells:
        lines.append(f"  - {c['name']}  ({len(c['code'].splitlines())} lines)")
    return "\n".join(lines)


@mcp.tool()
def kaggle_push_notebook(
    slug: str,
    title: str = "",
    cells_file: str = "",
    include: str = "",
    enable_gpu: bool = True,
    enable_internet: bool = True,
    is_private: bool = True,
    accelerator: str = "",
    prepend_code: str = "",
) -> str:
    """Build a notebook from the cells document and push it to Kaggle.

    This starts a GPU run on Kaggle's servers and consumes weekly quota, so
    confirm with the user before calling it unless they have already asked for
    this specific run.

    Args:
        slug: kernel slug, e.g. "memllm-controls". Bare slugs are qualified
            with your username. Pushing an existing slug REPLACES that kernel's
            source and starts a new run.
        title: display title; defaults to the slug.
        cells_file: path to the markdown cells doc; defaults to
            kaggle/PASTE_THESE_CELLS.md.
        include: comma-separated substrings selecting cells by heading, e.g.
            "Cell 1,Cell 2,Cell 4b". Empty means every cell, in document order.
            Cells are emitted in document order regardless of the order here,
            because later cells depend on names bound by earlier ones.
        enable_gpu: request a GPU. Turn off for CPU-only arms (the retrieval
            sweep) to avoid burning GPU quota on work that does not need it.
        enable_internet: required to pip install and to clone the repo.
        is_private: default True. Pushing public would publish the code and any
            output under your account.
        accelerator: optional explicit accelerator string passed to the Kaggle
            API (e.g. "nvidiaTeslaT4"). Leave blank for the account default.
        prepend_code: python inserted as the FIRST cell, before Cell 0. Use it
            to narrow a run without editing the document, e.g.
            'SIZES = ["7b", "14b"]' so Cell 1 pulls two models instead of four.
            Cell 0 would overwrite it, so the override is re-applied in a cell
            emitted immediately after Cell 0.

    Returns the kernel ref and URL, or the API error.
    """
    path = Path(cells_file) if cells_file else DEFAULT_CELLS
    if not path.exists():
        return f"no such file: {path}"
    cells = _parse_cells(path.read_text())
    if not cells:
        return f"no python cells found in {path}"

    if include.strip():
        wanted = [w.strip().lower() for w in include.split(",") if w.strip()]
        chosen = [c for c in cells
                  if any(w in c["name"].lower() for w in wanted)]
        missing = [w for w in wanted
                   if not any(w in c["name"].lower() for c in cells)]
        if missing:
            return (f"no cell matches {missing}. Available:\n" +
                    "\n".join(f"  - {c['name']}" for c in cells))
    else:
        chosen = cells
    if not chosen:
        return "cell selection matched nothing"

    try:
        api = _api()
        ref = _qualify(slug, api)
    except RuntimeError as e:
        return str(e)

    work = Path(tempfile.mkdtemp(prefix="kaggle_push_"))
    try:
        sources = [c["code"] for c in chosen]
        if prepend_code.strip():
            # Cell 0 assigns the config names, so an override placed before it
            # would simply be overwritten. Insert directly after Cell 0 when it
            # is present; otherwise it goes first.
            after0 = next(
                (i + 1 for i, c in enumerate(chosen)
                 if c["name"].lower().startswith("cell 0")), 0
            )
            sources.insert(
                after0, "# --- override injected by kaggle_push_notebook ---\n"
                + prepend_code.strip()
            )
        nb = _notebook(sources)
        (work / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
        meta = {
            "id": ref,
            "title": title or slug,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": bool(is_private),
            "enable_gpu": bool(enable_gpu),
            "enable_internet": bool(enable_internet),
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        kw = {"folder": str(work)}
        if accelerator.strip():
            kw["acc"] = accelerator.strip()
        resp = api.kernels_push(**kw)
    except Exception as e:  # noqa: BLE001
        return f"push failed: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(work, ignore_errors=True)

    err = getattr(resp, "error", None)
    if err:
        return f"push rejected by Kaggle: {err}"
    included = ", ".join(c["name"] for c in chosen)
    return (
        f"pushed {ref}\n"
        f"  cells: {included}\n"
        f"  gpu={enable_gpu} internet={enable_internet} private={is_private}\n"
        f"  url: https://www.kaggle.com/code/{ref.replace('/', '/')}\n"
        f"Poll with kaggle_status('{slug}')."
    )


@mcp.tool()
def kaggle_status(slug: str) -> str:
    """Return the run status of a kernel: queued, running, complete, or error.

    Cheap; safe to poll. Prefer a few minutes between calls -- a memllm arm
    takes tens of minutes and polling faster only burns turns.
    """
    try:
        api = _api()
        ref = _qualify(slug, api)
        st = api.kernels_status(ref)
    except Exception as e:  # noqa: BLE001
        return f"status failed: {type(e).__name__}: {e}"
    status = getattr(st, "status", None) or str(st)
    msg = getattr(st, "failure_message", None) or getattr(st, "failureMessage", None)
    return f"{ref}: {status}" + (f"\nfailure: {msg}" if msg else "")


@mcp.tool()
def kaggle_logs(slug: str, tail_lines: int = 120) -> str:
    """Fetch a kernel's execution log, last `tail_lines` lines.

    This is where a failed arm explains itself -- OOM, a stale clone missing a
    flag, or Ollama falling back to CPU.
    """
    try:
        api = _api()
        ref = _qualify(slug, api)
        raw = api.kernels_logs(ref)
    except Exception as e:  # noqa: BLE001
        return f"logs failed: {type(e).__name__}: {e}"
    text = raw if isinstance(raw, str) else json.dumps(raw, indent=1, default=str)
    lines = text.splitlines()
    if len(lines) <= tail_lines:
        return text or "(empty log)"
    return f"... {len(lines) - tail_lines} earlier lines omitted ...\n" + "\n".join(
        lines[-tail_lines:]
    )


@mcp.tool()
def kaggle_fetch_results(
    slug: str, dest: str = "results", pattern: str = "", overwrite: bool = False
) -> str:
    """Download a finished kernel's output files into a local directory.

    Args:
        slug: kernel slug.
        dest: local directory, default results/. Created if absent.
        pattern: optional filename filter passed to the Kaggle API.
        overwrite: replace files that already exist locally. Default False so a
            re-fetch cannot silently clobber a result you have already analysed.

    Downloads then lists what arrived, so a partial run is obvious.
    """
    try:
        api = _api()
        ref = _qualify(slug, api)
    except RuntimeError as e:
        return str(e)

    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in out.glob("*")}

    try:
        kw: dict[str, Any] = {"kernel": ref, "path": str(out), "force": bool(overwrite)}
        if pattern.strip():
            kw["file_pattern"] = pattern.strip()
        files, _token = api.kernels_output(**kw)
    except Exception as e:  # noqa: BLE001
        return f"fetch failed: {type(e).__name__}: {e}"

    after = {p.name for p in out.glob("*")}
    new = sorted(after - before)
    report = [f"{ref} -> {out}/", f"  files reported by Kaggle: {len(files or [])}"]
    if new:
        report.append("  new locally:")
        report += [f"    {n}" for n in new]
    else:
        report.append("  no new files (already present, or run produced none)")
        if not overwrite:
            report.append("  pass overwrite=True to replace existing files")
    return "\n".join(report)


@mcp.tool()
def kaggle_list_kernels(mine: bool = True, page_size: int = 20) -> str:
    """List Kaggle kernels, yours by default, most recent first."""
    try:
        api = _api()
        items = api.kernels_list(mine=mine, page_size=page_size) or []
    except Exception as e:  # noqa: BLE001
        return f"list failed: {type(e).__name__}: {e}"
    if not items:
        return "no kernels found"
    rows = []
    for k in items:
        ref = getattr(k, "ref", "?")
        title = getattr(k, "title", "")
        last = getattr(k, "last_run_time", None) or getattr(k, "lastRunTime", "")
        rows.append(f"  {ref:<45} {str(last)[:19]:<20} {title[:40]}")
    return f"{len(items)} kernels:\n" + "\n".join(rows)


@mcp.tool()
def kaggle_wait(slug: str, timeout_seconds: int = 600, poll_seconds: int = 30) -> str:
    """Block until a kernel finishes, or until timeout.

    Capped at 30 minutes per call because MCP calls should not hang a session
    for hours; a memllm ladder outlives that, so call this repeatedly or poll
    kaggle_status instead. Returns the terminal status, or the last status seen.
    """
    timeout_seconds = max(30, min(int(timeout_seconds), 1800))
    poll_seconds = max(10, min(int(poll_seconds), 120))
    try:
        api = _api()
        ref = _qualify(slug, api)
    except RuntimeError as e:
        return str(e)

    deadline = time.time() + timeout_seconds
    last = "unknown"
    while time.time() < deadline:
        try:
            st = api.kernels_status(ref)
            last = str(getattr(st, "status", None) or st)
        except Exception as e:  # noqa: BLE001
            return f"status failed while waiting: {type(e).__name__}: {e}"
        low = last.lower()
        if any(w in low for w in ("complete", "error", "cancel", "fail")):
            msg = getattr(st, "failure_message", None) or ""
            return f"{ref}: {last}" + (f"\nfailure: {msg}" if msg else "")
        time.sleep(poll_seconds)
    return f"{ref}: still {last} after {timeout_seconds}s (not finished)"


if __name__ == "__main__":
    mcp.run(transport="stdio")
