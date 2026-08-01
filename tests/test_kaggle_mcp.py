"""Tests for the Kaggle MCP server's offline logic.

Nothing here touches the network or needs credentials. The parts worth testing
are notebook assembly and cell selection: a malformed notebook or a silently
dropped cell wastes a GPU run and takes an hour to discover.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "kaggle_mcp"))

server = pytest.importorskip("server", reason="kaggle_mcp deps not installed")

MD = """
Preamble prose that is not a cell.

## Cell 1 — setup

Some explanation.

```python
x = 1
```

## Cell 2 — two fences

```python
a = 1
```

more prose

```python
b = 2
```

## Cell 3 — no code, only prose

Nothing executable here.
"""


def test_parses_cells_by_heading():
    cells = server._parse_cells(MD)
    assert [c["name"] for c in cells] == [
        "Cell 1 — setup", "Cell 2 — two fences"
    ]


def test_cell_with_no_python_is_dropped():
    assert all("Cell 3" not in c["name"] for c in server._parse_cells(MD))


def test_multiple_fences_in_one_cell_are_concatenated_in_order():
    code = server._parse_cells(MD)[1]["code"]
    assert code.index("a = 1") < code.index("b = 2")


def test_prose_outside_any_cell_is_not_executable():
    joined = "\n".join(c["code"] for c in server._parse_cells(MD))
    assert "Preamble" not in joined and "Nothing executable" not in joined


def test_notebook_is_valid_nbformat():
    nbformat = pytest.importorskip("nbformat")
    nb = server._notebook(["print(1)", "print(2)"])
    nbformat.validate(nbformat.reads(json.dumps(nb), as_version=4))


def test_notebook_cells_have_unique_ids():
    """Missing ids warn today and hard-error in future nbformat versions."""
    nb = server._notebook(["a", "b", "c"])
    ids = [c["id"] for c in nb["cells"]]
    assert len(set(ids)) == 3


def test_notebook_preserves_source_order_and_content():
    nb = server._notebook(["first = 1", "second = 2"])
    assert "".join(nb["cells"][0]["source"]) == "first = 1"
    assert "".join(nb["cells"][1]["source"]) == "second = 2"


def test_real_cells_document_parses():
    """The shipped notebook doc must stay machine-readable; a heading style
    change would otherwise silently push an empty notebook."""
    cells = server._parse_cells(server.DEFAULT_CELLS.read_text())
    assert len(cells) >= 5
    names = " ".join(c["name"] for c in cells)
    assert "Cell 4b" in names, "control-arm cell missing from the notebook doc"
    assert "Cell 4c" in names, "ablation cell missing from the notebook doc"


def test_auth_failure_is_reported_not_raised(monkeypatch):
    """Tools must return the setup instructions as text. Raising instead would
    surface as an opaque 'Error executing tool' with no way to fix it."""
    monkeypatch.setattr(
        server, "_api",
        lambda: (_ for _ in ()).throw(RuntimeError(server._AUTH_HELP)),
    )
    out = server.kaggle_auth_status()
    assert "kaggle auth login" in out
    assert not out.startswith("Error")


def test_auth_help_never_asks_for_a_token_in_chat():
    assert "not paste" in server._AUTH_HELP.lower()


def test_slugify_reproduces_kaggle_title_derivation():
    """Observed against a real push: Kaggle named the kernel from the title and
    only warned that it disagreed with the requested id, so polling the
    requested slug failed with a permission error resembling broken auth."""
    assert (server._slugify("memllm: memory-lift control arms")
            == "memllm-memory-lift-control-arms")


def test_slugify_is_idempotent_on_a_clean_slug():
    for s in ("memllm-controls", "abc", "a1-b2"):
        assert server._slugify(s) == s


def test_slugify_collapses_runs_and_strips_edges():
    assert server._slugify("  A  B__C!!  ") == "a-b-c"


def test_slugify_is_bounded():
    assert len(server._slugify("word " * 60)) <= 50


@pytest.mark.parametrize("raw,expected", [
    ("/code/ishantchintapatla/memllm-controls", "ishantchintapatla/memllm-controls"),
    ("/ishantchintapatla/memllm-controls", "ishantchintapatla/memllm-controls"),
    ("ishantchintapatla/memllm-controls", "ishantchintapatla/memllm-controls"),
])
def test_push_ref_normalises_to_owner_slug(raw, expected):
    """Kaggle returns a URL path, not the bare owner/slug the other endpoints
    take. Observed as '/code/owner/slug'; a naive lstrip('/') left 'code/'
    behind and produced a 404 URL."""
    assert "/".join([s for s in raw.split("/") if s][-2:]) == expected


class _DeadTokenApi:
    """Constructs fine, reads username fine, 401s on any real request --
    the state a rejected credential leaves the client in, whether the
    rejection is transient or permanent."""

    username = "someone"

    def kernels_list(self, **kw):
        raise RuntimeError("401 Client Error: Unauthorized for url: ...")


class _LiveApi:
    username = "someone"

    def kernels_list(self, **kw):
        return []


def test_verify_auth_catches_a_dead_token():
    problem = server._verify_auth(_DeadTokenApi())
    assert problem, "a 401 must not be reported as healthy"
    assert "kaggle auth login" in problem, "must say how to fix it"
    assert "transient" in problem


def test_verify_auth_passes_on_a_live_credential():
    assert server._verify_auth(_LiveApi()) == ""


def test_selftest_fails_when_the_token_is_dead(monkeypatch):
    """The regression this exists for: selftest previously reported
    'kaggle auth ok' while every real call returned 401, because it only
    constructed the client and read an attribute."""
    monkeypatch.setattr(server, "_api", lambda: _DeadTokenApi())
    out = server.kaggle_selftest()
    assert out.startswith("PROBLEMS FOUND")
    assert "401" in out


def test_selftest_passes_when_the_token_is_live(monkeypatch):
    monkeypatch.setattr(server, "_api", lambda: _LiveApi())
    out = server.kaggle_selftest()
    assert "live request verified" in out


def test_auth_status_reports_a_dead_token_as_unusable(monkeypatch):
    monkeypatch.setattr(server, "_api", lambda: _DeadTokenApi())
    out = server.kaggle_auth_status()
    assert "NOT usable" in out


def test_status_distinguishes_rejected_auth_from_a_wrong_slug(monkeypatch):
    """Kaggle returns 'permission denied' for a rejected credential and blames
    the slug, which sends the caller looking for a typo that isn't there."""
    class _Api(_DeadTokenApi):
        def kernels_status(self, ref):
            raise ValueError("Permission 'kernels.get' was denied. "
                             "The most likely cause is a wrong kernel slug.")
    monkeypatch.setattr(server, "_api", lambda: _Api())
    out = server.kaggle_status("whatever")
    assert "CHECKED:" in out
    assert "kaggle auth login" in out


def test_prepend_code_lands_after_cell_zero(monkeypatch):
    """Cell 0 assigns the config names, so an override placed before it would
    simply be overwritten and the run would silently use the defaults."""
    cells = [{"name": "Cell 0 — run configuration", "code": "SIZES = ['a']"},
             {"name": "Cell 1 — setup", "code": "use(SIZES)"}]
    after0 = next((i + 1 for i, c in enumerate(cells)
                   if c["name"].lower().startswith("cell 0")), 0)
    assert after0 == 1
