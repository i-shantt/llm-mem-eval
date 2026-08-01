# Kaggle MCP server

Runs memllm evaluation arms on Kaggle's free T4s without a human in the loop:
build a notebook from `kaggle/PASTE_THESE_CELLS.md`, push it, poll it, pull the
result JSON back into `results/`.

## Setup

```bash
pip install -r kaggle_mcp/requirements.txt

# Authenticate. OAuth is easiest and leaves no token to manage:
kaggle auth login

# Register the server with Claude Code:
claude mcp add kaggle -- $PWD/.venv/bin/python $PWD/kaggle_mcp/server.py
```

Do not paste a Kaggle token into a chat message. Anything sent as text is
written to the session transcript; `kaggle auth login`, `$KAGGLE_API_TOKEN`, or
`~/.kaggle/access_token` all keep it out. The server never accepts, logs, or
echoes a credential — `kaggle_auth_status` reports only the resolved username.

## Tools

| tool | does |
|---|---|
| `kaggle_selftest` | check auth, deps, paths and write access in one call |
| `kaggle_auth_status` | check auth, print setup instructions if missing |
| `kaggle_list_cells` | list the `## Cell ...` headings available to push |
| `kaggle_push_notebook` | assemble selected cells into a notebook and push |
| `kaggle_status` | queued / running / complete / error |
| `kaggle_wait` | block until terminal, capped at 30 min per call |
| `kaggle_logs` | tail the execution log — **only after the run ends**, see below |
| `kaggle_fetch_results` | download outputs into `results/` |
| `kaggle_list_kernels` | list your kernels, most recent first |

## Troubleshooting

**The server does not appear in the MCP list.** Scope and restart, in that
order. `claude mcp add` without `--scope user` registers the server for the
*current directory's* project only, so a session started anywhere else will not
see it. Check where it landed:

```bash
python3 -c "import json;d=json.load(open('$HOME/.claude.json'));\
print('user:',list(d.get('mcpServers',{})));\
print({p:list(v.get('mcpServers',{})) for p,v in d.get('projects',{}).items() if v.get('mcpServers')})"
```

It must appear under `user:`. Then **restart Claude Code** — servers registered
mid-session do not load into that session, only into later ones. A restart that
happened *before* the registration does not count.

**A fetch that tries to download gigabytes is pulling the Ollama models.**
Everything under `/kaggle/working` is kernel output, and the notebook puts the
model store there because `/root` is too small for multi-GB pulls. So the models
ship with the results, and a plain fetch can die on a 4.7 GB `IncompleteRead`.
Delete the model directory in the last cell, and pass `pattern="cond_"` to fetch
only what you want. **`pattern` is a prefix match, not a glob** — `"cond_"`
works, `"cond_*.json"` matches nothing and quietly returns only the log.

**`kaggle_logs` returns "(empty log)" while the kernel is running.** This is
Kaggle's behaviour, not a bug here: the log is published when the run reaches a
terminal state, so there is no partial output to follow and no way to watch
progress mid-run. Poll `kaggle_status` until it is terminal, *then* read the
log. Building a watcher that greps a running kernel's log for a progress marker
does not work — the marker cannot appear until the run is already over.

Because of that, make a run self-reporting: put pre-flight assertions early so a
bad clone or a missing flag kills the kernel in minutes, and the log you finally
read explains itself.

**Anything else:** call `kaggle_selftest`. It reports auth, dependency imports,
the resolved repo root, cell parsing, and write access in one call.

## Notes

- **This server runs from `.venv`.** The registered command points at
  `.venv/bin/python`, so rebuilding that venv without reinstalling
  `kaggle_mcp/requirements.txt` breaks the server at startup — which surfaces
  only as "failed to connect". `kaggle_selftest` names the cause.
- **Relative paths resolve against the repo, not the cwd.** Claude Code spawns
  the server with an arbitrary working directory, so `dest="results"` always
  means `<repo>/results`, never `$PWD/results`.

- **Kernels push private by default.** `is_private=False` publishes the code and
  its output under your account.
- **Pushing an existing slug replaces it** and starts a new run.
- **GPU quota is ~30h/week**, consumed by every run. Prefer one notebook with
  several arms over many small pushes. Pass `enable_gpu=False` for CPU-only
  work such as the retrieval sweep.
- `include=` selects cells by heading substring, so a 45-minute control run does
  not require re-pushing the 3-hour ladder. Cells are always emitted in document
  order regardless of the order requested, because later cells use names bound
  by earlier ones.
- Cell 4 is resumable: an arm with an existing results file is skipped, so a
  re-push after a timeout continues rather than restarting.
