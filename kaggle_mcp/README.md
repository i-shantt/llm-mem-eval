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
| `kaggle_auth_status` | check auth, print setup instructions if missing |
| `kaggle_list_cells` | list the `## Cell ...` headings available to push |
| `kaggle_push_notebook` | assemble selected cells into a notebook and push |
| `kaggle_status` | queued / running / complete / error |
| `kaggle_wait` | block until terminal, capped at 30 min per call |
| `kaggle_logs` | tail the execution log — where a failed arm explains itself |
| `kaggle_fetch_results` | download outputs into `results/` |
| `kaggle_list_kernels` | list your kernels, most recent first |

## Notes

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
