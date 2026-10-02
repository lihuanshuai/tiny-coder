# AGENTS.md

## Project Overview

`tiny-coder` is a `src`-layout Python package that provides a small graph runtime for local LLM
coding agents. A run is a directed graph of async node functions over a mutable `dict` state, with
OpenAI-compatible tool calling, structured JSON output, and JSON checkpointing.
`src/tiny_coder/structured_agent.py` builds a ready-to-run agent on top of that runtime.

## Coding Rules

- Python code must follow `/python-code-style-guide`; stricter rules defined in this repo take
  precedence.
- Keep package-level `src/tiny_coder/__init__.py` minimal; do not add broad re-exports unless the
  public API is intentionally being changed.
- Prefer `pathlib.Path` over `os.path` for path operations.
- When joining paths relative to a module, use `Path(__file__).parent`.
- Text file reads and writes must explicitly pass `encoding="utf-8"` and `newline="\n"`.
- Use LF line endings (`\n`) for text files.
- In PowerShell, use `@'` and `'@` for here-documents.
- Prefer `apply_patch` for small manual edits.

## Validation

Run focused checks after edits when practical:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/tiny_coder tests
uv run pre-commit run --all-files
```

Use `uv sync` to install development dependencies before running the full check suite.
