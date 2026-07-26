# AGENTS.md

## Project Overview

`tiny-coder` is a `src`-layout Python package for local LLM coding agents. The core module is
`src/tiny_coder/file_agent.py`, which implements a plugin-driven `BasicFileAgent` around
LangGraph, Pydantic response validation, safe workspace file access, and structured JSON output.

## Coding Rules

- Keep package-level `src/tiny_coder/__init__.py` minimal; do not add broad re-exports unless the
  public API is intentionally being changed.
- Import runtime objects from concrete modules, for example `tiny_coder.file_agent`.
- Prefer `pathlib.Path` over `os.path` for path operations.
- When joining paths relative to a module, use `Path(__file__).parent`.
- Text file reads and writes must explicitly pass `encoding="utf-8"` and `newline="\n"`.
- Use LF line endings (`\n`) for text files.
- In PowerShell, use `@'` and `'@` for here-documents.
- Prefer `apply_patch` for small manual edits.

## Module Responsibilities

- `src/tiny_coder/file_agent.py`: agent lifecycle, plugin hooks, prompt construction, JSON schema
  validation, path safety, and file write orchestration.
- `src/tiny_coder/json_utils.py`: model-output JSON extraction, fenced JSON handling, and streaming
  JSON string-field extraction.
- `src/tiny_coder/yaml_utils.py`: stable YAML dumping for generated project content.

## Validation

Run focused checks after edits when practical:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/tiny_coder
uv run pre-commit run --all-files
```

Use `uv sync` to install development dependencies before running the full check suite.
