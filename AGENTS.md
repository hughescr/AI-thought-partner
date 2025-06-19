# Contributor Guide

## Basics
- The typescript code is developed with the bun runtime
- The python code is developed using a `uv` environment; use `uv sync` to build the environment, `uv add` and `uv remove`, `uv run`, etc. for package management and running.

## Code style guidelines (use the correct guidelines per language)

### For typescript
- TypeScript with strict type checking.
- ES modules (`import`/`export`) with `"type": "module"` in `package.json`.
- ESLint rules come from `@hughescr/eslint-config-default`.
- camelCase naming for variables/functions and PascalCase for classes.
- Unused variables prefixed with `_`.
- Ignore errors with empty `catch` blocks if desired.
- Tests use the Bun test framework (`import { describe } from 'bun:test'`).
- Prefer Lodash utilities and `async/await` over raw Promises.
- Document classes expose `metadata` and `pageContent`.
- Verify style compliance with `bun lint`

### For python
- Use `uvx ruff check --fix .` for linting
- Use `uvx pytest .` for testing
- Use the `uv` environment, so `uv run...`, `uv add...` etc not just bare python

## Testing Instructions
- Find the CI plan in the .github/workflows folder if there is one.
- Run linting tools and type-checking tools
- The commit should pass all tests before you merge.
- Fix any test or type errors until the whole suite is green.
- After moving files or changing imports, run linting and type-checking again.
- Add or update tests for the code you change, even if nobody asked.

## PR instructions
Title format: [Codex] <Title>
