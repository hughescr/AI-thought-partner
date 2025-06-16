# Contributor Guide

## Basics
- The code is developed with the bun runtime

## Code style guidelines
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

## Testing Instructions
- Find the CI plan in the .github/workflows folder if there is one.
- Run `bun lint` to check for any lint errors
- From the package root you can just call `bun test`. The commit should pass all tests before you merge.
- Fix any test or type errors until the whole suite is green.
- After moving files or changing imports, run `bun lint` to be sure ESLint and TypeScript rules still pass.
- Add or update tests for the code you change, even if nobody asked.

## PR instructions
Title format: [Codex] <Title>
