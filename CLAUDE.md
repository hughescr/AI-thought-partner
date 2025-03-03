# AI-thought-partner Development Guide

## Build & Test Commands
- Run all tests: `bun test`
- Run single test: `bun test tests/FileName.test.ts`
- Type checking: `tsc`
- Linting: `bun run lint` or `eslint --fix`
- Full test suite with linting: `bun run test`
- Generate coverage: `bun run coverage`
- Check packages: `bun run package-check`

## Code Style Guidelines
- TypeScript with strict type checking
- ES modules (import/export) with "type": "module" in package.json
- ESLint rules from @hughescr/eslint-config-default
- Naming: camelCase for variables/functions, PascalCase for classes
- Unused variables prefixed with _ (configured in ESLint)
- Error handling: try/catch blocks with empty catches for ignored errors
- Tests use Bun test framework (import from bun:test)
- Lodash utilities for functional programming patterns
- Async/await for asynchronous code (not raw Promises)
- Document classes expose metadata and pageContent properties