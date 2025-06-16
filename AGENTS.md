# Contributor Guide

## Basics
- The code is developed with the bun runtime

## Code style guidelines
- Check the eslint rules to understand style guidelines
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
