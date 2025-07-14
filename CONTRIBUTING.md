# Contributing

Thank you for your interest in improving OtoKotoba Composer!

## Branching scheme
- **release/v2.x** – stable release maintenance for v2
- **feature/<slug>** – new work branches
- open pull requests against **main**

After cloning the repo run:
```bash
poetry run pre-commit install
```
to set up git hooks.

## Commit messages
Use the [Conventional Commits](https://www.conventionalcommits.org/) style.
Accepted prefixes include `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, and `build`.
Example: `feat: add cool sampler`

## Development checklist
Before pushing your branch:
1. Ensure pre-commit hooks run
   ```bash
   pre-commit run --files <changed files>
   ```
1. Run the full test suite
   ```bash
   pytest -q
   ```
2. Generate demo MIDI to ensure no runtime errors
   ```bash
   make demo
   ```
