#!/usr/bin/env bash
# Run from this repository's root after gh auth login. Never force-pushes.
set -euo pipefail
expected_remote="https://github.com/HarshaSanka/semiconductor-quality-lab.git"
test -f pyproject.toml || { echo 'Run this script from the project root.'; exit 1; }
gh auth status
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi
if git remote get-url origin >/dev/null 2>&1; then
  current_remote="$(git remote get-url origin)"
  test "$current_remote" = "$expected_remote" || { echo 'Origin points elsewhere; inspect it first.'; exit 1; }
else
  git remote add origin "$expected_remote"
fi
git add .
if ! git diff --cached --quiet; then
  git commit -m 'Add SECOM research platform and measured core benchmark'
fi
git push -u origin HEAD:main
echo 'Published: https://github.com/HarshaSanka/semiconductor-quality-lab'
