#!/usr/bin/env bash
# Git pre-commit hook: format staged Python files with autopep8 and run flake8
# Exits non-zero if flake8 finds violations to prevent commit.

set -euo pipefail

# Find staged python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.py$' || true)
if [ -z "$STAGED_FILES" ]; then
  # nothing to do
  exit 0
fi

# Filter: nur existierende Dateien für flake8
EXISTING_FILES=""
for file in $STAGED_FILES; do
  [ -f "$file" ] && EXISTING_FILES="$EXISTING_FILES $file"
done
if [ -z "$EXISTING_FILES" ]; then
  exit 0
fi

# Run autopep8 in-place on each staged file and re-add if changed
echo "Running autopep8 on staged Python files..."
changed=false
for file in $STAGED_FILES; do
  if [ -f "$file" ]; then
    autopep8 --in-place --aggressive --aggressive "$file" || true
    # If file changed, add it to the index again
    if ! git diff --quiet -- "$file"; then
      git add "$file"
      changed=true
    fi
  fi
done

# Run flake8 on die existierenden python files
echo "Running flake8 checks..."
flake8 $EXISTING_FILES || {
  echo "\nflake8 reported issues. Commit aborted. Please fix them before committing." >&2
  exit 1
}

# Optionally run tests or other linters here
# If autopep8 modified files we already re-added them to the index
if [ "$changed" = true ]; then
  echo "Files were formatted and re-added to commit."
fi

exit 0
