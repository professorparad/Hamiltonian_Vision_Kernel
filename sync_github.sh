#!/usr/bin/env bash
set -euo pipefail

message="${1:-auto sync}"
remote="${GIT_REMOTE:-origin}"
branch="${GIT_BRANCH:-$(git branch --show-current)}"

if [[ -z "$branch" ]]; then
  echo "Could not determine current git branch."
  exit 1
fi

git add .

if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "$message"
fi

git push "$remote" "$branch"
