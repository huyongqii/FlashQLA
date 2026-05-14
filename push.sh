#!/usr/bin/env bash
set -e

# 默认提交信息
MSG=${1:-"support blackwell"}

# 当前分支
BRANCH=$(git branch --show-current)

echo "Current branch: ${BRANCH}"
echo "Commit message: ${MSG}"

git add .
git commit -m "${MSG}" || {
  echo "No changes to commit."
}

git push origin "${BRANCH}"

git status
