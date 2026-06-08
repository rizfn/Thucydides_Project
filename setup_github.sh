#!/usr/bin/env bash
#
# Push this project to a new GitHub repository.
#
# Prerequisites:
#   - You have a GitHub account.
#   - You have either (a) the GitHub CLI `gh` installed
#     (https://cli.github.com), or (b) you create the empty repo
#     manually on github.com first.
#
# Usage:
#   1) Edit GITHUB_USER below to your GitHub username.
#   2) From the project folder run:  bash setup_github.sh
#
set -euo pipefail

# --------- EDIT THIS LINE --------------------------------------------------
GITHUB_USER="mathiasheltberg@hotmail.com"
REPO_NAME="Thucydides_Project"
VISIBILITY="public"     # or "private"
# ---------------------------------------------------------------------------

cd "$(dirname "$0")"

# Clean up any half-initialised git state from prior runs
if [[ -d .git ]]; then
  echo "Removing existing .git directory ..."
  rm -rf .git
fi

# Fresh init
git init -b main
git config user.name  "Mathias Heltberg"
git config user.email "mathiasheltberg@hotmail.com"
git add -A
git commit -m "Initial commit: model, applet, paper, evidence note, ancient replication"

# Try GitHub CLI; fall back to manual instructions
if command -v gh >/dev/null 2>&1; then
  echo
  echo "Creating GitHub repo via gh CLI ..."
  gh repo create "$GITHUB_USER/$REPO_NAME" --"$VISIBILITY" \
     --source=. --remote=origin --push
  echo
  echo "==========================================================="
  echo "Pushed to https://github.com/$GITHUB_USER/$REPO_NAME"
  echo
  echo "Next steps to enable the applet on GitHub Pages:"
  echo "  Open https://github.com/$GITHUB_USER/$REPO_NAME/settings/pages"
  echo "  Source: 'Deploy from a branch'  Branch: 'main'  Folder: '/ (root)'"
  echo "  Save. Wait ~30 s. The applet will be live at:"
  echo "    https://$GITHUB_USER.github.io/$REPO_NAME/applet.html"
  echo "==========================================================="
else
  echo
  echo "==========================================================="
  echo "GitHub CLI 'gh' not found. Do this manually:"
  echo
  echo "1) Go to https://github.com/new"
  echo "   - repo name: $REPO_NAME"
  echo "   - $VISIBILITY"
  echo "   - DO NOT add README / .gitignore / license (already in repo)"
  echo "2) Then run:"
  echo "   git remote add origin git@github.com:$GITHUB_USER/$REPO_NAME.git"
  echo "   git push -u origin main"
  echo
  echo "3) Enable Pages at:"
  echo "   https://github.com/$GITHUB_USER/$REPO_NAME/settings/pages"
  echo "   Source: 'Deploy from a branch'  Branch: 'main'  Folder: '/ (root)'"
  echo "==========================================================="
fi
