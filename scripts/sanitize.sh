#!/usr/bin/env bash
# sanitize.sh — scan for internal/sensitive patterns before open-source release
#
# AI-Generated Code - Claude Opus 4.8 (Anthropic)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Each entry: "description|regex"
PATTERNS=(
  "Internal registry URL|registry[-.].*\.scitix\.ai"
  "Internal API endpoint|console\.scitix\.ai"
  "Private IPv4 (10.x)|\\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\\b"
  "Private IPv4 (172.16-31)|\\b172\.(1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}\\b"
  "Private IPv4 (192.168)|\\b192\.168\.\d{1,3}\.\d{1,3}\\b"
  "Hardcoded home path|/home/[a-z][a-z0-9_-]+/"
  "Hardcoded macOS path|/Users/[A-Za-z][A-Za-z0-9_-]+/"
  "Hardcoded credential|(?i)(password|secret|token)\s*=\s*['\"][^'\"]{8,}"
)

# Lines matching ANY of these are silently ignored.
ALLOWLIST=(
  "pyproject.toml:.*@scitix.ai"     # author email
  "scripts/sanitize.sh"             # this script itself (contains the patterns)
  "docs/"                           # local design/planning docs
  "tests/"                          # test fixtures use example IPs/paths
  "CLAUDE.md"                       # project guidelines
  "\.git/"                          # git internals
  "127\.0\.0\.1"                    # localhost is fine
  "0\.0\.0\.0"                      # bind-all is fine
  "\.venv/"                         # third-party packages
  "data/"                           # benchmark datasets (may contain example IPs/paths in data)
  '"10\.0\.1\.100:30001"'           # ServiceBinding.address field-docstring example
)

ALLOWLIST_PATTERN="$(IFS='|'; echo "${ALLOWLIST[*]}")"
FAIL=0

for entry in "${PATTERNS[@]}"; do
  description="${entry%%|*}"
  pattern="${entry#*|}"
  hits=$(grep -rnPI "$pattern" \
    --include='*.py' --include='*.yaml' --include='*.yml' \
    --include='*.toml' --include='*.json' --include='*.md' \
    --include='*.sh' --include='*.cfg' --include='*.ini' \
    . 2>/dev/null | grep -Ev "$ALLOWLIST_PATTERN" || true)
  if [[ -n "$hits" ]]; then
    echo "❌ $description ($pattern):"
    echo "$hits" | head -20
    echo ""
    FAIL=1
  fi
done

if [[ "$FAIL" -eq 1 ]]; then
  echo "========================================="
  echo "Sanitization FAILED — see matches above."
  echo "False positive? Add it to ALLOWLIST in scripts/sanitize.sh"
  echo "========================================="
  exit 1
else
  echo "✅ Sanitization passed — no sensitive patterns found."
fi
