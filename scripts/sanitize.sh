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
  "Hardcoded shared-mount path|/(mnt|volume|scratch|nfs)/[A-Za-z0-9._-]+/"
  "Hardcoded credential|(?i)(password|secret|token)\s*=\s*['\"][^'\"]{8,}"
)

# Config files must NEVER hardcode an absolute filesystem path — use a
# ${ENV_VAR}, a /path/to/... placeholder, or a HuggingFace repo id instead.
# Scanned only over config formats (CONFIG_INCLUDES). This is the inverse of an
# allowlist: flag any absolute path whose first segment is not a safe system
# root, so it catches every mount root (/mnt, /volume, /data4, ...) without
# enumerating them — unlike the line above, which is a best-effort blocklist for
# code/docs where illustrative example paths are legitimate.
CONFIG_PATTERNS=(
  'Hardcoded absolute path in config|(?<![\w./:$}])/(?!(?:usr|bin|sbin|etc|tmp|var|proc|sys|dev|opt|run|srv|lib|lib64|root|path)/)[A-Za-z][A-Za-z0-9._-]*/[A-Za-z0-9._-]+'
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

# Pathspecs for git grep. We scan only tracked files, so local artifacts
# (.venv/, .mypy_cache/, outputs/, __pycache__/) are never matched — only what
# would actually ship. New cache dirs need no allowlist upkeep.
ALL_PATHS=(
  '*.py' '*.yaml' '*.yml' '*.toml' '*.json' '*.md' '*.sh' '*.cfg' '*.ini'
)
# Config formats only.
CONFIG_PATHS=(
  '*.yaml' '*.yml' '*.toml' '*.json' '*.cfg' '*.ini'
)

# scan DESCRIPTION REGEX PATHSPEC... — report any non-allowlisted hits, set FAIL.
scan() {
  local description="$1" pattern="$2"
  shift 2
  local hits
  hits=$(git grep -nIP -e "$pattern" -- "$@" 2>/dev/null | grep -Ev "$ALLOWLIST_PATTERN" || true)
  [[ -z "$hits" ]] && return 0
  local lines
  mapfile -t -n 20 lines <<<"$hits"
  echo "❌ $description ($pattern):"
  printf '%s\n' "${lines[@]}"
  echo ""
  FAIL=1
}

for entry in "${PATTERNS[@]}"; do
  scan "${entry%%|*}" "${entry#*|}" "${ALL_PATHS[@]}"
done

for entry in "${CONFIG_PATTERNS[@]}"; do
  scan "${entry%%|*}" "${entry#*|}" "${CONFIG_PATHS[@]}"
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
