#!/usr/bin/env sh
set -eu

REPO="RooAGI/Membrick"
AGENT=""
PROJECT="."
INSTALL_DIR=""

usage() {
  cat <<'EOF'
Migrate an existing Lint-AI project integration to Membrick.

Usage:
  migrate-lint-ai.sh --agent codex|claude|gemini|agy [--project PATH] [--install-dir DIR]

This installs Membrick and updates the selected agent configuration. Existing
.lint-ai project data is moved into .membrick during installation. If both
directories exist, non-conflicting entries are merged and conflicting legacy
entries are preserved under .membrick/migration-archive. The project-root
.lint-ai directory is removed after migration. The old lint-ai executable is
left in place.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --agent)
      [ "$#" -ge 2 ] || { echo "--agent requires a value" >&2; exit 2; }
      AGENT="$2"
      shift 2
      ;;
    --project)
      [ "$#" -ge 2 ] || { echo "--project requires a path" >&2; exit 2; }
      PROJECT="$2"
      shift 2
      ;;
    --install-dir)
      [ "$#" -ge 2 ] || { echo "--install-dir requires a path" >&2; exit 2; }
      INSTALL_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$AGENT" in
  codex|claude|gemini|agy) ;;
  *) echo "choose an agent with --agent codex|claude|gemini|agy" >&2; exit 2 ;;
esac

[ -d "$PROJECT" ] || { echo "project directory does not exist: $PROJECT" >&2; exit 2; }
PROJECT="$(cd "$PROJECT" && pwd -P)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM
SCRIPT="$TMP_DIR/install.sh"
URL="https://raw.githubusercontent.com/${REPO}/main/scripts/install.sh"

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o "$SCRIPT"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "$SCRIPT" "$URL"
else
  echo "curl or wget is required" >&2
  exit 1
fi

set -- --agent "$AGENT" --project "$PROJECT"
[ -z "$INSTALL_DIR" ] || set -- "$@" --install-dir "$INSTALL_DIR"
sh "$SCRIPT" "$@"

cat <<EOF

Lint-AI migration finished for $AGENT.
The Membrick installer updated that agent's connection. Existing project data
is preserved under .membrick, and the project-root .lint-ai directory has been
removed. Any conflicting legacy state is preserved under
.membrick/migration-archive.
The old lint-ai executable was not removed; remove it after confirming Membrick
works in your projects.
EOF
