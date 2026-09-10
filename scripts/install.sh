#!/usr/bin/env sh
set -eu

REPO="RooAGI/Lint-AI"
BASE_URL="https://github.com/${REPO}/releases/latest/download"
INSTALL_DIR="${LINT_AI_INSTALL_DIR:-${HOME}/.local/bin}"
AGENT=""
PROJECT="."

usage() {
  cat <<'EOF'
Install the latest official Lint-AI release and optionally configure an agent.

Usage:
  install.sh [--agent codex|claude|gemini|agy] [--project PATH] [--install-dir DIR]

Examples:
  install.sh
  install.sh --agent codex --project .
  install.sh --agent claude --project /path/to/project
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
      [ "$#" -ge 2 ] || { echo "--project requires a value" >&2; exit 2; }
      PROJECT="$2"
      shift 2
      ;;
    --install-dir)
      [ "$#" -ge 2 ] || { echo "--install-dir requires a value" >&2; exit 2; }
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
  ""|codex|claude|gemini|agy) ;;
  *) echo "unsupported agent: $AGENT" >&2; exit 2 ;;
esac

OS="$(uname -s)"
ARCH="$(uname -m)"
case "${OS}:${ARCH}" in
  Linux:x86_64|Linux:amd64)
    ASSET="lint-ai-linux-x86_64"
    ;;
  Darwin:x86_64|Darwin:amd64)
    ASSET="lint-ai-macos-x86_64"
    ;;
  Darwin:arm64|Darwin:aarch64)
    ASSET="lint-ai-macos-aarch64"
    ;;
  *)
    echo "No prebuilt Lint-AI release for ${OS} ${ARCH}." >&2
    echo "Fallback: cargo install --git https://github.com/${REPO} --features agent-integrations" >&2
    exit 1
    ;;
esac

if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -qO "$2" "$1"; }
else
  echo "curl or wget is required" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

BIN_TMP="${TMP_DIR}/${ASSET}"
SUM_TMP="${TMP_DIR}/${ASSET}.sha256"

echo "Downloading official Lint-AI release for ${OS} ${ARCH}..."
fetch "${BASE_URL}/${ASSET}" "$BIN_TMP"
fetch "${BASE_URL}/${ASSET}.sha256" "$SUM_TMP"

EXPECTED="$(awk '{print $1}' "$SUM_TMP")"
if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$BIN_TMP" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL="$(shasum -a 256 "$BIN_TMP" | awk '{print $1}')"
else
  echo "sha256sum or shasum is required to verify the release" >&2
  exit 1
fi

if [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "Checksum verification failed; refusing to install." >&2
  exit 1
fi

echo "Checksum verified."
mkdir -p "$INSTALL_DIR"
chmod 0755 "$BIN_TMP"
cp "$BIN_TMP" "${INSTALL_DIR}/lint-ai"
BIN="${INSTALL_DIR}/lint-ai"

echo "Installed $($BIN --version) to ${BIN}"

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    echo "Note: ${INSTALL_DIR} is not currently on PATH."
    echo "Add it to your shell profile, for example: export PATH=\"${INSTALL_DIR}:\$PATH\""
    ;;
esac

if [ -n "$AGENT" ]; then
  case "$AGENT" in
    codex)
      INSTALL_FLAG="--codex-install"
      VERIFY_FLAG="--codex-verify-mcp"
      ;;
    claude)
      INSTALL_FLAG="--claude-code-install"
      VERIFY_FLAG="--claude-code-verify-mcp"
      ;;
    gemini)
      INSTALL_FLAG="--gemini-cli-install"
      VERIFY_FLAG="--gemini-cli-verify-mcp"
      ;;
    agy)
      INSTALL_FLAG="--agy-install"
      VERIFY_FLAG="--agy-verify-mcp"
      ;;
  esac

  echo "Configuring Lint-AI for ${AGENT} in ${PROJECT}..."
  "$BIN" "$INSTALL_FLAG" "$PROJECT"
  echo "Verifying the Lint-AI MCP runtime..."
  "$BIN" "$VERIFY_FLAG" "$PROJECT"
  echo "Lint-AI is installed, configured for ${AGENT}, and MCP verification passed."
else
  echo "Lint-AI is installed. To configure an agent, rerun with --agent codex|claude|gemini|agy."
fi
