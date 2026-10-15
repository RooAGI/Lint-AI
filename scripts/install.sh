#!/usr/bin/env sh
set -eu

REPO="RooAGI/Membrick"
BASE_URL="https://github.com/${REPO}/releases/latest/download"
INSTALL_DIR="${MEMBRICK_INSTALL_DIR:-${HOME}/.local/bin}"
AGENT=""
PROJECT="."

usage() {
  cat <<'EOF'
Install the latest official Membrick release and optionally configure an agent.

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
    ASSET="membrick-linux-x86_64"
    ;;
  Darwin:x86_64|Darwin:amd64)
    ASSET="membrick-macos-x86_64"
    ;;
  Darwin:arm64|Darwin:aarch64)
    ASSET="membrick-macos-aarch64"
    ;;
  *)
    echo "No prebuilt Membrick release for ${OS} ${ARCH}." >&2
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
SUM_TMP="${TMP_DIR}/SHA256SUMS"

echo "Downloading official Membrick release for ${OS} ${ARCH}..."
fetch "${BASE_URL}/${ASSET}" "$BIN_TMP"
if fetch "${BASE_URL}/SHA256SUMS" "$SUM_TMP"; then
  EXPECTED="$(awk -v asset="$ASSET" '$2 == asset {print $1}' "$SUM_TMP")"
else
  # Support releases published before the aggregate manifest was introduced.
  fetch "${BASE_URL}/${ASSET}.sha256" "$SUM_TMP"
  EXPECTED="$(awk '{print $1}' "$SUM_TMP")"
fi
if [ -z "$EXPECTED" ]; then
  echo "No checksum found for ${ASSET}; refusing to install." >&2
  exit 1
fi
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
cp "$BIN_TMP" "${INSTALL_DIR}/membrick"
BIN="${INSTALL_DIR}/membrick"
touch "${INSTALL_DIR}/.membrick-managed"
printf '%s\n' standalone > "${INSTALL_DIR}/membrick-install-method"

echo "Installed $($BIN --version) to ${BIN}"

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    echo "Note: ${INSTALL_DIR} is not currently on PATH."
    echo "Add it to your shell profile, for example: export PATH=\"${INSTALL_DIR}:\$PATH\""
    ;;
esac

if [ -n "$AGENT" ]; then
  echo "Configuring Membrick for ${AGENT} in ${PROJECT}..."
  "$BIN" install "$AGENT" "$PROJECT"
  echo "Verifying the Membrick MCP runtime..."
  "$BIN" verify "$AGENT" "$PROJECT"
  echo "Membrick is installed, configured for ${AGENT}, and MCP verification passed."
else
  echo "Membrick is installed. To configure an agent, rerun with --agent codex|claude|gemini|agy."
fi
