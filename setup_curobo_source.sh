#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/yezhengfu/robot_motion_control"
ARCHIVE="/tmp/curobo-v0.7.8.tar.gz"
TARGET="$PROJECT_DIR/third_party/curobo"
URL="https://codeload.github.com/NVlabs/curobo/tar.gz/refs/tags/v0.7.8"

cd "$PROJECT_DIR"

if [[ -e "$TARGET" ]]; then
  echo "ERROR: $TARGET already exists; nothing was changed."
  exit 1
fi

mkdir -p "$PROJECT_DIR/third_party"

echo "Downloading cuRobo v0.7.8 from GitHub..."
curl -fL --http1.1 \
  --retry 10 \
  --retry-delay 3 \
  --retry-all-errors \
  --connect-timeout 20 \
  -o "$ARCHIVE" \
  "$URL"

echo "Validating archive..."
tar -tzf "$ARCHIVE" >/dev/null
FIRST_ENTRY="$(tar -tzf "$ARCHIVE" | sed -n '1p')"
if [[ "$FIRST_ENTRY" != curobo-0.7.8/* ]]; then
  echo "ERROR: unexpected archive layout: $FIRST_ENTRY"
  exit 1
fi

mkdir "$TARGET"
tar -xzf "$ARCHIVE" -C "$TARGET" --strip-components=1

echo "cuRobo source is ready:"
echo "  $TARGET"
echo "Top-level files:"
find "$TARGET" -maxdepth 1 -mindepth 1 -printf '  %f\n' | sort | sed -n '1,20p'
