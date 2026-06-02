#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export GRADLE_USER_HOME="${GRADLE_USER_HOME:-$ROOT_DIR/.chatgpt/gradle-home}"

if ! find "$GRADLE_USER_HOME/wrapper/dists" -name "gradle-*.zip.ok" -print -quit 2>/dev/null | grep -q .; then
    echo "Prepared Gradle cache not found at $GRADLE_USER_HOME." >&2
    echo "Use the GPT-compatible release ZIP or build it with the GitHub workflow." >&2
    exit 1
fi

GRADLE_ARGS=(--no-daemon --offline clean build)

if command -v xvfb-run >/dev/null 2>&1; then
    exec xvfb-run -a bash "$ROOT_DIR/gradlew" "${GRADLE_ARGS[@]}"
fi

exec bash "$ROOT_DIR/gradlew" "${GRADLE_ARGS[@]}"
