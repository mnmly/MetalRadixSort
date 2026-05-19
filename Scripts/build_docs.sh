#!/usr/bin/env bash
# Build static DocC site(s) for one or more SPM targets into ./docs
# (GitHub Pages-ready). Modeled after ml-explore/mlx-swift's
# tools/build-documentation.sh, with optional LLM-Markdown export.
#
# Usage:
#   Scripts/build_docs.sh                # build all $TARGETS into ./docs
#   Scripts/build_docs.sh preview        # local preview (first target only)
#   Scripts/build_docs.sh -f             # bypass gh-pages branch guard
#
# Required-ish env (edit defaults below or pass at call site):
#   TARGETS              Space-separated target names. Single-target packages
#                        can just set one. Default: MetalRadixSort.
#   HOSTING_BASE_PATH    Repo name on GitHub Pages (e.g. "my-lib"). Each
#                        target is hosted at <BASE>/<Target>/ so source URLs
#                        and asset paths resolve correctly.
#   REPO_URL             https URL to the GitHub repo (no trailing slash).
#                        Enables "View on GitHub" source links per symbol.
#   REPO_BRANCH          Branch the source links point at. Default: main.
#
# Optional env:
#   OUTPUT_DIR           Default: docs
#   REQUIRE_GH_PAGES=1   Refuse to build off the gh-pages branch unless -f.
#   EMIT_MARKDOWN=1      Pass --enable-experimental-markdown-output
#                        (per-symbol .md files under <out>/<target>/data/).
#   EMIT_LLMS_TXT=1      Above + concatenate into <OUTPUT_DIR>/llms.txt.
set -euo pipefail

cd "$(dirname "$0")/.."

TARGETS="${TARGETS:-MetalRadixSort}"
HOSTING_BASE_PATH="${HOSTING_BASE_PATH:-MetalRadixSort}"
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-main}"
OUTPUT_DIR="${OUTPUT_DIR:-docs}"

FORCE=0
MODE="build"
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=1 ;;
        preview)    MODE="preview" ;;
    esac
done

if [[ "${REQUIRE_GH_PAGES:-0}" == "1" && "$MODE" == "build" && $FORCE -eq 0 ]]; then
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo)"
    if [[ "$branch" != "gh-pages" ]]; then
        echo "Refusing to build off branch '$branch'. Use -f to override."
        exit 1
    fi
fi

export DOCC_JSON_PRETTYPRINT=YES

# Preview: first target only — `swift package preview-documentation` is
# single-target and interactive.
if [[ "$MODE" == "preview" ]]; then
    first_target="${TARGETS%% *}"
    exec swift package --disable-sandbox \
        preview-documentation --target "$first_target"
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

EXTRA_FLAGS=()
if [[ "${EMIT_MARKDOWN:-0}" == "1" || "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--enable-experimental-markdown-output)
fi

SOURCE_FLAGS=()
if [[ -n "$REPO_URL" ]]; then
    SOURCE_FLAGS+=(
        --source-service github
        --source-service-base-url "${REPO_URL%/}/blob/${REPO_BRANCH}"
        --checkout-path "$(pwd)"
    )
fi

for TARGET in $TARGETS; do
    slug="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"
    out="$OUTPUT_DIR/$TARGET"
    mkdir -p "$out"

    echo ">> Building DocC for $TARGET → $out"
    swift package --allow-writing-to-directory "$out" \
        generate-documentation \
        --target "$TARGET" \
        --fallback-bundle-identifier "${HOSTING_BASE_PATH}.${slug}" \
        --output-path "$out" \
        --emit-digest \
        --disable-indexing \
        --transform-for-static-hosting \
        --hosting-base-path "${HOSTING_BASE_PATH}/${TARGET}" \
        ${SOURCE_FLAGS[@]+"${SOURCE_FLAGS[@]}"} \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
done

if [[ "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    LLMS="$OUTPUT_DIR/llms.txt"
    {
        echo "# ${HOSTING_BASE_PATH} — DocC export for LLM consumption"
        echo
        echo "Generated $(date -u +%FT%TZ) from swift-docc."
        echo "Targets: $TARGETS"
        echo
        for TARGET in $TARGETS; do
            find "$OUTPUT_DIR/$TARGET/data" -name '*.md' -type f 2>/dev/null \
                | sort \
                | while IFS= read -r f; do
                    rel="${f#$OUTPUT_DIR/}"
                    echo
                    echo "---"
                    echo "## $rel"
                    echo
                    cat "$f"
                done
        done
    } > "$LLMS"
    echo "Wrote $LLMS ($(wc -l < "$LLMS" | tr -d ' ') lines)."
fi

echo
echo "Docs written to $OUTPUT_DIR/. Open $OUTPUT_DIR/<Target>/index.html"
echo "or push to gh-pages."
