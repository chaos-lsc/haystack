#!/usr/bin/env bash
set -euo pipefail

# Run as the deployment user. All downloaded tools live in this application directory.
workspace="${1:-$HOME/trust-rag-v2}"
cd "$workspace"
mkdir -p .tools/hatch .tools/qdrant .hatch-data .hatch-cache
curl --fail --location --retry 3 --connect-timeout 15 \
  https://github.com/pypa/hatch/releases/download/hatch-v1.18.0/hatch-x86_64-unknown-linux-gnu.tar.gz \
  --output .tools/hatch-linux.tar.gz
printf '%s\n' '8bbfa2464773e159c2b90126267455848871522f24478c7570328cc96b7caa03  .tools/hatch-linux.tar.gz' | sha256sum --check
tar -xzf .tools/hatch-linux.tar.gz -C .tools/hatch
curl --fail --location --retry 3 --connect-timeout 15 \
  https://github.com/qdrant/qdrant/releases/download/v1.19.1/qdrant-x86_64-unknown-linux-gnu.tar.gz \
  --output .tools/qdrant-linux.tar.gz
printf '%s\n' 'eef986e769d4d3e806dd2d546e1b4ecdd416211e54d34b4ed764fac7c58e1085  .tools/qdrant-linux.tar.gz' | sha256sum --check
tar -xzf .tools/qdrant-linux.tar.gz -C .tools/qdrant
export HATCH_DATA_DIR="$workspace/.hatch-data"
export HATCH_CACHE_DIR="$workspace/.hatch-cache"
export HATCH_PYTHON=/usr/bin/python3
export HAYSTACK_TELEMETRY_ENABLED=false
"$workspace/.tools/hatch/hatch" --version
"$workspace/.tools/hatch/hatch" -e trust-rag run lab --help
"$workspace/.tools/qdrant/qdrant" --version
