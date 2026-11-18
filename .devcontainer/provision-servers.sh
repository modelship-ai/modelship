#!/usr/bin/env bash
# Fetches modelship's pinned llama-server and sd-server builds via
# bootstrap/mship_bootstrap/{llama_cpp,sd_cpp}.py (stdlib-only, imported straight from source),
# since the `dev` Dockerfile target never runs `mship bootstrap`. Symlinks /.venv into the
# envs/<variant>/.venv layout those modules expect, so they find torch's bundled CUDA libs.
set -euo pipefail

# Must be unset: provision() short-circuits on these env vars, and
# remoteEnv points them at this script's own output.
unset MSHIP_LLAMA_SERVER_BIN MSHIP_SD_SERVER_BIN

MSHIP_HOME="${MSHIP_HOME:-/opt/mship}"
mkdir -p "$MSHIP_HOME/envs/cuda"
ln -sfn /.venv "$MSHIP_HOME/envs/cuda/.venv"

# provision() also prints progress to stdout, so capture the path via a file
# instead of $(...) to avoid mixing the two.
OUT_FILE=$(mktemp)
trap 'rm -f "$OUT_FILE"' EXIT

provision() {
  local module=$1 name=$2
  python3 - "$module" "$OUT_FILE" <<'PY'
import importlib
import sys

module, out_file = sys.argv[1:]
sys.path.insert(0, "/modelship/bootstrap")
from mship_bootstrap.variants import VARIANTS

wrapper = importlib.import_module(f"mship_bootstrap.{module}").provision(VARIANTS["cuda"])
if wrapper is None:
    sys.exit(f"error: {module} provisioning failed (see warning above)")
with open(out_file, "w") as f:
    f.write(wrapper)
PY
  local wrapper
  wrapper=$(cat "$OUT_FILE")
  # Stable path, so remoteEnv doesn't need updating on a tag bump.
  local link="$MSHIP_HOME/bin/$name.sh"
  mkdir -p "$(dirname "$link")"
  ln -sfn "$wrapper" "$link"
  echo "$name ready: $link -> $wrapper"
}

provision llama_cpp llama-server
provision sd_cpp sd-server
