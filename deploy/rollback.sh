#!/bin/bash
set -euo pipefail

REGISTRY="work_dirs/model_registry.json"
TARGET_LINK="deploy/current_model.pth"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --target-link)
      TARGET_LINK="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ! -f "$REGISTRY" ]]; then
  echo "Registry not found: $REGISTRY"
  exit 1
fi

ROLLBACK_CKPT=$(python - <<PY
import json
from pathlib import Path
reg = json.loads(Path("$REGISTRY").read_text(encoding="utf-8"))
if not isinstance(reg, list) or len(reg) == 0:
    raise SystemExit(1)
promoted = [e for e in reg if str(e.get("status", "")) == "promoted" and e.get("checkpoint")]
if not promoted:
    raise SystemExit(1)
print(promoted[-1]["checkpoint"])
PY
)

if [[ -z "${ROLLBACK_CKPT:-}" ]]; then
  echo "No promoted checkpoint found for rollback."
  exit 1
fi

mkdir -p "$(dirname "$TARGET_LINK")"
ln -sfn "$ROLLBACK_CKPT" "$TARGET_LINK"
echo "Rollback target set:"
echo "  checkpoint: $ROLLBACK_CKPT"
echo "  symlink:    $TARGET_LINK"
