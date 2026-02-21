# DeepRoof Release Checklist

## 1. Pre-release validation
1. `python -m py_compile $(git ls-files '*.py')`
2. `pytest -q`
3. Run inference regression suite on benchmark tiles.
4. Confirm KPI metrics JSON includes at least:
   - `mIoU`
   - `AP50`
   - `BFScore`

## 2. KPI gate
1. Register candidate model:
   - `python tools/model_registry.py --checkpoint <ckpt> --metrics-json <metrics.json>`
2. Ensure latest registry entry status is `promoted`.

## 3. Deployment readiness
1. Build container image.
2. Run entrypoint smoke checks in clean environment:
   - `deploy/docker_entrypoint.sh`
3. Validate output GeoJSON and QA flags (`qa_all_ok`).

## 4. Canary rollout
1. Deploy to canary environment with fixed benchmark AOIs.
2. Compare metrics and output counts vs previous stable model.
3. Monitor errors for at least 24h.

## 5. Full rollout
1. Promote image to production.
2. Tag git commit and checkpoint.
3. Archive run manifest and metrics report.

## 6. Rollback
1. Execute:
   - `deploy/rollback.sh --registry work_dirs/model_registry.json --target-link deploy/current_model.pth`
2. Redeploy using selected rollback checkpoint.
