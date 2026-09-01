# Walkthrough: building an MLOps risk platform, end to end

This is the story of this repository: what was built, in what order, what broke, and what each decision cost. It is written for a reviewer deciding whether the author understands MLOps beyond the buzzwords.

## What this is

A transaction-risk scoring platform served over HTTP: two model versions behind a stable API contract, probability calibration with a chosen operating point, drift reporting, reproducible training from a real (synthetic, CC0-licensed) Kaggle dataset, CI that trains and smoke-tests in the cloud, and a release pipeline that publishes a scanned Docker image to GHCR. It is explicitly a review-queue demonstrator, not an approve/decline system.

The sequence matters: API contract first, then real data, then calibration, then packaging, then publishing. Each step was verified before the next started.

## The path

### 1. Contract before model

The first commit was a FastAPI service with a fixed eight-field request contract, a health endpoint, metrics, and tests. Only then did model training exist. The payoff came later: when the toy dataset was replaced by a 1M-row Kaggle set with 27 features, the API surface did not change. An adapter (`src/kaggle_adapter.py`) maps rich features onto the stable contract, so v1 clients never noticed the data swap.

### 2. Real data, honestly handled

The Kaggle dataset (1M rows, 27 features, ~1.95% fraud, CC0-1.0) is downloaded by script, never committed; a SHA-256 of the training file is recorded in model metadata and exposed via `/v1/model-info`, so any served artifact can be tied to the exact bytes it learned from (with the honest caveat: file bytes, not row sampling).

Two splits matter here. The v1 adapter model uses a chronological 70/15/15 split; the calibrated v2 model uses chronological 60/20/20 with a final test window the threshold never sees. Chronological ordering matters for transaction data: a random split leaks the future into training.

### 3. Calibration as a product decision

Raw gradient-boosting probabilities were poorly calibrated (Brier 0.130, ECE 0.278). Platt sigmoid calibration on a held-out slice moved them to Brier 0.017, ECE 0.001, with ROC-AUC intact (0.831). Calibration does not improve ranking; it makes probabilities mean what they say, which matters when humans act on them.

The operating threshold is a product decision, not a training artifact. The first cut maximized precision subject to validation recall of at least 80%: recall 81.3%, precision 4.6% — about 22 alerts per confirmed fraud. A later pass added an explicit cost model (missed fraud costs 20x a wasted review, documented in code and report), which selected threshold 0.077: recall 40.6%, precision 15.3% — about 6.5 alerts per fraud. Both candidates are preserved side by side in `reports/calibration.json`; the model card explains the trade and says plainly that the 20:1 ratio is an assumption to be re-derived from real review capacity. Same calibrated probabilities, one config value to switch.

### 4. Packaging decisions with teeth

- The ~200KB calibrated artifact is committed, with its SHA-256 in the README, so the demo serves out of the box; the raw data stays out of git.
- `scikit-learn` is pinned to 1.8.0 everywhere, after the artifact refused to load under 1.9.0 in a container. Model artifacts and runtime versions drift apart silently; the pin is the fix, and the CI failure was the receipt.
- Docker images train the deterministic v1 model at build time, so a clean checkout produces a self-sufficient image.

### 5. CI that actually exercises the system

Every push: ruff, deterministic v1 training, 11 pytest tests (API contracts, artifact compatibility, serving hardening), optional Kaggle fetch + model reports when credentials exist, then a container smoke test that posts real payloads at both API versions in a built image. Release tags add a Trivy CRITICAL gate and a GHCR push.

### 6. What broke, and what it taught

| Breakage | Lesson | Fix |
|---|---|---|
| CI resolved ruff 0.16 while local used 0.15; import-sort rules differed | Lint must be pinned like any dependency | `ruff==0.16.5` in `pyproject.toml` |
| Tests ran before the model existed in CI | Step order is part of correctness | Train before pytest |
| Artifact failed to unpickle under sklearn 1.9 | Serialization is a version contract | Pin sklearn; document why |
| Published image served `/healthz` degraded | The image that ships is not the image CI tested; release built from a clean checkout | Train in the Dockerfile; verify the published image anonymously |
| trivy-action pin did not exist; its installer then failed on runners | Pin to real versions; have a fallback path | Scan via pinned `aquasec/trivy` container image |

### 7. Serving hardening

Requests carry an `X-Request-ID` (client-supplied or generated), every request emits one structured JSON log line (method, path, status, duration), scoring endpoints are rate-limited per client with `429` + `Retry-After`, and a Prometheus counter tracks rejections. Request bodies never reach the logs; a test asserts that. The rate limiter is single-process by design and documented as such.

## The numbers, honestly

| | v1 (adapter) | v2 (calibrated, cost-based) |
|---|---|---|
| Features | 8-field adapter | 27 |
| Calibration | none | Platt sigmoid |
| ROC-AUC | — | 0.8314 |
| PR-AUC | — | 0.1996 |
| Test recall | — | 40.6% (81.3% at the high-recall alternative) |
| Test precision | — | 15.3% (4.6% at the high-recall alternative) |

The dataset is synthetic. The metrics describe this dataset, not fraud detection in the wild. The value of the project is the machinery: contract-first APIs, calibrated probabilities, explicit threshold economics, reproducible packaging, and a pipeline that verifies all of it on every push.

## What production would add

Cross-validation and multiple seeds; monitoring on calibration drift and alert-volume drift (the Evidently report is a starting point); retraining triggered by those signals; authentication and audit logging; shadow deployment and champion/challenger; and a governance layer (bias analysis, privacy review, human-in-the-loop protocol) before any real decision path.