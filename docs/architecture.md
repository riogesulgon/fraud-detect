# Architecture

Synthetic data feeds a reproducible training script, which writes a local model artifact. FastAPI exposes scoring and model-health endpoints. CI runs lint, formatting, tests, training smoke checks, and container builds.

The service intentionally has no payment-rail integration, customer data, automated approval or decline decision, or cloud deployment requirement in v1.
