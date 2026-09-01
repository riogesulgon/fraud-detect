# Operations

Run `make install`, `make data`, `make train`, and `make test` locally. Start the service with `make run`, then inspect `/healthz`, `/metrics`, and `/v1/model-info`.

Model artifacts and reports are local-only and ignored by git. Logs must contain anonymous request metadata only. This demonstrator does not silently retrain models.
