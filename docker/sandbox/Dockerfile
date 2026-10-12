# Sandbox image for execute_code (step 2.2/2.3). Prebuilt so each
# execution only pays container-start cost, not dependency-install
# cost — each execution has a 10s wall-clock timeout.
FROM python:3.11-slim

# Stdlib only for v1 (the 18 catalog tasks need nothing else) — no pip install
# needed at build time beyond the base image, which keeps builds fast
# and keeps the sandbox's dependency surface minimal.

WORKDIR /sandbox
COPY entrypoint.py /sandbox/entrypoint.py

ENTRYPOINT ["python", "/sandbox/entrypoint.py"]
