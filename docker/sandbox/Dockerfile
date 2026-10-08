# Sandbox image for execute_code (step 2.2/2.3). Prebuilt so each
# execution only pays container-start cost, not dependency-install
# cost — see the design notes' 10s wall-clock timeout per execution.
FROM python:3.11-slim

# Stdlib only for v1 (see the design notes' 18 tasks) — no pip install
# needed at build time beyond the base image, which keeps builds fast
# and keeps the sandbox's dependency surface minimal.

WORKDIR /sandbox
COPY entrypoint.py /sandbox/entrypoint.py

ENTRYPOINT ["python", "/sandbox/entrypoint.py"]
