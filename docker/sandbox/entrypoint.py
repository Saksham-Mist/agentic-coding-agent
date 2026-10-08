"""Sandbox entrypoint — step 2.1 stub.

Just proves the image starts fast and can be driven over stdin/stdout.
The real harness (function mode / class mode, ListNode/TreeNode
conversion, per-test diffing) is built in step 2.3 and replaces the
body of main() below; the Dockerfile and image layout stay the same.
"""
import json
import sys


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"status": "ready", "received": None}))
        return
    payload = json.loads(raw)
    print(json.dumps({"status": "ready", "received": payload}))


if __name__ == "__main__":
    main()
