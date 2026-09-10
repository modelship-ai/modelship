import os

# Disable Ray's auto-init hook: an unmocked Ray call in a unit test then raises
# instead of silently booting a real head whose subprocesses outlive the test.
# Read once at `import ray`, so it must be set before conftest pulls ray in.
os.environ.setdefault("RAY_ENABLE_AUTO_CONNECT", "0")
