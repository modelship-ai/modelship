#!/bin/bash
set -e

# Use the environment variables set during build, or default to 1000
TARGET_UID=${MSHIP_UID:-1000}
TARGET_GID=${MSHIP_GID:-1000}

# The prod ENTRYPOINT prepends `mship`, so `docker run` args arrive as its
# subcommand; the dev one prepends nothing, keeping plain passthrough.
#
# When started as root (the standalone `docker run` path), fix up ownership for
# any root-owned bind mount and drop privileges to the unprivileged user before
# exec'ing the command.
#
# Under Kubernetes (incl. KubeRay) the pod instead sets
# `securityContext.runAsUser`, so this script may already be running
# unprivileged. A non-root user cannot chown files it doesn't own — the chown
# would fail under `set -e` and crash the container — and there is nothing to
# drop to. In that case skip straight to the command. (KubeRay also overrides
# the container command, bypassing this ENTRYPOINT entirely for Ray pods; this
# branch covers any pod that keeps it, e.g. the deploy Job.)
if [ "$(id -u)" = "0" ]; then
    # Fix permissions for the cache directories (may be root-owned bind mounts).
    # `chown -R` walks the whole weight cache, which is very slow on NFS/EFS, so
    # only do it when the directory isn't already owned by the target user — i.e.
    # the first run / a freshly-mounted root-owned volume. Restarts skip the walk.
    for dir in /.cache "${MSHIP_NODE_CACHE_DIR:-}"; do
        if [ -n "$dir" ] && [ -d "$dir" ] && [ "$(stat -c '%u:%g' "$dir")" != "$TARGET_UID:$TARGET_GID" ]; then
            chown -R "$TARGET_UID:$TARGET_GID" "$dir"
        fi
    done
    # Also ensure the workspace has the right permissions.
    chown "$TARGET_UID:$TARGET_GID" /modelship
    # Drop privileges and execute the main command (gosu takes UID:GID).
    exec gosu "$TARGET_UID:$TARGET_GID" "$@"
fi

# Already unprivileged (k8s securityContext): run the command as-is.
exec "$@"
