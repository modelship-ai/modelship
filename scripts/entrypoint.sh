#!/bin/bash
set -e

TARGET_UID=${MSHIP_UID:-1000}
TARGET_GID=${MSHIP_GID:-1000}

# Root: fix cache ownership, then drop privileges. Non-root can't chown, so run as-is.
if [ "$(id -u)" = "0" ]; then
    # Fixed image paths only, never env-supplied ones. Skipped once owned: the walk is slow.
    for dir in /.cache /opt/mship/node-cache; do
        if [ -d "$dir" ] && [ "$(stat -c '%u:%g' "$dir")" != "$TARGET_UID:$TARGET_GID" ]; then
            chown -R "$TARGET_UID:$TARGET_GID" "$dir"
        fi
    done
    chown "$TARGET_UID:$TARGET_GID" /modelship
    exec gosu "$TARGET_UID:$TARGET_GID" "$@"
fi

exec "$@"
