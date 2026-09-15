#!/bin/bash
set -e

# Root: pick the runtime user, fix cache ownership, then drop privileges. Non-root can't chown, so run as-is.
if [ "$(id -u)" = "0" ]; then
    TARGET_UID=$(stat -c '%u' /.cache)
    TARGET_GID=$(stat -c '%g' /.cache)
    if [ "$TARGET_UID" = "0" ]; then
        TARGET_UID=$(id -u modelship)
        TARGET_GID=$(id -g modelship)
    fi
    TARGET_UID=${MSHIP_UID:-$TARGET_UID}
    TARGET_GID=${MSHIP_GID:-$TARGET_GID}

    # getpwuid() fails for a UID with no passwd entry.
    getent group "$TARGET_GID" >/dev/null || groupadd -g "$TARGET_GID" mship
    getent passwd "$TARGET_UID" >/dev/null || useradd -m -u "$TARGET_UID" -g "$TARGET_GID" mship

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
