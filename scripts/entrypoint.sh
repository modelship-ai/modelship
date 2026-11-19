#!/bin/bash
# Runs "$@" as the image's user. The image's `mship` (scripts/bin/mship) comes through here, so
# `docker run` and `docker exec` get the same user.
set -e

# Non-root can't chown, so run as-is. Root in a user namespace (rootless podman/Docker, userns-remap)
# is already an unprivileged host user and keeps the GPU groups it was given, so run as-is too.
if [ "$(id -u)" != "0" ] || [ "$(awk 'NR == 1 { print $1, $2, $3 }' /proc/self/uid_map)" != "0 0 4294967295" ]; then
    exec "$@"
fi

# The owner of /.cache, or the image's user when it is root-owned.
TARGET_UID=$(stat -c '%u' /.cache)
TARGET_GID=$(stat -c '%g' /.cache)
if [ "$TARGET_UID" = "0" ]; then
    TARGET_UID=$(id -u modelship)
    TARGET_GID=$(id -g modelship)
fi

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

# setpriv keeps root's HOME; Ray writes its auth token under it.
HOME=$(getent passwd "$TARGET_UID" | cut -d: -f6)
export HOME

# The groups that own the GPU render nodes (--device /dev/dri), the user's only supplementary groups.
groups=$(stat -c '%g' /dev/dri/renderD* 2>/dev/null | sort -un | grep -vx -e 0 -e "$TARGET_GID" | paste -sd, -)
if [ -n "$groups" ]; then
    exec setpriv --reuid="$TARGET_UID" --regid="$TARGET_GID" --groups="$groups" -- "$@"
fi
exec setpriv --reuid="$TARGET_UID" --regid="$TARGET_GID" --clear-groups -- "$@"
