#!/usr/bin/env bash
# Presents the comma as a USB gadget so a Jetson can enumerate it.
#
# Only when the user has turned the link on: a gadget presented by default
# turns link_configured() true and routed manager away from the user's bundle.
# The param file is read directly, by gadget.params_dir()'s rule, not through
# openpilot.common.params: this runs before build.py, and on the first boot
# after an update the params library is not built yet (the updater's git clean
# removes it). Python here also cost every boot, link on or off, 1.2 to 1.7 s.
set -u
[ -f /AGNOS ] || exit 0
BASEDIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
STATUS=/dev/shm/jetlink-gadget

# put_bool writes 1; gadget.param_bool also takes true. A shell builtin read,
# not $(cat): this is on the boot path
param_on() {
  local v=""
  IFS= read -r v 2>/dev/null < "${PARAMS_ROOT:-/data/params}/${OPENPILOT_PREFIX:-d}/$1"
  case "$v" in 1|true|True) return 0 ;; *) return 1 ;; esac
}

param_on JetlinkEnabled || exit 0

REPO="$BASEDIR/jetlink_repo"
if [ ! -d "$REPO/jetlink" ]; then
  # submodule registered but never fetched; say so where the offroad alert reads
  echo "error: the jetlink package is not installed; run git submodule update --init jetlink_repo" \
    > "$STATUS" 2>/dev/null || true
  exit 0
fi

# An iPhone needs the composite gadget with a network interface; a Jetson or a
# Mac the plain one. Accelerator Link "iOS" is JetlinkIOS
MODE=""
param_on JetlinkIOS && MODE="--ios"

# the endpoints must exist before jetlinkd or modeld can open them, and that
# needs root. setup_gadget.sh leaves the reason in $STATUS for the offroad alert
sudo -n bash "$REPO/scripts/setup_gadget.sh" ${MODE:+"$MODE"} >/dev/null ||
  echo "jetlink: USB gadget setup failed" >&2
exit 0
