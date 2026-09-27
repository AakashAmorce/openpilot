#!/usr/bin/env bash
# The comma's USB-C port, for a link that runs over USB. Root, because the role
# lever is in debugfs; usbport.py says why this and not the obvious knobs.
#
#   hold  the port held at sink, which makes the far end the host
#   off   dual role, as AGNOS boots it
#
# Exits 3 on a device without the PMI8998 lever, where there is nothing to do.
set -u
VOTER=/sys/kernel/debug/pmic-votable/DISABLE_POWER_ROLE_SWITCH

[ -d "$VOTER" ] || exit 3

case "${1:-}" in
  hold)
    # force_val first: forcing applies whatever force_val holds at that moment
    echo 1 > "$VOTER/force_val" && echo 1 > "$VOTER/force_active"
    ;;
  off)
    # letting go applies the voters' own result, which is dual role
    echo 0 > "$VOTER/force_active" && echo 0 > "$VOTER/force_val"
    ;;
  *)
    echo "usage: $0 hold|off" >&2
    exit 2
    ;;
esac
