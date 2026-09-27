#!/usr/bin/env bash
# The comma's USB-C port, for a link that runs over USB. Root, because the role
# lever is in debugfs; usbport.py says why these writes and not the obvious ones.
#
#   hold  USB PD off and the port held at sink, which makes the far end the host
#   off   dual role and PD on, as AGNOS boots it
#
# Exits 3 on a device without the PMI8998 lever, where there is nothing to do.
set -u
VOTER=/sys/kernel/debug/pmic-votable/DISABLE_POWER_ROLE_SWITCH
PD=/sys/module/policy_engine/parameters/disable_usb_pd

[ -d "$VOTER" ] && [ -f "$PD" ] || exit 3

case "${1:-}" in
  hold)
    # PD first: the hold is an unplug and replug, and the policy engine reads
    # this when the replug starts. force_val before force_active, because
    # forcing applies whatever force_val holds at that moment
    echo 1 > "$PD" && echo 1 > "$VOTER/force_val" && echo 1 > "$VOTER/force_active"
    ;;
  off)
    # letting go applies the voters' own result, which is dual role
    echo 0 > "$VOTER/force_active" && echo 0 > "$VOTER/force_val" && echo 0 > "$PD"
    ;;
  *)
    echo "usage: $0 hold|off" >&2
    exit 2
    ;;
esac
