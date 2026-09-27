#!/usr/bin/env bash
# The comma's USB-C port, for a link that runs over USB. Root, because the role
# lever is in debugfs; usbport.py says why these writes and not the obvious ones.
#
#   link     USB PD off, so a host that plugs in next gets no PD contract, and
#            the port dual role, whatever an owner killed mid-hold left behind
#   hold     the port held at sink, which makes the far end the host
#   release  the port back to dual role
#   off      dual role and PD back on, as AGNOS boots it
#
# Exits 3 on a device without the PMI8998 lever, where there is nothing to do.
set -u
VOTER=/sys/kernel/debug/pmic-votable/DISABLE_POWER_ROLE_SWITCH
PD=/sys/module/policy_engine/parameters/disable_usb_pd

[ -d "$VOTER" ] && [ -f "$PD" ] || exit 3

case "${1:-}" in
  hold)
    # force_val first: forcing applies whatever force_val holds at that moment
    echo 1 > "$VOTER/force_val" && echo 1 > "$VOTER/force_active"
    ;;
  link|release|off)
    # letting go applies the voters' own result, which is dual role
    echo 0 > "$VOTER/force_active" && echo 0 > "$VOTER/force_val" || exit 1
    case "$1" in
      link) echo 1 > "$PD" ;;
      off) echo 0 > "$PD" ;;
    esac
    ;;
  *)
    echo "usage: $0 link|hold|release|off" >&2
    exit 2
    ;;
esac
