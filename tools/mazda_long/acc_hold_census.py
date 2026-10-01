#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Per body standstill-hold episode (EPB.HOLD_STATE == HOLDING): when the stock radar (bus 0)
dropped STOPPING and relaxed to raw -1, whether GEAR.BRAKE_HOLD ever followed, and whether
RESUME_UNLATCHING preceded the exit. Stock rows only: under alpha long our frames are not on
bus 0, so OPLONG rows show the body side alone.

2026-09-30: stock relaxes 0-40 ms after HOLDING in 43/43 holds, 15 of them with no BRAKE_HOLD
at all, and pulses before every exit that is not a gas drive-off. So HOLD_STATE is the hold
handshake and BRAKE_HOLD only joins it with Auto Hold armed.

HOLD_STATE (0x79 byte 2, low nibble): 1 boot, 2 idle, 3 holding, 5 releasing. The high nibble
reads 0x3 on hold-capable bodies (CX-5 2017+, CX-9) and 0x0 on the 2016.5 CX-5 KE.

Usage:  acc_hold_census.py <rlog> [<rlog> ...]
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "opendbc_repo"))

from opendbc.car.mazda.carstate import HOLD_STATE_HOLDING
from openpilot.tools.lib.logreader import LogReader

CRZ_INFO, EPB, GEAR = 0x21b, 0x79, 0x228


def accel_cmd_raw(dat):
  return (((dat[2] & 0x03) << 11) | (dat[3] << 3) | (dat[4] >> 5)) - 4096


def episodes(path):
  """Yield one dict per HOLDING episode that ends inside the log."""
  t0 = None
  state = None
  brake_hold = False
  stop = cmd = None
  gas = False
  sent = False
  ep = None
  for m in LogReader(path):
    t = m.logMonoTime * 1e-9
    t0 = t if t0 is None else t0
    w = m.which()
    if w == "sendcan":
      sent |= any(c.address == CRZ_INFO for c in m.sendcan)
    elif w == "carState":
      gas = m.carState.gasPressed
    elif w == "can":
      for c in m.can:
        if c.src != 0:
          continue
        dat = bytes(c.dat)
        if c.address == CRZ_INFO:
          stop, cmd = (dat[5] >> 2) & 1, accel_cmd_raw(dat)
          if ep is not None:
            if ep["relax"] is None and not stop and cmd == -1:
              ep["relax"] = t - ep["t"]
            if (dat[6] >> 6) & 1 and ep["unlatch"] is None:
              ep["unlatch"] = t
        elif c.address == GEAR:
          brake_hold = bool((dat[2] >> 4) & 1)
          if brake_hold and ep is not None and ep["brake_hold"] is None:
            ep["brake_hold"] = t - ep["t"]
        elif c.address == EPB:
          new = dat[2] & 0x0f
          if new == HOLD_STATE_HOLDING and state != HOLD_STATE_HOLDING:
            ep = {"t": t, "stop0": stop, "cmd0": cmd, "relax": None, "unlatch": None,
                  "brake_hold": 0.0 if brake_hold else None}
          elif new != HOLD_STATE_HOLDING and state == HOLD_STATE_HOLDING and ep is not None:
            ep.update(t_rel=ep["t"] - t0, dur=t - ep["t"], exit=new, gas=gas, oplong=sent,
                      unlatch_lead=None if ep["unlatch"] is None else t - ep["unlatch"])
            yield ep
            ep = None
          state = new


def fmt(x):
  return "-" if x is None else f"{x:.2f}"


if __name__ == "__main__":
  for path in map(os.path.abspath, sys.argv[1:]):
    name = os.path.basename(os.path.dirname(path)) if os.path.basename(path) == "rlog.zst" else os.path.basename(path)
    for ep in episodes(path):
      who = "OPLONG" if ep["oplong"] else "stock"
      onset = f"onset stop={ep['stop0']} cmd={ep['cmd0']}"
      body = f"relax_after={fmt(ep['relax'])} brake_hold_after={fmt(ep['brake_hold'])}"
      end = f"exit->{ep['exit']} unlatch_lead={fmt(ep['unlatch_lead'])} gas={int(ep['gas'])}"
      print(f"{name} {who} t={ep['t_rel']:7.2f} dur={ep['dur']:5.1f} {onset} | {body} | {end}")
