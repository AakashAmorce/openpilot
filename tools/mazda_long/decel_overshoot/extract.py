"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Extract a 20 Hz decel-overshoot table from every device rlog: speed, dash, plan source and
targets (incl. each limiter's own aTarget), ICBM command, MRCC ACCEL_CMD (0x21B), grade,
lead and bookmarks. One npz per segment; re-runs skip segments already cached.

Usage: extract.py [rlog_root] [out_dir] [workers]
  rlog_root defaults to tools/mazda_long/device_data (dirs <route>--<seg>/rlog.zst)
"""
import glob
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

from openpilot.common.constants import CV

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'device_data')
OUT = os.path.join(HERE, '..', 'test_data', 'decel_overshoot')
MPH = CV.MS_TO_MPH

COLS = ['t', 'v', 'a', 'dash', 'crz', 'gas', 'brake', 'ss', 'eng', 'src', 'vT', 'aT', 'visSt', 'visVT', 'visAT', 'curLat',
        'maxLat', 'vAhead', 'mapSt', 'mapVT', 'mapAT', 'slaAT', 'icbmSt', 'icbmBtn', 'icbmVT', 'cmd', 'accAct', 'pitch',
        'leadSt', 'dRel', 'vRel', 'curv', 'opLong']


def accel_cmd(dat):
  # CRZ_INFO.ACCEL_CMD 17|13@0+ (0.001,-4.096)
  return ((((dat[2] & 0x3) << 11) | (dat[3] << 3) | (dat[4] >> 5)) - 4096) * 0.001


def run(seg, root=ROOT, out_dir=OUT):
  out = os.path.join(out_dir, seg + '.npz')
  path = os.path.join(root, seg, 'rlog.zst')
  if os.path.exists(out) or not os.path.exists(path):
    return seg, None
  from openpilot.tools.lib.logreader import LogReader
  fp = ''
  cur = dict.fromkeys(COLS, np.nan)
  for k in ('eng', 'gas', 'brake', 'ss', 'crz', 'src', 'visSt', 'mapSt', 'icbmSt', 'icbmBtn', 'accAct', 'leadSt', 'opLong'):
    cur[k] = 0
  rows, bms, n = [], [], 0
  try:
    for m in LogReader(path):
      w = m.which()
      if w == 'carState':
        c = m.carState
        cur.update(t=m.logMonoTime * 1e-9, v=c.vEgo, a=c.aEgo, dash=c.cruiseState.speedCluster, crz=int(c.cruiseState.enabled),
                   gas=int(c.gasPressed), brake=int(c.brakePressed), ss=int(c.standstill))
        n += 1
        if n % 5 == 0:
          rows.append([cur[k] for k in COLS])
      elif w == 'selfdriveState':
        cur['eng'] = int(m.selfdriveState.enabled)
      elif w == 'longitudinalPlanSP':
        p = m.longitudinalPlanSP
        vis, mp = p.smartCruiseControl.vision, p.smartCruiseControl.map
        cur.update(src=p.longitudinalPlanSource.raw, vT=p.vTarget, aT=p.aTarget, visSt=vis.state.raw, visVT=vis.vTarget,
                   visAT=vis.aTarget, curLat=vis.currentLateralAccel, maxLat=vis.maxPredictedLateralAccel, vAhead=vis.vAheadMin,
                   mapSt=mp.state.raw, mapVT=mp.vTarget, mapAT=mp.aTarget, slaAT=p.speedLimit.assist.aTarget)
      elif w == 'carControlSP':
        i = m.carControlSP.intelligentCruiseButtonManagement
        cur.update(icbmSt=i.state.raw, icbmBtn=i.sendButton.raw, icbmVT=i.vTarget)
      elif w == 'can':
        for c in m.can:
          if c.address == 0x21B and c.src == 0:
            cur['cmd'] = accel_cmd(c.dat)
            cur['accAct'] = (c.dat[4] >> 1) & 1
      elif w == 'liveLocationKalman':
        o = m.liveLocationKalman.orientationNED.value
        if len(o) >= 2:
          cur['pitch'] = o[1]
      elif w == 'radarState':
        lead = m.radarState.leadOne
        cur.update(leadSt=int(lead.present), dRel=lead.dRel, vRel=lead.vRel)
      elif w == 'controlsState':
        cur['curv'] = m.controlsState.curvature
      elif w == 'carParams':
        cur['opLong'] = int(m.carParams.openpilotLongitudinalControl)
        fp = m.carParams.carFingerprint
      elif w in ('userBookmark', 'bookmarkButton'):
        bms.append(m.logMonoTime * 1e-9)
  except Exception:
    if not rows:
      return seg, 'unreadable'
  np.savez_compressed(out, data=np.asarray(rows, dtype=np.float64), cols=np.array(COLS), bms=np.array(bms), fp=fp)
  return seg, len(rows)


def seg_key(f):
  """Sort key for <route>--<seg>.npz: the segment number, so 10 follows 9."""
  return int(f.rsplit('--', 1)[1][:-4])


def route_files(route, cache):
  """A route's cached segment files in segment order."""
  return sorted(glob.glob(os.path.join(cache, route + '--*.npz')), key=seg_key)


def load_route(route, out=OUT):
  """Concatenate a route's cached segments into {column: array}, plus its bookmark times."""
  files = route_files(route, out)
  ds, bms, cols = [], [], None
  for f in files:
    z = np.load(f)
    if z['data'].shape[0]:
      ds.append(z['data'])
      bms += list(z['bms'])
      cols = list(z['cols'])
  if not ds:
    return None, []
  d = np.concatenate(ds)
  return {c: d[:, i] for i, c in enumerate(cols)}, np.array(bms)


def routes(out=OUT):
  return sorted({f.rsplit('--', 1)[0] for f in os.listdir(out) if f.endswith('.npz')})


if __name__ == '__main__':
  root = sys.argv[1] if len(sys.argv) > 1 else ROOT
  out_dir = sys.argv[2] if len(sys.argv) > 2 else OUT
  os.makedirs(out_dir, exist_ok=True)
  segs = sorted(d for d in os.listdir(root) if d.count('--') == 2)
  with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as pool:
    for seg, res in pool.imap_unordered(partial(run, root=root, out_dir=out_dir), segs, chunksize=2):
      if res == 'unreadable':
        print(seg, res)
  print(f"{len(routes(out_dir))} routes cached in {out_dir}")
