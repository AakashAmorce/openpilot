#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Fit speed_dependent.toml seeds from rlogs, the way the per-bin learner would.

  extract: replays torqued's point filter (TorqueEstimator.handle_log) over every Mazda rlog
           listed in a file, with the steer axis in applied CAN counts / STEER_MAX
           (carOutput.torqueOutputCan), so every build lands on one scale whatever it ran.
  fit:     runs torqued_ext's per-bin fit (fit_torque_points) on those points, at most
           POINTS_PER_BUCKET per steer bucket and |steer| < 0.5, over torqued_ext's bin bounds,
           with a segment bootstrap for the spread. Bins are the toml entry's centers or --centers.

The fit depends on the controller as well as the car (the TLS runs on a wide point cloud), so
seed from the builds users will run: --since takes the build date of each segment's
gitCommit from this checkout. See docs/zoompilot/lateral-tune.md, "Seeds".

  speed_bin_seeds.py extract rlogs.txt points.pkl
  speed_bin_seeds.py fit points.pkl --platform MAZDA_CX5_2022 --since 2026-09-01
"""
import argparse
import functools
import os
import pickle
import subprocess
import sys
from multiprocessing import Pool
from types import SimpleNamespace as NS

import numpy as np

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASEDIR)

from opendbc.car.mazda.values import CarControllerParams
from opendbc.sunnypilot.car.interfaces import get_speed_dep_config
from openpilot.selfdrive.locationd.torqued import POINTS_PER_BUCKET, STEER_BUCKET_BOUNDS
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import TorqueEstimatorExt, fit_torque_points

FEED = {'carControl', 'carOutput', 'carState', 'extrinsicsCalibration', 'deviceMotion', 'lateralDelay'}
STEER_MAX = CarControllerParams.EPS_STEER_MAX


def _init():
  from openpilot.common.prefix import OpenpilotPrefix
  OpenpilotPrefix(f"seeds{os.getpid()}").__enter__()


def _extract(path):
  from openpilot.tools.lib.logreader import LogReader
  from openpilot.selfdrive.locationd.torqued import TorqueEstimator
  msgs, CP, meta = [], None, {'path': path}
  try:
    for m in LogReader(path):
      w = m.which()
      if w == 'carParams' and CP is None:
        CP = m.carParams
      elif w == 'initData':
        meta.update(git=m.initData.gitCommit[:10], dongle=m.initData.dongleId)
      elif w in FEED:
        msgs.append((m.logMonoTime * 1e-9, w, getattr(m, w)))
  except Exception as e:
    return {**meta, 'err': repr(e)[:160]}
  msgs.sort(key=lambda x: x[0])  # cheaper than sorting the whole log by time
  if CP is None or not str(CP.carFingerprint).startswith('MAZDA') or CP.lateralTuning.which() != 'torque':
    return None
  est, pts = TorqueEstimator(CP), []
  est._on_torque_point = lambda s, l, v: pts.append((s, l, v))
  for t, w, msg in msgs:
    if w == 'carOutput':
      msg = NS(actuatorsOutput=NS(torque=msg.actuatorsOutput.torqueOutputCan / STEER_MAX))
    est.handle_log(t, w, msg)
  return {**meta, 'fp': str(CP.carFingerprint), 'pts': np.array(pts, dtype=np.float32).reshape(-1, 3)}


def learner_fit(steer, lat, rng):
  """torqued_ext's estimator on the points a bin would hold, up to POINTS_PER_BUCKET per steer
  bucket (drawn at random: the tool has no ring-buffer order): (latAccelFactor, friction)."""
  idx = []
  for lo, hi in STEER_BUCKET_BOUNDS:
    b = np.nonzero((steer >= lo) & (steer < hi))[0]
    if not len(b):
      return np.nan, np.nan  # not calculable, as TorqueBuckets.is_calculable
    idx.append(rng.choice(b, POINTS_PER_BUCKET, replace=False) if len(b) > POINTS_PER_BUCKET else b)
  idx = np.concatenate(idx)
  slope, _, friction = fit_torque_points(np.c_[steer[idx], np.ones(len(idx)), lat[idx]])
  return slope, friction


@functools.cache
def build_date(commit):
  out = subprocess.run(['git', '-C', BASEDIR, 'show', '-s', '--format=%cs', commit], capture_output=True, text=True)
  return out.stdout.strip() if out.returncode == 0 else ''


def fit(args):
  segs = [r for r in pickle.load(open(args.points, 'rb')) if r.get('fp') == args.platform and len(r.get('pts', ()))]
  if args.dongle:
    segs = [r for r in segs if r['dongle'] == args.dongle]
  if args.since:
    segs = [r for r in segs if build_date(r['git']) >= args.since]
  if not segs:
    sys.exit('no segments left after the filters')
  P = np.concatenate([r['pts'] for r in segs]).astype(float)
  sid = np.concatenate([np.full(len(r['pts']), i) for i, r in enumerate(segs)])
  steer, lat, v = P[:, 0], P[:, 1], P[:, 2]
  centers = args.centers or get_speed_dep_config()[args.platform]['speed_bp']
  rng = np.random.default_rng(0)
  print(f'{args.platform}: {len(segs)} segments, {len(P)} points')
  print(' center  range m/s      n       LAF     +-      friction  +-')
  for c, (lo, hi) in zip(centers, TorqueEstimatorExt._centers_to_bounds(centers), strict=True):
    m = (v >= lo) & (v < hi)
    laf, fric = learner_fit(steer[m], lat[m], rng)
    rows = np.nonzero(m)[0]
    row_sid = sid[rows]
    ids, boot = np.unique(row_sid), []
    for _ in range(args.boot):
      cnt = np.bincount(rng.choice(ids, len(ids)), minlength=len(segs))
      sel = np.repeat(rows, cnt[row_sid])
      boot.append(learner_fit(steer[sel], lat[sel], rng))
    sd = np.nanstd(np.array(boot), axis=0) if boot else (np.nan, np.nan)
    print(f' {c:5.1f}  {lo:5.2f}-{hi:5.2f}  {m.sum():7d}   {laf:5.2f}  {sd[0]:5.2f}   {fric:.3f}   {sd[1]:.3f}')


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)
  e = sub.add_parser('extract')
  e.add_argument('rlog_list')
  e.add_argument('out')
  f = sub.add_parser('fit')
  f.add_argument('points')
  f.add_argument('--platform', default='MAZDA_CX5_2022')
  f.add_argument('--dongle')
  f.add_argument('--since', help='earliest build date, YYYY-MM-DD')
  f.add_argument('--centers', type=float, nargs='+')
  f.add_argument('--boot', type=int, default=40)
  args = p.parse_args()
  if args.cmd == 'extract':
    paths = [line.strip() for line in open(args.rlog_list) if line.strip()]
    with Pool(os.cpu_count(), initializer=_init) as pool:
      res = [r for r in pool.imap_unordered(_extract, paths, chunksize=4) if r is not None]
    pickle.dump(res, open(args.out, 'wb'))
    print(f"{len(res)} Mazda segments, {sum(len(r.get('pts', ())) for r in res)} points, {sum('err' in r for r in res)} errors")
  else:
    fit(args)


if __name__ == '__main__':
  main()
