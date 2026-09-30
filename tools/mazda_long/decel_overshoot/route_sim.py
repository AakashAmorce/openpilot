"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Closed-loop replay of real roads: the checked-out vision planner and ICBM servo drive the fitted
MRCC plant (fit_plant.py) along roads from the logs. At each simulated position the planner sees
the model path that was recorded nearest that spot (it plans on geometry, so the path indexed on
distance is valid at another speed), and the curvature the car actually drove there is the truth.

Each curve that needs slowing from the set speed is approached from 25 s out at the set speed.
Straight stretches (nothing needing slowing within 300 m) score phantom cost: speed given up.
Leads are not simulated; windows with a close lead in the log are skipped.

Run it on two checkouts to compare them (PYTHONPATH=<worktree> for a baseline).
Each result carries a 0.5 s trace (s, mph, accel, dash mph, planner target mph).
Usage: route_sim.py <model_cache_dir> <table_cache_dir> <plant_fit.pkl> <out.pkl> [workers]
                    [--routes R1,R2] [--every N [--offset K]]
"""
import argparse
import pickle
from multiprocessing import Pool

import numpy as np

import curve_sim as CS
from extract import MPH, load_route, routes
from model_reach import driven, load_chunks

MIN_FRAMES = 601  # a stretch must be over 30 s of frames


class RecordedRoad:
  """A continuous stretch of log: model paths by distance and the driven curvature."""
  def __init__(self, m):
    self.t, self.big = m['t'], m['big']
    self.v, self.s, self.kt = driven(m)
    self.rz, self.vx, self.px, self.py = m['rz'], m['vx'], m['px'], m['py']

  def kappa(self, s):
    return np.interp(s, self.s, self.kt)

  def sm(self, pos):
    i = int(np.clip(np.searchsorted(self.s, pos), 0, len(self.s) - 1))
    msg = CS.messaging.new_message('modelV2')
    p = CS.log.XYZTData.new_message()
    p.x, p.y = [float(x) for x in self.px[i]], [float(y) for y in self.py[i]]
    msg.modelV2.position = p
    vel = CS.log.XYZTData.new_message()
    vel.x = [float(x) for x in self.vx[i]]
    msg.modelV2.velocity = vel
    o = CS.log.XYZTData.new_message()
    o.z = [float(z) for z in self.rz[i]]
    msg.modelV2.orientationRate = o
    msg.modelV2.big = bool(self.big[i])
    cs = CS.messaging.new_message('controlsState')
    cs.controlsState.curvature = float(self.kappa(pos))
    return {'modelV2': msg.modelV2, 'controlsState': cs.controlsState}


def drive(road, plant, s0, s1, set_mph):
  """Simulate the stack from s0 to s1 at set speed; returns position, speed, accel, dash mph and plan target mph traces."""
  st = CS.Stack(plant, set_mph, s0)
  rows = []
  while st.pos < s1:
    rows.append(st.step(road.sm(st.pos)))
  S, V, A, D, T = np.array(rows).T
  return S, V, A, D, np.minimum(T, st.v_set) * MPH


def trace(S, V, A, D, T):
  """Every 10th step (0.5 s) of a drive: position m, speed mph, accel, dash mph, plan target mph."""
  return np.stack([S, V * MPH, A, D, T])[:, ::10].astype(np.float32)


def lead_close(tbl, t0, t1):
  """Share of the logged frames from t0 to t1 with a lead inside 2.5 s (30 m at least)."""
  if tbl is None:
    return 0.
  m = (tbl['t'] >= t0) & (tbl['t'] <= t1)
  if not m.any():
    return 0.
  return float(np.mean((tbl['leadSt'][m] > 0) & (tbl['dRel'][m] < np.maximum(2.5 * tbl['v'][m], 30))))


def run_route(args):
  route, mdl_dir, tbl_dir, plant_pkl = args
  plant = CS.Plant(pickle.load(open(plant_pkl, 'rb')))
  tbl, _ = load_route(route, tbl_dir)
  out = []
  for m in load_chunks(route, mdl_dir):
    if len(m['t']) < MIN_FRAMES:
      continue
    road = RecordedRoad(m)
    v_rec = road.v
    v_allow = np.sqrt(CS.A_LAT / np.maximum(road.kt, 1e-6))
    # apexes: local maxima of driven curvature that need slowing from the speed driven 20 s before
    cand = np.flatnonzero((road.kt > 1e-3) & (np.r_[0, np.diff(road.kt)] > 0) & (np.r_[np.diff(road.kt), 0] <= 0))
    apexes = []
    for i in cand:
      lo = np.searchsorted(road.t, road.t[i] - 20.)
      set_mph = float(np.round(np.max(v_rec[lo:i + 1]) * MPH / 5.) * 5.)
      if set_mph < 25 or v_allow[i] * MPH > set_mph - 3:
        continue
      if apexes and road.s[i] - road.s[apexes[-1][0]] < 150.:
        if road.kt[i] > road.kt[apexes[-1][0]]:
          apexes[-1] = (i, set_mph)
        continue
      apexes.append((i, set_mph))
    for i, set_mph in apexes:
      s_apex = road.s[i]
      s0 = s_apex - 25. * set_mph / MPH
      if s0 < road.s[0] + 10. or s_apex + 150. > road.s[-1] - 350.:
        continue
      if lead_close(tbl, road.t[np.searchsorted(road.s, s0)], road.t[i]) > 0.05:
        continue
      S, V, A, D, T = drive(road, plant, s0, s_apex + 150., set_mph)
      j = int(np.argmin(np.abs(S - s_apex)))
      pre = S < s_apex
      # the recorded path is as long as the recorded car was fast: where a 0.75 brake from the set
      # speed has to start, how far ahead the replayed model path actually reaches
      v_s = set_mph / MPH
      d_need = (v_s ** 2 - min(v_allow[i], v_s) ** 2) / 1.5 + v_s
      k_need = int(np.clip(np.searchsorted(road.s, s_apex - d_need), 0, len(road.s) - 1))
      win = (road.s >= s0) & (road.s <= s_apex)
      rec = {'v_rec_apex': float(v_rec[i] * MPH), 'v_rec_min': float(v_rec[win].min() * MPH) if win.any() else np.nan,
             'd_need': float(d_need), 'reach': float(road.px[k_need, -1])}
      out.append({'route': route, 'kind': 'curve', 'big': int(road.big[i]), 'set_mph': set_mph, 'v_allow': v_allow[i] * MPH,
                      'v_apex': V[j] * MPH, 'lat_apex': V[j] ** 2 * road.kt[i], 'a_min': float(A[pre].min()) if pre.any() else 0.,
                      'v_min': float(V[pre].min()) * MPH if pre.any() else V[j] * MPH,
                      't': float(len(S) * CS.DT), 't_ideal': float((s_apex + 150. - s0) / (set_mph / MPH)),
                      's_apex': float(s_apex), 'trace': trace(S, V, A, D, T), **rec})
    # straight stretches: nothing needing slowing from the set speed within 300 m either side
    need = v_allow * MPH < (np.max(v_rec) * MPH) - 3
    s_need = road.s[need]
    k = 0
    while k < len(road.s):
      s_a = road.s[k]
      set_mph = float(np.round(v_rec[k] * MPH / 5.) * 5.)
      if set_mph >= 40 and s_a + 800. < road.s[-1] and (len(s_need) == 0 or np.min(np.abs(s_need - (s_a + 400.))) > 700.):
        if lead_close(tbl, road.t[k], road.t[min(len(road.t) - 1, np.searchsorted(road.s, s_a + 800.))]) < 0.05:
          S, V, A, D, T = drive(road, plant, s_a, s_a + 800., set_mph)
          out.append({'route': route, 'kind': 'straight', 'big': int(road.big[k]), 'set_mph': set_mph,
                          'lost': float(set_mph - V.min() * MPH), 'a_min': float(A.min()),
                          't': float(len(S) * CS.DT), 't_ideal': 800. / (set_mph / MPH),
                          's_start': float(s_a), 'trace': trace(S, V, A, D, T)})
        k = np.searchsorted(road.s, s_a + 800.)
      else:
        k = np.searchsorted(road.s, s_a + 200.)
  return out


def main():
  ap = argparse.ArgumentParser()
  for a in ('mdl_dir', 'tbl_dir', 'plant_pkl', 'out_pkl'):
    ap.add_argument(a)
  ap.add_argument('workers', nargs='?', type=int, default=8)
  ap.add_argument('--routes', help='comma-separated route names (default: every route in the cache)')
  ap.add_argument('--every', type=int, default=1, help='deterministic subsample: every Nth route, sorted')
  ap.add_argument('--offset', type=int, default=0)
  args = ap.parse_args()
  todo = routes(args.mdl_dir)
  if args.routes:
    todo = [r for r in todo if r in set(args.routes.split(','))]
  todo = todo[args.offset::args.every]
  print(f"{len(todo)} routes, {args.workers} workers, vc={CS.vc.__file__}", flush=True)
  res = []
  with Pool(args.workers, maxtasksperchild=4) as p:
    for r in p.imap_unordered(run_route, [(r, args.mdl_dir, args.tbl_dir, args.plant_pkl) for r in todo]):
      res += r
  with open(args.out_pkl, 'wb') as f:
    pickle.dump(res, f)
  cur = [r for r in res if r['kind'] == 'curve']
  st = [r for r in res if r['kind'] == 'straight']
  print(f"{len(cur)} curve approaches, {len(st)} straight stretches")


if __name__ == '__main__':
  main()
