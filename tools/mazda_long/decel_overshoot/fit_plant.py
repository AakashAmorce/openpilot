"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Fit the MRCC plant: realized aEgo as a delayed first-order lag of f(gap, v), gap = vEgo - dash
(mph), f piecewise-linear in gap with a linear speed term. Least squares per (tau, delay) on
clean stock-ACC stretches (engaged, no pedals, no lead inside 3 s, no openpilot-long route);
decel samples weighted 5x. Writes plant_fit.pkl next to the cache (or --out) for curve_sim.py.

Usage: fit_plant.py [cache_dir] [--fp MAZDA_CX9_2021] [--out plant_fit.pkl]
  --fp keeps routes whose carFingerprint starts with it (MAZDA_ for every Mazda).
"""
import argparse
import glob
import os
import pickle

import numpy as np

from extract import MPH, OUT, load_route, routes

DT = 0.05
GAP_BP = np.array([-12., -6., -3., -1.5, 0., 1.5, 2.5, 4., 6., 8., 10., 14., 20., 30.])


def basis(g):
  g = np.clip(g, GAP_BP[0], GAP_BP[-1])
  B = np.zeros((len(g), len(GAP_BP)))
  k = np.clip(np.searchsorted(GAP_BP, g) - 1, 0, len(GAP_BP) - 2)
  w = (g - GAP_BP[k]) / (GAP_BP[k + 1] - GAP_BP[k])
  B[np.arange(len(g)), k] = 1 - w
  B[np.arange(len(g)), k + 1] = w
  return B


def lag(X, tau):
  alpha, acc, Y = DT / tau, X[0].copy(), np.empty_like(X)
  for i in range(len(X)):
    acc += alpha * (X[i] - acc)
    Y[i] = acc
  return Y


def fingerprint(route, cache):
  z = np.load(glob.glob(os.path.join(cache, route + '--*.npz'))[0])
  return str(z['fp']) if 'fp' in z.files else ''


def stretches(cache, fp=''):
  out = []
  for r in routes(cache):
    if fp and not fingerprint(r, cache).startswith(fp):
      continue
    d, _ = load_route(r, cache)
    if d is None or np.nanmax(d['opLong']) > 0:
      continue
    v, dt = d['v'], np.diff(d['t'])
    ok = (d['eng'] > 0) & (d['crz'] > 0) & (d['accAct'] > 0) & (d['gas'] == 0) & (d['brake'] == 0) & (v > 8)
    ok &= ~((d['leadSt'] > 0) & (d['dRel'] < np.maximum(3.0 * v, 40)))
    ok[1:] &= (dt > 0) & (dt < 0.1)  # never join across a step back in time (loose files keyed by fingerprint)
    gap = (v - d['dash']) * MPH
    i = 0
    while i < len(ok):
      if not ok[i]:
        i += 1
        continue
      j = i
      while j < len(ok) and ok[j]:
        j += 1
      if j - i >= 200 and (gap[i:j] > 2.5).any():
        out.append((r, v[i:j], gap[i:j], d['a'][i:j]))
      i = j
  return out


def design(runs, tau, delay):
  Xs, ys, ws = [], [], []
  for _, v, g, a in runs:
    n = int(delay / DT)
    gd = np.r_[np.full(n, g[0]), g[:len(g) - n]] if n else g
    B = basis(gd)
    X = lag(np.c_[B, B * ((v - 20.) / 10.)[:, None]], tau)
    Xs.append(X[40:])
    ys.append(a[40:])
    ws.append(np.where(g[40:] > 2.0, 5., 1.))
  return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cache', nargs='?', default=OUT)
  ap.add_argument('--fp', default='', help='carFingerprint prefix, e.g. MAZDA_ or MAZDA_CX9_2021')
  ap.add_argument('--out', help='output pickle (default <cache>/plant_fit.pkl)')
  args = ap.parse_args()
  runs = stretches(args.cache, args.fp)
  n = sum(len(r[1]) for r in runs)
  print(f"{len(runs)} stretches from {len({r[0] for r in runs})} routes, {n} samples ({n * DT / 3600:.1f} h)")
  if not runs:
    raise SystemExit("nothing to fit")
  best = None
  for tau in (0.5, 0.8, 1.1):
    for delay in (0.0, 0.3, 0.6):
      X, y, w = design(runs, tau, delay)
      sw = np.sqrt(w)
      coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
      rms = np.sqrt(np.mean((y - X @ coef)[w > 1] ** 2))
      print(f"tau {tau} delay {delay}: rms in the decel regime {rms:.3f} m/s^2")
      if best is None or rms < best[0]:
        best = (rms, tau, delay, coef)
  rms, tau, delay, coef = best
  nb = len(GAP_BP)
  coef[nb - 1], coef[2 * nb - 1] = -1.25, 0.  # the 30 mph node has almost no data
  print(f"best tau {tau} delay {delay}\n gap   30mph  45mph  60mph  70mph")
  for k, g in enumerate(GAP_BP):
    print(f"{g:5.1f} " + " ".join(f"{coef[k] + coef[nb + k] * (v - 20.) / 10.:6.2f}" for v in (13.4, 20.1, 26.8, 31.3)))
  with open(args.out or os.path.join(args.cache, 'plant_fit.pkl'), 'wb') as f:
    pickle.dump({'GB': GAP_BP, 'coef': coef, 'tau': tau, 'delay': delay}, f)


if __name__ == '__main__':
  main()
