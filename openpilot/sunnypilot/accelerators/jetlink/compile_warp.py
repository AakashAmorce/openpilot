#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Build the comma-side warp JIT. Invoked by accelerators/SConscript, once per
camera it builds for, and by nothing at runtime: modeld only loads what this
wrote (warp_cache.load_warp), and a device without one runs the small model.

Mirrors upstream's compile_dm_warp.py. The capture goes through
warp_cache.call_warp, the same call every frame makes, so the JIT's input names
match the ones modeld passes.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

# compile_modeld first: it patches tinygrad's firmware fetch before tinygrad loads
from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, _parse_size, make_warp
from openpilot.sunnypilot.accelerators.jetlink.warp_cache import call_warp
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor


def compile_warp(cam_w: int, cam_h: int, model_w: int, model_h: int, out: Path) -> Path:
  """Build the warp JIT and pickle it to `out`. Holds the GPU while it runs.

  Three runs before pickling: TinyJit captures on the second call, and a
  pickle taken earlier is an empty jit that silently does nothing.
  """
  nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
  warp_jit = TinyJit(make_warp(nv12, model_w, model_h), prune=True)

  # one set of input tensors: TinyJit captures against the buffers it is
  # first handed. Random so nothing constant-folds
  rng = np.random.default_rng(42)
  tfm_npy, big_tfm_npy = np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)
  tfm = Tensor(tfm_npy, device='NPY')
  big_tfm = Tensor(big_tfm_npy, device='NPY')
  frame = Tensor.randint(nv12.size, low=0, high=256, dtype='uint8', device=Device.DEFAULT).realize()
  big_frame = Tensor.randint(nv12.size, low=0, high=256, dtype='uint8', device=Device.DEFAULT).realize()
  for _ in range(3):
    tfm_npy[:] = rng.standard_normal((3, 3)).astype(np.float32)
    big_tfm_npy[:] = rng.standard_normal((3, 3)).astype(np.float32)
    call_warp(warp_jit, tfm, big_tfm, frame, big_frame).realize()
  Device.default.synchronize()

  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  # through a temporary: a compile killed mid-write leaves nothing under the
  # name load_warp opens
  tmp = out.with_suffix('.pkl.tmp')
  with open(tmp, 'wb') as f:
    pickle.dump(warp_jit, f)
  tmp.replace(out)
  return out


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--camera-resolution', type=_parse_size, required=True, help='camera resolution WxH')
  p.add_argument('--model-size', type=_parse_size, required=True, help='model input WxH')
  p.add_argument('--output', required=True)
  args = p.parse_args()

  cam_w, cam_h = args.camera_resolution
  model_w, model_h = args.model_size
  print(f"Compiling jetlink warp for {cam_w}x{cam_h} -> {model_w}x{model_h}...")
  out = compile_warp(cam_w, cam_h, model_w, model_h, Path(args.output))
  print(f"  Saved to {out}")
