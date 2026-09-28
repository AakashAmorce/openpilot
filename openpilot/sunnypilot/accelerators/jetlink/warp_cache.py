"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The comma-side warp JIT: where it lives, and what loads it.

The warp stays on the comma (see model_state), and upstream's fused run_model
JIT (#38684) has no warp to borrow, so compile_modeld.make_warp is JIT-compiled
as a scons target by compile_warp.py; see accelerators/SConscript. A source
build makes the one for its own camera, a prebuilt release one for every camera
it installs on. Nothing compiles one at runtime: in modeld the ~9 s compile
would hold back the first frame on every ignition, and in a provisioning run,
which only runs offroad, it was lost to ignition. A device without one runs the small model.

load_warp is what stands between a bad pickle and the car.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from openpilot.common.swaglog import cloudlog

# in the source tree, next to upstream's dm_warp_*.pkl and under the *.pkl
# ignore, which the release scripts add past with -f. Not Paths.comma_home():
# on AGNOS that is a tmpfs overlay and the pickle was gone every boot
CACHE_DIR = Path(__file__).resolve().with_name('models')


# what TinyJit records for the keyword call in call_warp: sorted(kwargs)
WARP_INPUT_NAMES = ['big_frame', 'big_tfm', 'frame', 'tfm']


def call_warp(warp, tfm, big_tfm, frame, big_frame):
  """Call a warp JIT. Every caller goes through here, capture included.

  TinyJit names inputs from enumerate(args) plus sorted(kwargs) and refuses a
  call whose names differ from the capture. A positional compile and a keyword
  call raised JitError on the first frame of a drive.
  """
  return warp(tfm=tfm, big_tfm=big_tfm, frame=frame, big_frame=big_frame)


def init_device() -> None:
  """Bring the GPU up now, on the caller's thread.

  tinygrad initialises the device on its first kernel run and spawns a libusb
  event thread doing it. A thread created after config_realtime_process(7, 54)
  inherits SCHED_FIFO 54 and the core-7 pin and preempts the frame loop; that
  cost 5% of frames once. Failure is not a reason to refuse the accelerator.
  """
  try:
    from tinygrad.tensor import Tensor
    Tensor([0.0]).realize()
  except Exception:
    cloudlog.exception("jetlink: could not bring the gpu up before modeld goes realtime")
  # the same trap for tinygrad's compile pool (engine/worker.py): created on
  # the first compile, after modeld goes realtime, its handler threads sat at
  # SCHED_FIFO 54 on core 7. An older tinygrad or PARALLEL=0 is not a failure
  try:
    from tinygrad.engine.worker import get_worker_pool
    get_worker_pool()
  except Exception:
    cloudlog.exception("jetlink: could not start tinygrad's compile pool before modeld goes realtime")


def device_geometry() -> tuple[int, int, int, int]:
  """(cam_w, cam_h, model_w, model_h) for this device.

  The same choice modeld/SConscript makes, and accelerators/SConscript builds
  modeld's cameras, so the warp built is the one modeld asks for. If they
  disagree, load_warp raises and the drive is small-model.
  """
  from openpilot.common.hardware import HARDWARE
  from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
  from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE

  camera = _os_fisheye if HARDWARE.get_device_type() == "mici" else _ar_ox_fisheye
  return camera.width, camera.height, *MEDMODEL_INPUT_SIZE


def warp_path(cam_w: int, cam_h: int, model_w: int, model_h: int) -> Path:
  return CACHE_DIR / f'warp_{cam_w}x{cam_h}_{model_w}x{model_h}_tinygrad.pkl'


def is_cached(cam_w: int, cam_h: int, model_w: int, model_h: int) -> bool:
  """Is there a warp for this geometry?

  Presence only. Staleness is scons' job: the target depends on tinygrad and
  the capture sources. A pickle from an incompatible tinygrad raises in load_warp.
  """
  return warp_path(cam_w, cam_h, model_w, model_h).is_file()


# nothing compiles a warp at runtime and the build runs before manager, so the
# answer holds for the life of the process; the UI asks at 5 Hz
_built: bool | None = None


def built() -> bool:
  """Is there a warp for this device's camera? Without one the link cannot
  run the large model, which the offroad alert says (backend.unavailable_reason)."""
  global _built
  if _built is None:
    _built = is_cached(*device_geometry())
  return _built


def load_warp(cam_w: int, cam_h: int, model_w: int, model_h: int):
  """The cached warp JIT. Raises if it is not there or is stale.

  modeld's big-model load is wrapped in the one-way fallback to the small
  model, and a warp that cannot be trusted must not reach the car.
  """
  if not is_cached(cam_w, cam_h, model_w, model_h):
    raise RuntimeError(f"no warp built for {cam_w}x{cam_h} -> {model_w}x{model_h}; "
                       + "accelerators/SConscript makes it")
  with open(warp_path(cam_w, cam_h, model_w, model_h), 'rb') as f:
    warp = pickle.load(f)

  # a JIT pickled before TinyJit captured loads fine and computes nothing; one
  # captured with a different call convention raises JitError on the first
  # frame of a drive. Both have happened
  captured = getattr(warp, 'captured', None)
  if captured is None:
    raise RuntimeError("cached warp was pickled before it captured; it computes nothing")
  names = list(getattr(captured, 'expected_names', []))
  if names != WARP_INPUT_NAMES:
    raise RuntimeError(f"cached warp expects {names}, call_warp passes {WARP_INPUT_NAMES}")
  return warp


def warm(warp, cam_w: int, cam_h: int) -> None:
  """Run a loaded warp JIT until it is cheap to call.

  Measured: load_warp 0.3 s, the first call 1.9 s, the second 5 ms. Paid on
  modeld's frame loop that was ~26 dropped frames and 16 s of modeldLagging
  after every join.
  """
  import numpy as np
  from tinygrad.device import Device
  from tinygrad.tensor import Tensor

  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

  size = get_nv12_info(cam_w, cam_h)[3]
  frames = [np.zeros(size, dtype=np.uint8) for _ in range(2)]
  blobs = [Tensor.from_blob(f.ctypes.data, (size,), dtype='uint8', device=Device.DEFAULT) for f in frames]
  eye = [np.eye(3, dtype=np.float32) for _ in range(2)]
  tfm, big_tfm = (Tensor(e, device='NPY').realize() for e in eye)
  for _ in range(2):
    call_warp(warp, tfm, big_tfm, blobs[0], blobs[1]).realize()
  Device.default.synchronize()

