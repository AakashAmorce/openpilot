"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The selected model's spec, and whether its engine is built, in one param.

Reading the shapes and output slices means parsing a 766 MB ONNX. The server
does that when it builds the engine and answers with the spec; provisioning
keeps the answer here, and modeld reads it and never touches the file. The
record also says whether the server has built the engine for the sha it names,
so the spec and the readiness can never name different models. It is not
CLEAR_ON_MANAGER_START: readiness must survive a reboot or every ignition
rebuilds a 160 s engine.
"""
from __future__ import annotations

from jetlink.comma import gadget

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.accelerators.jetlink import helpers


def _raw() -> dict | None:
  # _get tolerates a params library older than these keys
  value = helpers._get(gadget.P_SPEC)
  return value if isinstance(value, dict) else None


def load():
  """The cached ModelSpec, or None if there is not a usable one."""
  from jetlink.spec import ModelSpec
  try:
    d = _raw()
    return ModelSpec.from_dict(d) if d else None
  except Exception:
    cloudlog.exception("jetlink: cached spec is unreadable")
    return None


def store(spec) -> None:
  """The server has built the engine for this spec and answered with it."""
  Params().put(gadget.P_SPEC, {**spec.to_dict(), 'ready': True})


def engine_ready_for(sha256: str | None) -> bool:
  """Has the server built the engine for this model? Params only."""
  d = _raw()
  return bool(sha256) and d is not None and d.get('sha256') == sha256 and d.get('ready') is True


def clear_ready() -> None:
  """The engine is no longer known to be built. The spec stays: it still
  sizes the warp, and the next provisioning run asks again."""
  d = _raw()
  if d is not None and d.get('ready'):
    Params().put(gadget.P_SPEC, {**d, 'ready': False})
