"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The selected model's spec, cached in a param.

Reading the shapes and output slices means parsing a 766 MB ONNX. jetlinkd does
that once when it provisions; modeld reads the answer here and never touches
the file.
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
  Params().put(gadget.P_SPEC, spec.to_dict())
