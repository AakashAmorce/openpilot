"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

An accelerator that runs the large driving model off the comma: jetlink.

comma's chestnut board is not one of these: modeld, hardwared and the UI handle
it natively and only ask here when no board is fitted. Selection is
`if chestnut_present(): native elif accelerators.ready(): jetlink`.

Every function is a thin call into jetlink.backend and is safe on any device:
feature off costs a param read, package absent answers the negative default.
present(), ready(), progress() and enabled() are polled by the UI at 5 Hz and
must stay cheap.

Both models are the model manager's. The small one runs on whichever modeld
its bundle needs, and the accelerator joins that modeld. The big one is the
big-model slot, the same pick a chestnut runs from (the rule is with
ModelManagerSP._fetch_big_model_files). Nothing here changes which modeld
manager runs.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple

# Nothing heavy at module level. manager, the UI and hardwared all import this
# package, and so does the gadget owner by way of its shim,
# accelerators/jetlink/owner.py, which is only 10 MB while nothing here drags
# numpy, capnp and zmq in behind it. jetlink/tests/test_comma_layer.py holds the
# line. The backend is resolved on the first call and kept, the null one
# included: Python does not cache a failed import, and searching sys.path again
# cost every UI and hardwared call 185 us.
_resolved = None


def _backend():
  global _resolved
  if _resolved is None:
    try:
      from openpilot.sunnypilot.accelerators.jetlink import backend
    except ModuleNotFoundError as e:
      # the backend stands on the comma layer, jetlink.comma, which is in the
      # submodule; without a checkout every answer is the negative default
      if (e.name or '').split('.')[0] != 'jetlink':
        raise
      backend = _NoBackend
    _resolved = backend
  return _resolved


class _NoBackend:
  """The backend's answers when the jetlink package is not checked out."""

  @staticmethod
  def link_mode() -> str:
    return 'off'

  @staticmethod
  def link_transport() -> str:
    return 'USB'

  @staticmethod
  def present() -> bool:
    return False

  @staticmethod
  def ready() -> bool:
    return False

  @staticmethod
  def unavailable_reason() -> str | None:
    return None

  @staticmethod
  def prepare() -> bool:
    return False

  @staticmethod
  def load(cam_w: int, cam_h: int, small) -> None:
    return None

  @staticmethod
  def enabled() -> bool:
    return False

  @staticmethod
  def big_catalog(catalog: dict) -> dict:
    return catalog

  @staticmethod
  def extends_catalog() -> bool:
    return False

  @staticmethod
  def selected_model_name() -> str | None:
    return None

  @staticmethod
  def default_big_model_name() -> str | None:
    return None

  @staticmethod
  def active_model_name() -> str | None:
    return None

  @staticmethod
  def shutdown(reason: str, timeout: float) -> None:
    return None


def _params():
  # helpers keeps one handle per store: constructing a Params costs 144 us on
  # the comma against 110 us for the read itself, and progress() is on the UI's
  # 5 Hz pass
  from openpilot.sunnypilot.accelerators.jetlink import helpers
  return helpers.params()


def _log():
  from openpilot.common.swaglog import cloudlog
  return cloudlog

# written by the provisioning run and the joining state, read by the UI; a param because
# the writer is another process
P_PROGRESS = "AcceleratorProgress"
# Each report is a file write, and the UI reads it at 5 Hz. An upload reports
# once per 4 MB chunk, 440 of them for a 1.7 GB model, and onroad that is IO a
# recording would have to share the disk with.
PROGRESS_MIN_INTERVAL = 0.25
_last_progress = ('', 0.0)


class Daemon(NamedTuple):
  """A process the accelerator needs, for as long as the link is enabled.

  A description, not a PythonProcess: manager imports this package, so this
  package cannot import manager. process_config owns the onroad gating.
  """
  name: str
  module: str
  should_run: Callable[..., bool]


class Accelerator(NamedTuple):
  """The accelerator joined to modeld, from load().

  Everything modeld keeps of it: the model to run, and what goes where a
  chestnut puts its ChestnutState.
  """
  model: Any
  status: Any


def installed() -> bool:
  """Is the backend's package checked out? What makes the link worth offering in the UI."""
  return _backend() is not _NoBackend


def present() -> bool:
  """Is a Jetson attached, or asleep and known to be there? USB-independent."""
  return _backend().present()


def ready() -> bool:
  """Can the large model run right now? Params only, what the UI calls 'compiled'."""
  return _backend().ready()


def unavailable_reason() -> str | None:
  """Why the link the user asked for cannot run, for the offroad alert. None unless enabled."""
  return _backend().unavailable_reason()


def prepare() -> bool:
  """Will the accelerator join this modeld? enabled(), then the process-wide setup
  modeld must do before going realtime, which is also a last veto. modeld only."""
  return _backend().prepare()


def load(cam_w: int, cam_h: int, small) -> Accelerator | None:
  """Join the accelerator to modeld, once the camera is up and `small` is built.

  None unless prepare() said yes in this process. The model is the joining
  one, the small model driving now and the Jetson swapped in later; if that
  cannot be built, it is `small`, and the failure is logged. modeld only.
  """
  return _backend().load(cam_w, cam_h, small)


def enabled() -> bool:
  """Has the user turned the link on, with no chestnut fitted? Configuration only,
  never link state or ready(). A chestnut runs the big model natively and the link
  stays off beside it, so jetlinkd never takes the USB controller from it."""
  return _backend().enabled()


# Accelerator Link, stored in LINK_PARAM as an index into LINK_MODES: off, a
# Jetson, a Linux PC or a Mac on USB, or an iPhone (iOS). The panels write the param; the
# gadget follows once the car is parked. jetlink.comma.gadget's LINK_MODES and
# P_LINK, written out so the panels can build the setting at import without a
# jetlink checkout; test_comma_layer holds the two equal
LINK_MODES = ('off', 'usb', 'ios')
LINK_PARAM = 'JetlinkLink'


def link_mode() -> str:
  """The Accelerator Link setting, one of LINK_MODES. Read off the param files,
  so it never raises."""
  return _backend().link_mode()


def link_transport() -> str:
  """What carries the link now, for the panels: USB, or iOS over USB. Never raises."""
  return _backend().link_transport()


def big_catalog(catalog: dict) -> dict:
  """The big-model catalog the model manager fetched, with the models newer
  catalogs list folded in."""
  return _backend().big_catalog(catalog)


def extends_catalog() -> bool:
  """Should the big-model catalog carry them? Hardware, not the link toggle."""
  return _backend().extends_catalog()


def selected_model_name() -> str | None:
  """The big model the accelerator will run: the model manager's big-model pick, or the default."""
  return _backend().selected_model_name()


def active_model_name() -> str | None:
  """selected_model_name() once the accelerator can run it, else None."""
  return _backend().active_model_name()


def default_big_model_name() -> str | None:
  """The big model the accelerator runs with no pick, named as the chestnut's
  default is, without a build date. None when it has no catalog to name it
  from. Per frame from the UI: off a cache, never the network."""
  return _backend().default_big_model_name()


def daemons() -> list[Daemon]:
  """Processes for process_config to build.

  One, and it runs onroad too: it owns the USB gadget for as long as the link
  is enabled, and a gadget whose owner exits is an unplug the far end has to
  recover from. It is about 13 MB because it does nothing else; the heavy half
  is a run it starts when there is provisioning to do, and that exits.
  """
  return [Daemon("jetlinkd", "openpilot.sunnypilot.accelerators.jetlink.owner",
                 lambda started, params, CP: _backend().enabled())]


def progress() -> dict | None:
  """{stage, frac, msg} while something provisions, else None.

  Read from the UI's param thread, so nothing may escape, UnknownKeyName included.
  """
  try:
    value = _params().get(P_PROGRESS)
  except Exception:
    return None
  return value if isinstance(value, dict) else None


def report_progress(stage: str, frac: float, msg: str = '') -> None:
  """Never raises: called from except handlers.

  Held to 4 Hz within a stage. The end of one always goes through, so the last
  thing the panel is told is never dropped.
  """
  global _last_progress
  last_stage, last_at = _last_progress
  now = time.monotonic()
  if frac < 1.0 and stage == last_stage and now - last_at < PROGRESS_MIN_INTERVAL:
    return
  _last_progress = (stage, now)
  try:
    _params().put(P_PROGRESS, {'stage': stage, 'frac': round(frac, 4), 'msg': msg})
  except Exception:
    _log().exception("accelerators: could not report progress")


def clear_progress() -> None:
  try:
    _params().remove(P_PROGRESS)
  except Exception:
    _log().exception("accelerators: could not clear progress")


def shutdown(reason: str = '', timeout: float = 25.0) -> None:
  """The device is powering off for good. Tell the Jetson, within `timeout`.

  hardwared calls this before DoShutdown and publishes no deviceState until it
  returns, so the request runs on a thread and is abandoned at the deadline.
  """
  if not _backend().enabled():
    return

  def request():
    try:
      _backend().shutdown(reason, timeout)
    except Exception:
      _log().exception("accelerators: shutdown request failed")

  t = threading.Thread(target=request, name='accelerator-shutdown', daemon=True)
  t.start()
  t.join(timeout)
  if t.is_alive():
    _log().warning("accelerators: shutdown request still pending after %.0f s, going on without it", timeout)
