#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Making the Jetson ready to run the model that is picked.

Both borrowers of the link do this. A provisioning run does it offroad, where
it can spend minutes downloading a gigabyte; modeld's join thread does it
onroad, where it cannot download but can upload a file the comma already has
and then wait out the build with the small model driving. What they share
comes first: the model's identity, the upload nobody may make without proving
the hash first, the estimate that turns "build 12%" into a number of minutes,
and the record of what the server ended up with.

The run is the rest of the module, and `python -m` on it is one. It is the
heavy half of jetlink, numpy and the client, which is why it is a run and not
a daemon. The owner (jetlink.comma.owner, which owner.py runs as jetlinkd)
holds the gadget for the whole time the link is enabled and starts a run
when something changes; the run borrows the endpoint files from it exactly
as modeld does (jetlink.comma.lending), so the gadget never leaves the bus and
a parked car keeps one resident jetlink process of about 13 MB instead of a
run's 47. Only the owner ever holds ep0: a run that gets no loan logs it and
exits, and the owner starts another.

The result is cached on the Jetson, recorded in a param and left loaded on the
server. The owner stops a run at the onroad transition with SIGTERM, so every
long wait polls `stop`; the server's build thread carries on regardless and
modeld picks the engine up over its own link.
"""
from __future__ import annotations

import functools
import json
import signal
from pathlib import Path

from jetlink.client import EngineMissing
from jetlink.comma import gadget, lending

from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot import accelerators
from openpilot.sunnypilot.accelerators.jetlink import helpers, spec_cache, warp_cache

# An upload and a build on a busy box. The build itself is 102 to 294 s on this
# hardware; the ceiling is only there so a server that has stopped answering
# does not hold the caller for the rest of the day.
BUILD_TIMEOUT = 1800.0


def identity(entry: dict) -> tuple[str, int]:
  """The picked model's sha256 and byte count.

  From the catalog model's LFS pointer, so the comma can name the model
  without holding or hashing the ONNX. The lookup happens once per model ever
  and is kept in a param; it needs the internet, and that is the only part of
  provisioning that does.
  """
  sha256, nbytes = entry.get('oid'), entry.get('size')
  if sha256 and nbytes:
    return sha256, int(nbytes)
  accelerators.report_progress('connect', 0.0, 'looking up the model')
  return helpers.resolve_pointer(entry['ref'])


# What has already been hashed this run, keyed on the file as it was then. A
# join that fails and tries again would otherwise read a gigabyte off the disk
# every time, on a thread that is now doing it next to a running frame loop.
_hashed: dict[tuple[str, int, int], str] = {}


def verified_upload(model_path: Path | None, sha256: str, nbytes: int) -> Path | None:
  """The file to upload, once its hash is proven to match the registry.

  Uploading under a sha the bytes do not have would leave the Jetson with a
  plan whose name lies about its contents, so the hash happens here, on the
  one path where the bytes go somewhere.
  """
  if model_path is None:
    return None
  try:
    st = model_path.stat()
    if st.st_size != nbytes:
      cloudlog.error("jetlink: %s is %d bytes, the registry says %d; not uploading it",
                     model_path.name, st.st_size, nbytes)
      return None
    key = (str(model_path), st.st_size, st.st_mtime_ns)
    have = _hashed.get(key)
    if have is None:
      from jetlink.spec import sha256_file
      have, _ = sha256_file(str(model_path))
      _hashed[key] = have
    if have != sha256:
      cloudlog.error("jetlink: %s hashes to %s, the registry says %s; not uploading it",
                     model_path.name, have[:16], sha256[:16])
      return None
  except OSError:
    return None
  return model_path


def ensure(client, sha256: str, nbytes: int, model_path: Path | None, *,
           progress=None, should_stop=None, build_timeout: float = BUILD_TIMEOUT):
  """Make the server ready for this model and remember what it answered.

  Asks without the file first: the server answers from the sha alone when it
  already has the model, which is every poll of a parked car and every join of
  a drive. EngineMissing means the Jetson has neither the plan nor the bytes
  and neither has the caller.
  """
  ask = functools.partial(client.ensure_engine, sha256, nbytes, progress=progress,
                          build_timeout=build_timeout, should_stop=should_stop)
  try:
    spec = ask(onnx_path=None)
  except EngineMissing:
    upload = verified_upload(model_path, sha256, nbytes)
    if upload is None:
      raise
    spec = ask(onnx_path=upload)
  spec_cache.store(spec)
  return spec


def estimated_build_seconds(size: int) -> int:
  """Orin Nano Super, TensorRT 10.3: the 766 MB models built in 102 to 166 s,
  the 1.75 GB ones in 230 to 294 s."""
  return int(60 + 130 * size / 1e9)


def _eta(seconds: float) -> str:
  if seconds >= 90:
    return f"about {seconds / 60:.0f} min left"
  return f"about {max(seconds, 1):.0f}s left"


def report_with_eta(stage: str, frac: float, msg: str = '') -> None:
  """Progress, with how long the build still has to run.

  Estimated from the model's size, on measurements of this hardware. It
  belongs here rather than in the UI, which knows nothing about jetlink. Only
  the build is estimated: the upload reports MB of MB and a connect has
  nothing to predict.
  """
  if stage == 'build':
    size = (helpers.selected_model() or {}).get('size')
    if size:
      msg = _eta(estimated_build_seconds(size) * max(0.0, 1.0 - frac))
  accelerators.report_progress(stage, frac, msg)


# -- the provisioning run ----------------------------------------------------

# how long to wait for the Jetson to enumerate before giving up on this run.
# The owner presented the gadget; a box that is asleep answers the bind in
# about 8 s, one that is off never does and the next run will find it
WAKE_TIMEOUT = 20.0


class ProvisioningRun:
  def __init__(self):
    self.client = None
    self.stop = False
    self.fetch_failed = False
    # does the far end suspend when the gadget goes? From the server's hello.
    # The owner needs it to decide whether letting go is worth what it costs,
    # and cannot ask: it never speaks the protocol. None until a hello says:
    # a run with nothing to do never asks
    self.server_sleeps: bool | None = None

  # -- lifecycle ------------------------------------------------------------

  def request_stop(self, *_) -> None:
    self.stop = True

  def close_link(self) -> None:
    """Always go through this: a FunctionFS owner that exits without closing
    can wedge the driver until a reboot."""
    client, self.client = self.client, None
    if client is not None:
      try:
        client.close()
      except Exception:
        cloudlog.exception("jetlink: error closing the link")


  def open_link(self) -> bool:
    """Borrow the link from the owner that started this run. Without a loan
    there is nothing to open: only the owner ever holds ep0."""
    if self.client is not None:
      return True
    try:
      loan = lending.borrow('provision')
      if loan is None:
        cloudlog.error("jetlink: the owner lent no link, nothing to provision over")
        return False
      self.client = helpers.connect(loan, deadline=5.0, name='provision')
      return True
    except Exception:
      cloudlog.exception("jetlink: could not open the link")
      return False

  # -- provisioning ---------------------------------------------------------

  def fetch_model(self):
    """Download the pinned large model, once.

    Minutes on a slow link, so it reports progress and stops when the owner
    stops the run; it is the one call in a run that blocks for long.
    """
    if self.fetch_failed:
      return None
    try:
      path = helpers.fetch_shipped_model(
        progress=lambda frac: accelerators.report_progress('download', frac, 'downloading the large model'),
        should_stop=lambda: self.stop,
      )
    except Exception:
      cloudlog.exception("jetlink: could not fetch the large model")
      accelerators.report_progress('failed', 1.0, 'could not download the large model')
      # one attempt per run; retrying a gigabyte on a loop is worse than staying small
      self.fetch_failed = True
      return None
    return path

  def provision(self) -> bool:
    """Make the Jetson ready for the selected model. Host must be attached.

    The identity comes from the catalog model's LFS pointer (the oid is the
    sha256, size the byte count), so the comma can ask without holding or
    hashing the ONNX.
    The file is only fetched when the server asks for the bytes; the Jetson
    keeps its own copy of every ONNX and never prunes it.
    """
    entry = helpers.selected_model()
    if entry is None:
      # no catalog yet; not an error
      spec_cache.clear_ready()
      accelerators.clear_progress()
      return False
    sha256, nbytes = identity(entry)

    # only needed if the server turns out not to have this model; None is a
    # legitimate state here, see EngineMissing below
    model_path = helpers.shipped_model_path()

    cloudlog.warning("jetlink: provisioning %s (%d MB, sha %s)",
                     entry.get('name', sha256[:16]), nbytes >> 20, sha256[:16])
    accelerators.report_progress('connect', 0.0, 'talking to the accelerator')

    hello = self.client.hello(timeout=10.0)
    self.note_sleep_after(hello)
    cloudlog.warning("jetlink: server %s trt %s", hello.get('device'), hello.get('trt_version'))
    try:
      spec = ensure(self.client, sha256, nbytes, model_path,
                    progress=report_with_eta, should_stop=lambda: self.stop)
    except EngineMissing:
      # the server has nothing to build from, and neither have we: fetch the
      # model and hand it over in this run. The owner lends the link for the
      # whole run, so leaving after the download only put the build a
      # WORKER_BACKOFF later, five minutes of a finished download doing nothing
      if model_path is not None or (model_path := self.fetch_model()) is None:
        raise
      # a download of minutes can outlast the session; a hello starts a new one
      self.client.hello(timeout=10.0)
      spec = ensure(self.client, sha256, nbytes, model_path,
                    progress=report_with_eta, should_stop=lambda: self.stop)

    accelerators.report_progress('ready', 1.0, 'engine ready')
    cloudlog.warning("jetlink: engine ready for %s", spec.sha256[:16])
    return True

  # -- the parked car -------------------------------------------------------

  def note_sleep_after(self, hello: dict) -> None:
    """Record whether the server suspends itself when the gadget goes.

    Letting go is only worth what it costs if the Jetson sleeps when it is
    orphaned. On ignition power it does not, and releasing anyway meant a
    powered, awake box spent the whole parked period unenumerated: the icon
    read DISCONNECTED five seconds later, and every handover after that was an
    unplug the server had to recover from.
    """
    try:
      after = hello.get('sleep_after')
      self.server_sleeps = True if after is None else float(after) > 0
    except (TypeError, ValueError):
      self.server_sleeps = True
    cloudlog.warning("jetlink: the jetson %s when the gadget goes",
                     "sleeps" if self.server_sleeps else "stays up")


  def has_work(self) -> bool:
    """Is there a reason to wake the Jetson? Only things the link can fix count."""
    spec = spec_cache.load()
    if spec is None or not spec_cache.engine_ready_for(spec.sha256):
      return True
    selected = helpers.selected_model()
    return selected is not None and selected.get('oid') != spec.sha256

  def shutdown_jetson(self, reason: str) -> None:
    """hardwared is shutting the comma down and wants the Jetson off too.
    The request file is removed whatever happens: hardwared is waiting on it."""
    cloudlog.warning("jetlink: shutting the jetson down: %s", reason)
    try:
      if not self.open_link():
        raise RuntimeError("could not open the link")
      if not gadget.wait_for_host(WAKE_TIMEOUT, bounce=self.bounce,
                                  should_stop=lambda: self.stop):
        raise TimeoutError(f"no jetson attached within {WAKE_TIMEOUT:.0f} s")
      resp = self.client.shutdown(reason, timeout=5.0)
      cloudlog.warning("jetlink: jetson answered the shutdown request: %s", resp)
    except Exception:
      cloudlog.exception("jetlink: could not shut the jetson down")
    finally:
      gadget.finish_shutdown()

  # -- one run --------------------------------------------------------------

  def bounce(self) -> bool:
    """Ask the owner to bounce the gadget, over the lease. On a phone's cable
    the client's rebind is a no-op: nothing is stuck in an endpoint file."""
    try:
      return bool(self.client.rebind()) if self.client is not None else False
    except Exception:
      cloudlog.exception("jetlink: could not bounce the gadget")
      return False

  def note_state(self, unfinished: bool) -> None:
    """What the owner cannot work out for itself: whether the far end sleeps
    when the gadget goes, and whether this run left anything undone.

    A run that never heard a hello keeps what an earlier one learned. Writing
    the default instead told the owner a phone or an always-on Jetson sleeps
    after every run with nothing to do, and it let the gadget go."""
    sleeps = gadget.far_end_sleeps() if self.server_sleeps is None else self.server_sleeps
    try:
      gadget.STATE.write_text(json.dumps({
        'sleep_after': 1.0 if sleeps else 0.0,
        'unfinished': unfinished,
      }))
    except OSError:
      cloudlog.exception("jetlink: could not record what the owner needs")

  def run(self) -> bool:
    """One provisioning round. True when there is nothing left to do."""
    if not gadget.enabled():
      return True

    reason = gadget.pending_shutdown()
    if reason is not None:
      self.shutdown_jetson(reason)
      return True

    # without a warp for this camera the engine would never run; the offroad
    # alert says so, and waking the Jetson to build one would not change it
    if not warp_cache.built():
      cloudlog.warning("jetlink: no warp built for this camera, nothing to provision for")
      self.note_state(unfinished=False)
      return True

    if not self.has_work():
      cloudlog.warning("jetlink: nothing to provision")
      self.note_state(unfinished=False)
      return True

    finished = False
    try:
      if not self.open_link():
        return False
      if not gadget.wait_for_host(WAKE_TIMEOUT, bounce=self.bounce,
                                  should_stop=lambda: self.stop):
        cloudlog.warning("jetlink: no jetson within %.0f s, leaving it for the next run", WAKE_TIMEOUT)
        return False
      finished = self.provision()
    except Exception:
      cloudlog.exception("jetlink: provisioning failed")
      accelerators.report_progress('failed', 1.0, 'see the log')
    finally:
      self.note_state(unfinished=not finished)
      self.close_link()
    return finished


def main() -> None:
  d = ProvisioningRun()
  # the owner stops this at the onroad transition; closing the link properly is
  # what keeps the driver healthy for modeld
  signal.signal(signal.SIGTERM, d.request_stop)
  signal.signal(signal.SIGINT, d.request_stop)
  d.run()


if __name__ == "__main__":
  main()
