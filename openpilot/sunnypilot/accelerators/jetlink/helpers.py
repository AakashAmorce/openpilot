"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Where the model is, whether the Jetson is attached, and how far along it is.

The large model is the model manager's big-model pick, the same slot a
chestnut runs from (see ModelManagerSP._fetch_big_model_files). Its bundles
are tinygrad pkls for a GPU the Jetson does not have, but each one names the
comma commit it was compiled from, and that commit's ONNX in comma's LFS is
what the Jetson runs. The catalog says what exists, the slot says which one,
the pointer at the commit says which bytes. A commit that ships a precompiled
pkl instead names its export, and jetlink's registry follows that to comma's
model repo; see jetlink.registry.lfs.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from jetlink.comma import gadget
from jetlink.registry.catalog import DEFAULT_BIG_MODEL_REF

from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


# One handle per params store. Constructing a Params costs 144 us on the comma
# against 110 us for the read itself, so a fresh one per read more than doubles
# every param this module touches, and the UI reads several five times a
# second. Keyed on the prefix because a test or a
# bench runs under its own store and must not be handed the device's.
# jetlink.comma logs through a plain logger so the owner needs no swaglog; every
# process that imports helpers is heavy already and wants its lines in the drive
gadget.set_logger(cloudlog)

_params: dict[str, Params] = {}


def params() -> Params:
  prefix = os.environ.get('OPENPILOT_PREFIX', '')
  store = _params.get(prefix)
  if store is None:
    store = _params[prefix] = Params()
  return store


def _get(key: str, default=None):
  """Read a param, tolerating a params library that predates the key.

  Called from hardwared and the UI's param thread, so UnknownKeyName here
  would take down a process that has nothing to do with jetlink.
  """
  try:
    return params().get(key)
  except Exception:
    return default


def await_shutdown(timeout: float) -> bool:
  """Wait for the owner's run to take the request. False if nobody did in time."""
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if not gadget.SHUTDOWN_REQUEST.exists():
      return True
    time.sleep(0.25)
  gadget.finish_shutdown()
  return False


# jetlinkd, the owner, holds the gadget for as long as the link is enabled, so
# presence no longer blinks at every handover. What is left to bridge is a USB3
# link recovery passing through "addressed", and a bounce made on purpose when
# a host will not enumerate (gadget.wait_for_host)
PRESENCE_HOLD = 5.0
_last_configured = 0.0


def gadget_present() -> bool:
  """Is a Jetson actually on the other end right now?

  True once something holds the gadget open and a host has configured us,
  held for PRESENCE_HOLD after that stops. A phone on the cable is a host on
  the gadget like any other.
  """
  global _last_configured
  if gadget.dormant():
    # no enumeration during suspend; the CC line still tells a sleeping host from an unplugged one
    return gadget.port_has_host()
  now = time.monotonic()
  if gadget.host_attached():
    _last_configured = now
    return True
  return now - _last_configured < PRESENCE_HOLD


def connect(loan, deadline: float | None = None, name: str | None = None):
  """Open the link over what jetlinkd lent (see jetlink.comma.lending).

  Only the owner ever holds ep0, and it decided which link this is when it
  lent it: a phone's dial, which the link rides on, or the endpoint files,
  which are all this end opens. `deadline` is per frame and defaults to
  FRAME_TIMEOUT: modeld blocks on a frame the way it blocks on a chestnut.
  `name` is what the server logs this connection as; two comma processes share
  one gadget and the Jetson's journal has no clock to tell them apart by.
  """
  from jetlink.client import FRAME_TIMEOUT, JetlinkClient
  deadline = FRAME_TIMEOUT if deadline is None else deadline
  if loan.sock is not None:
    cloudlog.warning("jetlink: connecting over the phone's dial (%s)", gadget.link_peer())
    return JetlinkClient.open_socket(loan.sock, deadline=deadline, name=name)
  return JetlinkClient.open_borrowed_ffs(loan.mount, loan.udc, bounce=loan.bounce,
                                         deadline=deadline, name=name)


# -- the model ------------------------------------------------------------

P_POINTERS = "JetlinkModelPointers"  # ref -> {oid, size}; a commit's tree never changes
# the model manager's copy of sunnypilot's big-model catalog; the key is
# models.fetcher.ModelFetcher.MODEL_SOURCES['chestnut']'s
CATALOG_PARAM = "ModelManager_ModelsCache_Chestnut"

POINTER_TIMEOUT = 10.0
_REF = re.compile(r'[0-9a-f]{40}')
# the build date a catalog name ends in, " (September 17, 2026)"; any other parenthesis stays
_TRAILING_DATE = re.compile(r' \([A-Za-z]+ \d{1,2}, \d{4}\)$')
# the index and the slot are JSON params, and the UI names the active model
# every frame; the status line can lag a new pick by this long
INDEX_TTL = 2.0
_index_cache: tuple[float, list[dict], dict[str, dict]] | None = None
_slot_cache: tuple[float, dict | None] | None = None


def catalog() -> list[dict]:
  """sunnypilot's big-model bundles as {name, ref}, newest first, as the model
  manager's own picker lists them.

  From the cached JSON rather than the parsed bundles: parsing builds capnp
  objects and writes chunk manifests, for two fields. Read from the UI's
  param thread, so nothing escapes.
  """
  try:
    from openpilot.sunnypilot.models.helpers import REQUIRED_JSON_VERSION
    bundles = (params().get(CATALOG_PARAM) or {}).get('bundles', [])
    found = [b for b in bundles if _REF.fullmatch(str(b.get('ref')))
             and int(b.get('minimum_selector_version', 0)) == REQUIRED_JSON_VERSION]
  except Exception:
    cloudlog.exception("jetlink: could not read the big-model catalog")
    return []
  found.sort(key=lambda b: int(b.get('index', 0)), reverse=True)
  return [{'name': str(b.get('display_name') or b['ref'][:10]), 'ref': b['ref']} for b in found]


def pointers() -> dict[str, dict]:
  value = _get(P_POINTERS)
  return value if isinstance(value, dict) else {}


def fetch_pointer(ref: str) -> tuple[str, int]:
  """The oid and size of the ONNX a comma commit names, in its tree or, for a
  precompiled-pkl commit, in comma's model repo. The Jetson's registry does
  the same lookup, so the two ends agree on every model's identity."""
  from jetlink.registry.lfs import fetch_pointer as registry_fetch_pointer
  pointer = registry_fetch_pointer(ref, timeout=POINTER_TIMEOUT)
  return pointer.oid, pointer.size


def resolve_pointer(ref: str) -> tuple[str, int]:
  """The oid and size behind a catalog model, fetched the first time and kept for good."""
  global _index_cache
  known = pointers()
  if ref in known:
    return known[ref]['oid'], int(known[ref]['size'])
  oid, size = fetch_pointer(ref)
  known[ref] = {'oid': oid, 'size': size}
  # blocking: the next lookup reads this back, and a put still in flight would be lost under it
  params().put(P_POINTERS, known, block=True)
  _index_cache = None
  cloudlog.warning("jetlink: %s is %s, %d MB", ref[:10], oid[:16], size >> 20)
  return oid, size


def _row(name: str, ref: str, known: dict[str, dict]) -> dict:
  p = known.get(ref) or {}
  return {'name': name, 'ref': ref, 'oid': p.get('oid'), 'size': int(p['size']) if p.get('size') else None}


def _index() -> tuple[list[dict], dict[str, dict]]:
  """The catalog's rows and the pointers behind them, read together once per INDEX_TTL."""
  global _index_cache
  now = time.monotonic()
  if _index_cache is None or now - _index_cache[0] >= INDEX_TTL:
    known = pointers()
    _index_cache = (now, [_row(b['name'], b['ref'], known) for b in catalog()], known)
  return _index_cache[1], _index_cache[2]


def model_index() -> list[dict]:
  """Every catalog model, {name, ref, oid, size}. oid and size are None until
  the model has been selected and resolved. No network."""
  return _index()[0]


def selected_slot() -> dict | None:
  """The big-model slot's pick as {name, ref}, from the raw param rather than a
  parsed bundle, and memoised: the UI asks every frame."""
  global _slot_cache
  now = time.monotonic()
  if _slot_cache is not None and now - _slot_cache[0] < INDEX_TTL:
    return _slot_cache[1]
  from openpilot.sunnypilot.models.helpers import ACTIVE_BUNDLE_KEYS
  slot = _get(ACTIVE_BUNDLE_KEYS["chestnut"])
  ref = slot.get('ref') if isinstance(slot, dict) else None
  pick = None
  if isinstance(ref, str) and ref:
    pick = {'name': str(slot.get('displayName') or ref[:10]), 'ref': ref}
  _slot_cache = (now, pick)
  return pick


def default_model() -> dict | None:
  """What runs with no pick: jetlink's default big model, else the newest the
  catalog lists. Not the chestnut's default, which is the model in the tree."""
  models = model_index()
  return next((m for m in models if m['ref'] == DEFAULT_BIG_MODEL_REF), models[0] if models else None)


def default_model_name() -> str | None:
  """default_model()'s name without the catalog's trailing build date, the
  form the chestnut's DEFAULT_BIG_MODEL has. Per frame from the UI, off the
  index's cache."""
  model = default_model()
  return _TRAILING_DATE.sub('', model['name']) if model else None


def selected_model() -> dict | None:
  """The slot's pick, listed in the catalog or not: a ref is enough to find its
  pointer. With no pick, default_model()."""
  if (pick := selected_slot()) is not None:
    return _row(pick['name'], pick['ref'], _index()[1])
  return default_model()


def model_dir() -> Path:
  """Ours, under the model manager's root: its cache clear removes every file
  it does not recognise and leaves directories alone."""
  return Path(Paths.model_root()) / 'jetlink'


def model_file_name(model: dict) -> str:
  """One file per model, so switching back does not re-download."""
  return f"{model['oid'][:16]}.onnx"


def shipped_model_path() -> Path | None:
  """The chosen large model, if it has been fetched.

  Keyed on the oid, not the in-tree pointer, which moves with upstream syncs.
  Size is the cheap check that the file is the one we mean.
  """
  model = selected_model()
  if model is None or not model['oid']:
    return None
  path = model_dir() / model_file_name(model)
  if path.is_file() and path.stat().st_size == model['size']:
    return path
  return None


def lfs_endpoints() -> list[str]:
  """Where a model's bytes are asked for, nearest first: the LFS server this
  checkout's .lfsconfig names (a release ships none), then the ones the Jetson
  asks."""
  from jetlink.registry.lfs import LFS_ENDPOINTS
  out = []
  try:
    for line in (Path(BASEDIR) / '.lfsconfig').read_text().splitlines():
      key, sep, value = line.strip().partition('=')
      if sep and key.strip() == 'url' and value.strip():
        out.append(value.strip().removesuffix('/'))
        break
  except OSError:
    pass
  return out + [e for e in LFS_ENDPOINTS if e not in out]


def fetch_shipped_model(progress=None, should_stop=None) -> Path | None:
  """Download the chosen large model if it is not here yet. None when nothing is chosen.

  jetlink's registry streams it to a .part file and hashes it on the way, so
  only the whole model ever takes the name.
  """
  from jetlink.registry.catalog import NetworkError
  from jetlink.registry.lfs import Pointer, lfs_download, lfs_resolve
  model = selected_model()
  if model is None or not model['oid']:
    return None
  dest = model_dir() / model_file_name(model)
  if dest.is_file() and dest.stat().st_size == model['size']:
    return dest
  pointer = Pointer(model['oid'], int(model['size']))
  for endpoint in lfs_endpoints():
    href = lfs_resolve(endpoint, pointer)
    if href is None:
      continue
    cloudlog.warning("jetlink: fetching the large model (%d MB) from %s", pointer.size >> 20, endpoint)
    return lfs_download(href, pointer, dest, progress=progress, should_stop=should_stop)
  raise NetworkError(f"no LFS server has {pointer.oid[:16]}")
