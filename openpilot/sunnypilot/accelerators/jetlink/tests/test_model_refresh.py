"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

sunnypilot's "refresh model list" on a comma with jetlink and no chestnut.

The panels zero the model manager's two sync keys and spin until both are
stamped again (selfdrive/ui/sunnypilot/model_info.py). The manager's next tick
refetches both catalogs; the big-model one is extended with the newer
catalogs (accelerators.big_catalog) and its slot is the accelerator's pick,
which the owner watches along with JetlinkSpec by mtime (jetlink.comma.owner).
Each test runs the manager's real loop for one tick against its own params.
"""
from __future__ import annotations

import copy
import os
import unittest
from unittest import mock

import requests

from jetlink.comma import gadget
from jetlink.comma.owner import WATCHED
from jetlink.spec import ModelSpec

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.accelerators.jetlink import backend, helpers as jetlink_helpers, spec_cache
from openpilot.sunnypilot.models import helpers as model_helpers, manager as manager_module
from openpilot.sunnypilot.models.fetcher import ModelFetcher, ModelParser
from openpilot.sunnypilot.models.helpers import ACTIVE_BUNDLE_KEYS, REQUIRED_JSON_VERSION, resolve_bundle_by_ref

V3, V4 = '3' * 40, '4' * 40
V4_OID = 'c' * 64
SLOT = ACTIVE_BUNDLE_KEYS["chestnut"]


def bundle(ref: str, index: int, name: str, selector: int = REQUIRED_JSON_VERSION, big: bool = True) -> dict:
  return {'ref': ref, 'index': index, 'minimum_selector_version': str(selector), 'is_big': big, 'is_20hz': True,
          'display_name': name, 'short_name': name[:4], 'generation': '12', 'environment': 'development',
          'runner': 'tinygrad', 'build_time': '2026-09-25T00:00:00Z', 'overrides': {'folder': 'Master Models'},
          'models': [{'type': 'chunked', 'artifact': {'file_name': f'{name}.pkl', 'download_uri': {'url': 'x', 'sha256': 'y'}}}]}


SMALL = {'bundles': [bundle('1' * 40, 30, 'North Dakota', big=False)]}
# sunnypilot's big-model catalog as the manager fetches it, and a model only a newer one lists
BIG = {'bundles': [bundle(V3, 12, 'Cinque Terre V3')]}
NEWER = {'bundles': [bundle(V4, 13, 'Cinque Terre V4', selector=REQUIRED_JSON_VERSION + 1)]}


class _Tick(BaseException):
  """Ends main_thread after one pass. An Exception would be the loop's to swallow."""


class RefreshTest(OpenpilotTestCase):
  CHESTNUT = False

  def setUp(self):
    super().setUp()
    self.params = Params()
    self.offline = False
    self.big = BIG
    self.newer: dict | Exception = NEWER
    for patcher in (
      mock.patch('openpilot.sunnypilot.models.fetcher.requests.get', side_effect=self.serve),
      mock.patch('jetlink.registry.catalog.fetch_catalogs', side_effect=self.probe),
      mock.patch.object(backend, '_chestnut_fitted', return_value=self.CHESTNUT),
      mock.patch.object(model_helpers, 'chestnut_present', return_value=self.CHESTNUT),
    ):
      patcher.start()
      self.addCleanup(patcher.stop)
    self.new_process()
    for module, name in ((jetlink_helpers, '_index_cache'), (jetlink_helpers, '_slot_cache')):
      setattr(module, name, None)
      self.addCleanup(setattr, module, name, None)
    self.addCleanup(model_helpers._LAST_VALIDATED_RAW.clear)

  def serve(self, url, timeout=None):
    if self.offline:
      raise requests.exceptions.ConnectionError("offline")
    response = mock.MagicMock(status_code=200)
    response.json.return_value = copy.deepcopy({ModelFetcher.MODEL_URL: SMALL, ModelFetcher.MODEL_URL_CHESTNUT: self.big}[url])
    return response

  def probe(self):
    if isinstance(self.newer, Exception):
      raise self.newer
    return copy.deepcopy(self.newer)

  def new_process(self) -> None:
    """The manager runs offroad only, so each parked period is a fresh process."""
    model_helpers._LAST_VALIDATED_RAW.clear()
    with mock.patch.object(manager_module.messaging, 'PubMaster'), mock.patch.object(manager_module.messaging, 'SubMaster'):
      self.manager = manager_module.ModelManagerSP()
    self.manager.sm.__getitem__.return_value.chestnutPresent = self.CHESTNUT

  def tick(self) -> None:
    with mock.patch.object(manager_module, 'Ratekeeper') as rk, \
         mock.patch.object(manager_module.cloudlog, 'exception', wraps=manager_module.cloudlog.exception) as logged:
      rk.return_value.keep_time.side_effect = _Tick
      with self.assertRaises(_Tick):
        self.manager.main_thread()
    # the loop swallows what the tick raised, and says so
    self.assertFalse([c for c in logged.call_args_list if str(c.args[0]).startswith("Error in main thread")])
    jetlink_helpers._index_cache = jetlink_helpers._slot_cache = None

  def refresh(self) -> None:
    """What the panels' button does (model_info.refresh_model_list), then the manager's next tick."""
    for cache in self.manager.model_fetcher.model_caches.values():
      self.params.put(cache._LAST_SYNC_KEY, 0, block=True)
    self.tick()

  def stamps(self) -> list:
    return [self.params.get(cache._LAST_SYNC_KEY) for cache in self.manager.model_fetcher.model_caches.values()]

  def pick(self, ref: str) -> None:
    """Pick a big model as the manager stores it without a chestnut: the catalog's
    entry, files not fetched (ModelManagerSP._download_bundle)."""
    bundles = ModelParser.parse_models(self.params.get("ModelManager_ModelsCache_Chestnut"))
    self.params.put(SLOT, resolve_bundle_by_ref(ref, {"chestnut": bundles})[0].to_dict(), block=True)

  def marks(self) -> dict:
    """What the owner stats to decide a provisioning run is due (Owner.marks)."""
    out = {}
    for key in WATCHED:
      try:
        out[key] = os.stat(gadget.params_dir() / key).st_mtime_ns
      except OSError:
        out[key] = 0
    return out


class TestRefreshOnAJetlinkDevice(RefreshTest):
  def setUp(self):
    super().setUp()
    self.params.put(gadget.P_LINK, gadget.LINK_MODES.index('usb'), block=True)
    self.assertTrue(gadget.enabled())
    self.tick()   # the catalogs as the manager had them before anyone pressed refresh
    self.pick(V4)
    # the pick's pointer is resolved and the Jetson has built its engine
    self.params.put(jetlink_helpers.P_POINTERS, {V4: {'oid': V4_OID, 'size': 766_000_000}}, block=True)
    spec = ModelSpec(sha256=V4_OID, nbytes=766_000_000, frame_skip=4, input_shapes={'features_buffer': (1, 24, 512)},
                     output_shapes={'outputs': (1, 16)}, output_slices={'plan': slice(0, 16)}, checkpoint=None)
    self.params.put(gadget.P_SPEC, {**spec.to_dict(), 'ready': True}, block=True)
    self.tick()
    self.slot, self.spec, self.before = self.params.get(SLOT), self.params.get(gadget.P_SPEC), self.marks()

  def assert_pick_and_readiness_kept(self):
    self.assertEqual(self.params.get(SLOT), self.slot)
    self.assertEqual(self.params.get(gadget.P_SPEC), self.spec)
    model = jetlink_helpers.selected_model()
    self.assertEqual((model['ref'], model['oid']), (V4, V4_OID))
    self.assertTrue(spec_cache.engine_ready_for(model['oid']))
    self.assertEqual(self.marks(), self.before, "the owner would start a provisioning run")

  def test_a_refresh_stamps_both_catalogs_so_the_spinner_ends(self):
    self.refresh()
    self.assertTrue(all(self.stamps()), self.stamps())
    cached = self.params.get("ModelManager_ModelsCache_Chestnut")
    self.assertIs(cached[ModelFetcher.EXTENDED_KEY], True)
    self.assertEqual([b['ref'] for b in cached['bundles']], [V3, V4])
    self.assert_pick_and_readiness_kept()

  def test_a_failed_fetch_leaves_the_pick_and_readiness_alone(self):
    self.offline = True
    self.refresh()
    # nothing restamped: the panels' spinner gives up at its timeout, as upstream
    self.assertEqual(self.stamps(), [0, 0])
    self.assertEqual(self.params.get("ModelManager_ModelsCache_Chestnut")['bundles'][1]['ref'], V4)
    self.assert_pick_and_readiness_kept()

  def test_the_pick_survives_a_catalog_that_no_longer_lists_it(self):
    # the pick is the slot and its pointer, not a catalog row (helpers.selected_model)
    self.newer = {'bundles': []}
    self.refresh()
    self.assertTrue(all(self.stamps()))
    self.assertNotIn(V4, [b['ref'] for b in self.params.get("ModelManager_ModelsCache_Chestnut")['bundles']])
    self.assert_pick_and_readiness_kept()

  def test_a_failed_probe_for_the_newer_catalogs_keeps_the_pick_listed(self):
    # sunnypilot's catalog came, jetlink's did not: the models the last probe found
    # stay listed, or the manager's next start would drop the pick and the owner
    # would build the default model's engine in its place
    self.newer = OSError("offline")
    self.refresh()
    self.assertTrue(all(self.stamps()))
    self.assertIn(V4, [b['ref'] for b in self.params.get("ModelManager_ModelsCache_Chestnut")['bundles']])
    self.new_process()
    self.tick()
    self.assert_pick_and_readiness_kept()

  def test_with_the_link_off_a_refresh_is_the_same(self):
    # the catalog follows the hardware, not the toggle, so a pick made with the
    # link off is still listed when it is turned on
    self.params.put(gadget.P_LINK, gadget.LINK_MODES.index('off'), block=True)
    self.assertFalse(gadget.enabled())
    self.before = self.marks()
    self.refresh()
    self.assertTrue(all(self.stamps()))
    self.assertIs(self.params.get("ModelManager_ModelsCache_Chestnut")[ModelFetcher.EXTENDED_KEY], True)
    self.assert_pick_and_readiness_kept()


class TestRefreshBesideAChestnut(RefreshTest):
  """A chestnut runs sunnypilot's catalog as fetched; jetlink adds nothing to the refresh."""
  CHESTNUT = True

  def test_the_catalog_is_the_one_fetched(self):
    self.tick()
    self.refresh()
    self.assertTrue(all(self.stamps()))
    self.assertEqual(self.params.get("ModelManager_ModelsCache_Chestnut"), {**BIG, ModelFetcher.EXTENDED_KEY: False})


if __name__ == '__main__':
  unittest.main()
