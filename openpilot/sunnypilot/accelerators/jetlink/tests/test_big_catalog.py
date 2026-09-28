"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

A big model sunnypilot publishes after this build is still pickable with a
Jetson: the model manager's big-model catalog has the newer catalogs' models
folded in, as entries the Jetson can run and a chestnut never downloads.
"""
from __future__ import annotations

import unittest
from unittest import mock

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.accelerators.jetlink import backend
from openpilot.sunnypilot.models.fetcher import ModelFetcher, ModelParser
from openpilot.sunnypilot.models.helpers import _bundle_needs_reset, resolve_bundle_by_ref

OLD, NEW = 'a' * 40, 'e' * 40


def bundle(ref: str, index: int, selector: str, name: str) -> dict:
  return {'ref': ref, 'index': index, 'minimum_selector_version': selector, 'is_big': True, 'is_20hz': True,
          'display_name': name, 'short_name': name[:4], 'generation': '12', 'environment': 'development',
          'runner': 'tinygrad', 'build_time': '2026-09-25T00:00:00Z', 'overrides': {'folder': 'Master Models'},
          'models': [{'type': 'chunked', 'artifact': {'file_name': f'{name}.pkl', 'download_uri': {'url': 'x', 'sha256': 'y'}}}]}


PINNED = {'tinygrad_ref': 'pinned', 'bundles': [bundle(OLD, 12, '19', 'Cinque Terre V3')]}
NEWER = {'tinygrad_ref': 'next', 'bundles': [bundle(OLD, 12, '20', 'Cinque Terre V3'), bundle(NEW, 13, '20', 'Cinque Terre V4')]}


class TestBigCatalog(OpenpilotTestCase):
  def merged(self, newer=NEWER):
    probe_result = {'side_effect': newer} if isinstance(newer, Exception) else {'return_value': newer}
    with mock.patch('jetlink.registry.catalog.fetch_catalogs', **probe_result) as probe:
      out = backend.big_catalog(PINNED)
    return out, probe

  def test_a_model_only_a_newer_catalog_lists_can_be_picked(self):
    out, probe = self.merged()
    probe.assert_called_once_with()
    bundles = ModelParser.parse_models(out)
    self.assertEqual([b.ref for b in bundles], [OLD, NEW])
    picked, source = resolve_bundle_by_ref(NEW, {'chestnut': bundles})
    self.assertEqual((picked.displayName, source), ('Cinque Terre V4', 'chestnut'))
    # nothing for a chestnut to fetch
    self.assertEqual(list(picked.models), [])
    # the slot written from it survives the model manager's validation
    self.assertFalse(_bundle_needs_reset(picked, bundles, check_files=False))

  def test_a_model_the_pinned_catalog_has_keeps_its_build(self):
    out, _ = self.merged()
    self.assertIs(out['bundles'][0], PINNED['bundles'][0])
    self.assertEqual(out['tinygrad_ref'], 'pinned')

  def test_nothing_newer_or_a_failed_probe_leaves_it_alone(self):
    self.assertIs(self.merged(newer=PINNED)[0], PINNED)
    self.assertIs(self.merged(newer=OSError('offline'))[0], PINNED)

  def test_a_failed_probe_keeps_what_the_last_one_found(self):
    # the model manager's cached copy has the newer model as merge_catalogs made
    # it; a probe that fails now must not drop a pick that is only listed there
    from openpilot.common.params import Params
    from openpilot.sunnypilot.accelerators.jetlink import helpers
    last, _ = self.merged()
    Params().put(helpers.CATALOG_PARAM, {**last, ModelFetcher.EXTENDED_KEY: True}, block=True)
    out, _ = self.merged(newer=OSError('offline'))
    self.assertEqual([b['ref'] for b in out['bundles']], [OLD, NEW])
    self.assertEqual(out['bundles'][1], last['bundles'][1])
    # sunnypilot's own entries come from the fetch, never the cache
    self.assertIs(out['bundles'][0], PINNED['bundles'][0])
    self.assertEqual(ModelParser.parse_models(out)[1].ref, NEW)


class TestExtendsCatalog(OpenpilotTestCase):
  """Hardware, not the link toggle: the model manager drops a pick its catalog does
  not list, so a catalog that followed the toggle lost one on a boot with it off."""

  def extends(self, chestnut=False, link=False):
    with mock.patch.object(backend, '_chestnut_fitted', return_value=chestnut), \
         mock.patch.object(backend.gadget, 'enabled', return_value=link):
      return backend.extends_catalog()

  def test_without_a_chestnut_it_is_extended_whatever_the_toggle(self):
    self.assertTrue(self.extends(link=False))
    self.assertTrue(self.extends(link=True))

  def test_a_chestnut_leaves_it_as_fetched(self):
    # without a jetlink checkout the backend never imports, and
    # sunnypilot.accelerators answers False; test_comma_layer covers that
    self.assertFalse(self.extends(chestnut=True, link=True))
    self.assertFalse(self.extends(chestnut=True, link=False))


class TestFetcherHook(OpenpilotTestCase):
  """The model manager asks once per fetch, for the big-model source only, and the
  cache records the answer."""

  def fetch(self, source, extends=True):
    response = mock.MagicMock(status_code=200)
    response.json.return_value = dict(PINNED)
    params = mock.MagicMock()
    with mock.patch('openpilot.sunnypilot.models.fetcher.requests.get', return_value=response), \
         mock.patch('openpilot.sunnypilot.accelerators.extends_catalog', return_value=extends) as asked, \
         mock.patch('openpilot.sunnypilot.accelerators.big_catalog', side_effect=lambda c: c) as hook:
      ModelFetcher(params)._fetch_and_cache_models(source)
    cached = next((c.args[1] for c in params.put.call_args_list if c.args[0] == 'ModelManager_ModelsCache_Chestnut'), None)
    return hook, asked, cached

  def test_the_big_model_source_is_extended_and_says_so(self):
    hook, asked, cached = self.fetch('chestnut')
    hook.assert_called_once_with(PINNED)
    asked.assert_called_once_with()
    self.assertIs(cached[ModelFetcher.EXTENDED_KEY], True)

  def test_beside_a_chestnut_it_is_cached_as_fetched(self):
    hook, _, cached = self.fetch('chestnut', extends=False)
    hook.assert_not_called()
    self.assertEqual(cached, {**PINNED, ModelFetcher.EXTENDED_KEY: False})

  def test_the_small_model_source_is_not(self):
    hook, asked, _ = self.fetch('qcom')
    hook.assert_not_called()
    asked.assert_not_called()


class TestCatalogFollowsTheHardware(OpenpilotTestCase):
  """A catalog cached beside a chestnut hides the newer models for an hour once it
  comes out, and the reverse. Refetch when the hardware changes."""

  def setUp(self):
    self.fetcher = ModelFetcher(mock.MagicMock())
    self.refetch = mock.patch.object(self.fetcher, '_fetch_and_cache_models', return_value=[]).start()
    self.addCleanup(mock.patch.stopall)

  def bundles(self, stamped, extends):
    cached = {**PINNED, ModelFetcher.EXTENDED_KEY: stamped}
    with mock.patch.object(self.fetcher.model_caches['chestnut'], 'get', return_value=(cached, False)), \
         mock.patch('openpilot.sunnypilot.accelerators.extends_catalog', return_value=extends):
      self.fetcher.get_bundles_for_source('chestnut')

  def test_a_cache_from_the_same_hardware_is_used(self):
    self.bundles(stamped=True, extends=True)
    self.bundles(stamped=False, extends=False)
    self.refetch.assert_not_called()

  def test_a_chestnut_coming_out_refetches(self):
    self.bundles(stamped=False, extends=True)
    self.refetch.assert_called_once_with('chestnut')

  def test_offline_it_is_tried_once_per_change(self):
    self.refetch.return_value = None   # the fetch failed; the old cache stands
    for _ in range(3):
      self.bundles(stamped=False, extends=True)
    self.assertEqual(self.refetch.call_count, 1)
    self.bundles(stamped=True, extends=False)   # a chestnut back in
    self.assertEqual(self.refetch.call_count, 2)

  def test_the_small_model_source_never_asks(self):
    with mock.patch.object(self.fetcher.model_caches['qcom'], 'get', return_value=({'bundles': []}, False)), \
         mock.patch('openpilot.sunnypilot.accelerators.extends_catalog') as extends:
      self.fetcher.get_bundles_for_source('qcom')
    extends.assert_not_called()


if __name__ == '__main__':
  unittest.main()
