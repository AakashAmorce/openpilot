"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import tempfile
import urllib.request
from pathlib import Path
from unittest import mock

from jetlink.comma import gadget

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.accelerators.jetlink import helpers


class TestPresence(OpenpilotTestCase):
  """What the panels are told is on the other end, and hardwared's wait for
  the owner's run to take a shutdown request. The markers are jetlink.comma's."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    for name in ('DORMANT', 'SHUTDOWN_REQUEST'):
      # gadget owns the paths and reads them from its own namespace
      patcher = mock.patch.object(gadget, name, self.tmp / name.lower())
      self.addCleanup(patcher.stop)
      patcher.start()

  def test_dormant_counts_as_present_without_a_host(self):
    with mock.patch.object(gadget, 'host_attached', return_value=False), \
         mock.patch.object(gadget, 'CC_ORIENTATION', self.tmp / 'cc'):
      (self.tmp / 'cc').write_text('1')
      helpers._last_configured = 0.0
      assert not helpers.gadget_present()
      gadget.set_dormant(True)
      assert helpers.gadget_present()
      (self.tmp / 'cc').write_text('0')
      assert not helpers.gadget_present()

  def test_await_shutdown_gives_up_and_cleans_up(self):
    gadget.request_shutdown('car battery')
    assert not helpers.await_shutdown(0.3)
    assert gadget.pending_shutdown() is None

  def test_await_shutdown_returns_when_taken(self):
    gadget.request_shutdown('car battery')
    gadget.finish_shutdown()
    assert helpers.await_shutdown(0.3)


class TestConnect(OpenpilotTestCase):
  """Which transport the client is opened over: whatever the owner lent, a
  phone's dial or the endpoint files. Never the gadget itself."""

  def setUp(self):
    self.client = mock.patch('jetlink.client.JetlinkClient').start()
    self.addCleanup(mock.patch.stopall)

  def test_a_loan_with_a_dial_is_opened_over_the_socket(self):
    sock = mock.Mock(name='sock')
    helpers.connect(deadline=2.0, name='modeld', loan=mock.Mock(sock=sock))
    self.client.open_socket.assert_called_once_with(sock, deadline=2.0, name='modeld')
    self.client.open_borrowed_ffs.assert_not_called()

  def test_a_loan_of_the_endpoints_is_opened_over_them(self):
    # whatever the link record says: the owner decided once, when it lent, and
    # the record is for the panels
    loan = mock.Mock(sock=None, mount='/dev/ffs-jetlink', udc='udc0')
    with mock.patch.object(gadget, 'link_kind', return_value='cable') as kind:
      helpers.connect(name='modeld', loan=loan)
    self.client.open_borrowed_ffs.assert_called_once()
    assert self.client.open_borrowed_ffs.call_args.args[:2] == ('/dev/ffs-jetlink', 'udc0')
    self.client.open_socket.assert_not_called()
    self.client.open_ffs.assert_not_called()
    kind.assert_not_called()


def bundle(ref: str, name: str, index: int = 0, version=19) -> dict:
  """A bundle as the catalog JSON carries it."""
  return {'ref': ref, 'display_name': name, 'index': index, 'minimum_selector_version': str(version)}


REF_A, REF_B, REF_C = 'a' * 40, 'b' * 40, 'c' * 40
POINTERS = {REF_A: {'oid': '1' * 64, 'size': 766_000_000},
            REF_B: {'oid': '2' * 64, 'size': 1_757_000_000}}


def catalog_param(*bundles):
  params = mock.patch.object(helpers, 'params')
  params.start().return_value.get.return_value = {'bundles': list(bundles)}
  return params


class TestCatalog(OpenpilotTestCase):
  """The list is sunnypilot's big-model catalog, read as the model manager cached it."""

  def setUp(self):
    self.addCleanup(mock.patch.stopall)

  def test_newest_first_with_names(self):
    catalog_param(bundle(REF_A, 'Alpha (September 04, 2026)', 3), bundle(REF_B, 'Beta', 9))
    self.assertEqual(helpers.catalog(), [{'name': 'Beta', 'ref': REF_B}, {'name': 'Alpha (September 04, 2026)', 'ref': REF_A}])

  def test_only_commits_of_this_selector_version(self):
    # a bundle without a comma commit has no ONNX to find; one for another
    # selector version is one the model manager itself would not list
    catalog_param(bundle('not-a-commit', 'Odd', 5), bundle(REF_A, 'Alpha', 1), {'display_name': 'Blank'},
                  bundle(REF_C, 'Gamma', 7, version=18))
    self.assertEqual([b['ref'] for b in helpers.catalog()], [REF_A])

  def test_no_catalog_yet_is_empty(self):
    catalog_param()
    self.assertEqual(helpers.catalog(), [])

  def test_an_unreadable_catalog_is_empty_not_an_error(self):
    # read from the UI's param thread, where an exception takes the panel down
    params = mock.patch.object(helpers, 'params').start()
    params.return_value.get.side_effect = RuntimeError('no params')
    self.assertEqual(helpers.catalog(), [])


class TestModelIndex(OpenpilotTestCase):
  """Every catalog model, with the ONNX behind it once that has been looked up."""

  def setUp(self):
    helpers._index_cache = None
    self.addCleanup(setattr, helpers, '_index_cache', None)

  def index_with(self, bundles, pointers=POINTERS):
    with mock.patch.object(helpers, 'catalog', return_value=bundles), \
         mock.patch.object(helpers, 'pointers', return_value=pointers):
      return helpers.model_index()

  def test_a_resolved_model_carries_its_identity(self):
    (entry,) = self.index_with([{'name': 'Alpha', 'ref': REF_A}])
    self.assertEqual(entry, {'name': 'Alpha', 'ref': REF_A, 'oid': '1' * 64, 'size': 766_000_000})

  def test_an_unresolved_model_is_still_listed(self):
    # the pointer is fetched when the model is first asked for
    (entry,) = self.index_with([{'name': 'Gamma', 'ref': REF_C}])
    self.assertEqual((entry['name'], entry['oid'], entry['size']), ('Gamma', None, None))

  def test_a_second_read_within_the_ttl_costs_nothing(self):
    # the UI names the active model every frame
    with mock.patch.object(helpers, 'catalog', return_value=[]) as read, mock.patch.object(helpers, 'pointers', return_value={}):
      first = helpers.model_index()
      self.assertIs(helpers.model_index(), first)
    self.assertEqual(read.call_count, 1)


class TestResolvePointer(OpenpilotTestCase):
  """The pointer at a commit is the oid and size the Jetson is asked for,
  fetched the first time a model is asked for and kept for good."""

  POINTER = f"version https://git-lfs.github.com/spec/v1\noid sha256:{'3' * 64}\nsize 766040736\n"

  def setUp(self):
    self.params = mock.patch.object(helpers, 'params').start()
    self.addCleanup(mock.patch.stopall)
    helpers._index_cache = (float('inf'), [], {})   # a stale index must be dropped on a hit
    self.addCleanup(setattr, helpers, '_index_cache', None)

  def response(self, body: bytes):
    r = mock.MagicMock()
    r.__enter__.return_value = r
    r.read.return_value = body
    return r

  def test_fetches_once_and_records_it(self):
    from jetlink.registry.lfs import POINTER_URL
    with mock.patch.object(helpers, '_get', return_value=dict(POINTERS)), \
         mock.patch.object(urllib.request, 'urlopen', return_value=self.response(self.POINTER.encode())) as urlopen:
      self.assertEqual(helpers.resolve_pointer(REF_C), ('3' * 64, 766040736))
    self.assertEqual(urlopen.call_args.args[0], POINTER_URL.format(ref=REF_C))
    written = self.params.return_value.put.call_args.args[1]
    self.assertEqual(written[REF_C], {'oid': '3' * 64, 'size': 766040736})
    self.assertEqual(written[REF_A], POINTERS[REF_A])
    self.assertIsNone(helpers._index_cache)

  def test_a_known_pointer_needs_no_fetch(self):
    with mock.patch.object(helpers, '_get', return_value=dict(POINTERS)), \
         mock.patch.object(urllib.request, 'urlopen') as urlopen:
      self.assertEqual(helpers.resolve_pointer(REF_A), ('1' * 64, 766_000_000))
    urlopen.assert_not_called()

  def test_a_miss_raises_and_records_nothing(self):
    from jetlink.registry.catalog import RegistryError
    for failure in ({'side_effect': OSError('offline')}, {'return_value': self.response(b'<html>not found</html>')}):
      with self.subTest(failure), mock.patch.object(helpers, '_get', return_value={}), \
           mock.patch.object(urllib.request, 'urlopen', **failure), self.assertRaises(RegistryError):
        helpers.resolve_pointer(REF_C)
    self.params.return_value.put.assert_not_called()

  def test_the_lookup_is_the_registry_s(self):
    """One resolver for both ends; the precompiled-pkl commits are tested there."""
    from jetlink.registry.lfs import Pointer
    with mock.patch.object(helpers, '_get', return_value={}), \
         mock.patch('jetlink.registry.lfs.fetch_pointer', return_value=Pointer('4' * 64, 766354845)) as fetch:
      self.assertEqual(helpers.resolve_pointer(REF_C), ('4' * 64, 766354845))
    fetch.assert_called_once_with(REF_C, timeout=helpers.POINTER_TIMEOUT)


class TestSelectedModel(OpenpilotTestCase):
  """The pick is the model manager's big-model slot, the same one a chestnut runs
  from, listed in the catalog or not; the catalog only supplies the default."""

  INDEX = [
    {'name': 'Alpha', 'ref': REF_A, 'oid': 'a' * 64, 'size': 10},
    {'name': 'Beta', 'ref': REF_B, 'oid': 'b' * 64, 'size': 20},
  ]

  def select_with(self, slot_ref, default=REF_B, index=None, known=None):
    pick = {'name': 'Picked', 'ref': slot_ref} if slot_ref else None
    with mock.patch.object(helpers, '_index', return_value=(self.INDEX if index is None else index, known or {})), \
         mock.patch.object(helpers, 'DEFAULT_BIG_MODEL_REF', default), \
         mock.patch.object(helpers, 'selected_slot', return_value=pick):
      return helpers.selected_model()

  def test_an_empty_slot_takes_the_default_big_model(self):
    assert self.select_with(None)['name'] == 'Beta'

  def test_the_default_is_jetlinks_not_the_chestnuts(self):
    # a chestnut's default is the model in the tree; the accelerator's is jetlink's
    from jetlink.registry.catalog import DEFAULT_BIG_MODEL_REF
    assert helpers.DEFAULT_BIG_MODEL_REF == DEFAULT_BIG_MODEL_REF

  def test_a_default_not_in_the_catalog_falls_to_the_newest(self):
    assert self.select_with(None, default='f' * 40)['name'] == 'Alpha'

  def test_the_pick_is_the_slot_with_its_pointer(self):
    for ref in (REF_A, REF_C):   # listed or not
      known = {ref: {'oid': 'c' * 64, 'size': '30'}}
      assert self.select_with(ref, known=known) == {'name': 'Picked', 'ref': ref, 'oid': 'c' * 64, 'size': 30}

  def test_a_pick_not_yet_resolved_has_no_oid(self):
    # the worker resolves it from the ref, as for any catalog model
    assert self.select_with(REF_C) == {'name': 'Picked', 'ref': REF_C, 'oid': None, 'size': None}

  def test_a_pick_needs_no_catalog(self):
    assert self.select_with(REF_C, index=[])['ref'] == REF_C

  def test_no_pick_and_no_catalog_is_no_model(self):
    assert self.select_with(None, index=[]) is None


class TestDefaultModelName(OpenpilotTestCase):
  """The accelerator's default, named as the chestnut's DEFAULT_BIG_MODEL is: the
  catalog's display name without its build date."""

  def name_with(self, names, default=REF_B):
    index = [{'name': name, 'ref': ref, 'oid': None, 'size': None} for name, ref in zip(names, (REF_A, REF_B), strict=False)]
    with mock.patch.object(helpers, '_index', return_value=(index, {})), \
         mock.patch.object(helpers, 'DEFAULT_BIG_MODEL_REF', default):
      return helpers.default_model_name()

  def test_the_build_date_is_dropped(self):
    assert self.name_with(['Alpha', 'Cinque Terre V3 Model (September 17, 2026)']) == 'Cinque Terre V3 Model'

  def test_a_parenthesis_that_is_not_a_date_stays(self):
    assert self.name_with(['Alpha', 'Beta (big)']) == 'Beta (big)'

  def test_a_default_not_listed_names_the_newest(self):
    assert self.name_with(['Alpha (September 01, 2026)', 'Beta'], default='f' * 40) == 'Alpha'

  def test_no_catalog_is_no_name(self):
    assert self.name_with([]) is None


class TestSelectedSlot(OpenpilotTestCase):
  def setUp(self):
    helpers._slot_cache = None
    self.addCleanup(setattr, helpers, '_slot_cache', None)

  def read_with(self, slot):
    helpers._slot_cache = None
    with mock.patch.object(helpers, '_get', return_value=slot):
      return helpers.selected_slot()

  def test_a_second_read_within_the_ttl_costs_nothing(self):
    # the UI names the active model every frame
    with mock.patch.object(helpers, '_get', return_value={'ref': REF_A}) as read:
      helpers._slot_cache = None
      assert helpers.selected_slot()['ref'] == REF_A
      assert helpers.selected_slot()['ref'] == REF_A
    assert read.call_count == 1

  def test_reads_the_slots_ref_and_name(self):
    assert self.read_with({'ref': REF_A, 'displayName': 'Alpha'}) == {'name': 'Alpha', 'ref': REF_A}
    assert self.read_with({'ref': REF_A}) == {'name': REF_A[:10], 'ref': REF_A}

  def test_anything_else_is_no_pick(self):
    for slot in (None, {}, {'ref': ''}, {'ref': 7}, 'junk'):
      assert self.read_with(slot) is None, slot


class TestSelectedModelReadiness(OpenpilotTestCase):
  def test_old_cached_engine_is_not_the_new_selection(self):
    # What the UI calls compiled. The join does not stop here: an engine the
    # Jetson has not got is built onroad, see backend._open_link.
    from types import SimpleNamespace
    from openpilot.sunnypilot.accelerators.jetlink import backend

    with mock.patch.object(gadget, 'enabled', return_value=True), \
         mock.patch.object(backend.warp_cache, 'built', return_value=True), \
         mock.patch.object(backend.spec_cache, 'engine_ready_for', return_value=True), \
         mock.patch.object(backend.spec_cache, 'load', return_value=SimpleNamespace(sha256='a' * 64)), \
         mock.patch.object(helpers, 'selected_model', return_value={'oid': 'b' * 64}) as selected:
      self.assertFalse(backend.ready())
      selected.return_value = {'oid': 'a' * 64}
      self.assertTrue(backend.ready())


class TestShippedModelPath(OpenpilotTestCase):
  """A file counts only when it is the model we mean, at the size we expect.
  Models live one file per oid so switching back does not re-download."""

  MODEL = {'name': 'Alpha', 'ref': REF_A, 'oid': 'a' * 64, 'size': 4096}

  def setUp(self):
    self.root = tempfile.mkdtemp()
    paths = mock.patch('openpilot.sunnypilot.accelerators.jetlink.helpers.Paths')
    self.addCleanup(paths.stop)
    paths.start().model_root.return_value = self.root
    chosen = mock.patch.object(helpers, 'selected_model', return_value=self.MODEL)
    self.addCleanup(chosen.stop)
    chosen.start()

  def fetched(self, size: int) -> Path:
    path = helpers.model_dir() / helpers.model_file_name(self.MODEL)
    path.parent.mkdir()
    path.write_bytes(b'\0' * size)
    return path

  def test_a_model_not_resolved_yet_is_no_path(self):
    with mock.patch.object(helpers, 'selected_model', return_value={**self.MODEL, 'oid': None, 'size': None}):
      assert helpers.shipped_model_path() is None

  def test_nothing_fetched_yet(self):
    assert helpers.shipped_model_path() is None

  def test_the_chosen_model_is_accepted(self):
    fetched = self.fetched(4096)
    assert helpers.shipped_model_path() == fetched

  def test_a_truncated_download_is_rejected(self):
    # Half a model is exactly what must not reach TensorRT.
    self.fetched(2048)
    assert helpers.shipped_model_path() is None

  def test_no_model_chosen_is_no_path(self):
    with mock.patch.object(helpers, 'selected_model', return_value=None):
      assert helpers.shipped_model_path() is None


class TestFetchShippedModel(OpenpilotTestCase):
  """The bytes come through jetlink's registry downloader; all the fork adds is
  the LFS server its own .lfsconfig names, asked first."""

  MODEL = {'name': 'Alpha', 'ref': REF_A, 'oid': 'a' * 64, 'size': 4096}

  def setUp(self):
    self.root = Path(tempfile.mkdtemp())
    for target, value in (('BASEDIR', str(self.root)), ('Paths', mock.Mock(model_root=lambda: str(self.root / 'models')))):
      patcher = mock.patch.object(helpers, target, value)
      self.addCleanup(patcher.stop)
      patcher.start()
    chosen = mock.patch.object(helpers, 'selected_model', return_value=self.MODEL)
    self.addCleanup(chosen.stop)
    chosen.start()

  def test_the_checkouts_server_comes_first(self):
    from jetlink.registry.lfs import LFS_ENDPOINTS
    (self.root / '.lfsconfig').write_text('[lfs]\n\turl = https://example.com/info/lfs/\n')
    self.assertEqual(helpers.lfs_endpoints(), ['https://example.com/info/lfs', *LFS_ENDPOINTS])

  def test_without_one_it_is_jetlinks_list(self):
    # a release ships no .lfsconfig
    from jetlink.registry.lfs import LFS_ENDPOINTS
    self.assertEqual(helpers.lfs_endpoints(), list(LFS_ENDPOINTS))

  def test_no_duplicate_when_it_is_already_one_of_jetlinks(self):
    from jetlink.registry.lfs import LFS_ENDPOINTS
    (self.root / '.lfsconfig').write_text(f'[lfs]\n\turl = {LFS_ENDPOINTS[0]}\n')
    self.assertEqual(helpers.lfs_endpoints(), list(LFS_ENDPOINTS))

  def test_already_fetched_is_returned_as_is(self):
    dest = helpers.model_dir() / helpers.model_file_name(self.MODEL)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'\0' * 4096)
    with mock.patch('jetlink.registry.lfs.lfs_resolve') as resolve:
      self.assertEqual(helpers.fetch_shipped_model(), dest)
    resolve.assert_not_called()

  def test_falls_through_to_the_next_server(self):
    from jetlink.registry.lfs import LFS_ENDPOINTS, Pointer
    (self.root / '.lfsconfig').write_text('[lfs]\n\turl = https://dead.example/info/lfs\n')
    stop = object()
    with mock.patch('jetlink.registry.lfs.lfs_resolve', side_effect=[None, 'https://x/y']) as resolve, \
         mock.patch('jetlink.registry.lfs.lfs_download', side_effect=lambda href, pointer, dest, **kw: dest) as download:
      dest = helpers.fetch_shipped_model(should_stop=stop)
    self.assertEqual([c.args[0] for c in resolve.call_args_list], ['https://dead.example/info/lfs', LFS_ENDPOINTS[0]])
    self.assertEqual(download.call_args.args[:3], ('https://x/y', Pointer('a' * 64, 4096), dest))
    self.assertIs(download.call_args.kwargs['should_stop'], stop)

  def test_nowhere_to_get_it_raises(self):
    from jetlink.registry.catalog import NetworkError
    with mock.patch('jetlink.registry.lfs.lfs_resolve', return_value=None), self.assertRaises(NetworkError):
      helpers.fetch_shipped_model()

  def test_nothing_chosen_is_nothing_fetched(self):
    with mock.patch.object(helpers, 'selected_model', return_value={**self.MODEL, 'oid': None}):
      self.assertIsNone(helpers.fetch_shipped_model())
