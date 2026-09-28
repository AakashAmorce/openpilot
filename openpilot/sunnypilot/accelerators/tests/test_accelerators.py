"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The answers core openpilot gets from the accelerators module.

modeld, manager, hardwared and the UI all reach the Jetson through this
module, so what is pinned here is selection: what a device with the feature
off, on but not provisioned, and ready gets told, and that a backend that
hangs costs the large model and nothing else. A missing package is
test_comma_layer's. No hardware.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from jetlink.comma import gadget

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot import accelerators
from openpilot.sunnypilot.accelerators.jetlink import backend, helpers


class SelectionTest(OpenpilotTestCase):
  """ready() is params only, and every answer follows from the link setting,
  the pick, the spec record and whether the build made a warp for this camera."""

  def configure(self, enabled=None, model=None, spec_sha=None, ready=False, gadget_error=None, warp=True):
    # the spec record carries whether the engine for the sha it names is built
    params = {gadget.P_LINK: enabled, gadget.P_SPEC: {'sha256': spec_sha, 'ready': ready} if spec_sha else None}
    # the link setting is read off the param file, so stub the read rather than
    # Params: on is USB, off is off. Everything else goes through helpers._get
    for p in (mock.patch.object(helpers, '_get', side_effect=lambda k, d=None: params.get(k, d)),
              mock.patch.object(gadget, 'raw_param',
                                side_effect=lambda k: None if params.get(k) is None else
                                str(gadget.LINK_MODES.index('usb' if params[k] else 'off')).encode()),
              mock.patch.object(gadget, 'gadget_error', return_value=gadget_error),
              mock.patch.object(backend.warp_cache, 'built', return_value=warp),
              mock.patch.object(gadget, 'host_attached', return_value=False),
              mock.patch.object(gadget, 'dormant', return_value=False),
              mock.patch.object(helpers, 'selected_model',
                                return_value={'name': model, 'oid': 'a' * 64} if model else None),
              mock.patch.object(backend.spec_cache, 'load',
                                return_value=SimpleNamespace(sha256=spec_sha) if spec_sha else None)):
      p.start()
      self.addCleanup(p.stop)
    helpers._last_configured = 0.0

  def test_the_link_off_or_unset_is_disabled(self):
    for enabled in (None, False):
      with self.subTest(enabled=enabled):
        self.configure(enabled=enabled, model='m', spec_sha='a' * 64, ready=True, gadget_error='no gadget')
        self.assertFalse(accelerators.present())
        self.assertFalse(accelerators.ready())
        # A device with the feature off is never nagged about its kernel.
        self.assertIsNone(accelerators.unavailable_reason())
        self.assertFalse(accelerators.enabled())

  def test_enabled_but_not_provisioned(self):
    self.configure(enabled=True, model='m')
    self.assertFalse(accelerators.ready())
    self.assertIsNone(accelerators.unavailable_reason())
    self.assertIsNone(accelerators.active_model_name())

  def test_enabled_with_a_broken_gadget_says_why(self):
    self.configure(enabled=True, model='m', spec_sha='a' * 64, ready=True, gadget_error='no gadget')
    self.assertFalse(accelerators.ready())
    self.assertEqual(accelerators.unavailable_reason(), 'no gadget')

  def test_no_warp_for_this_camera_says_why(self):
    # the build made none, and nothing compiles one at runtime: every drive
    # would be the small model while the panel said ready
    self.configure(enabled=True, model='m', spec_sha='a' * 64, ready=True, warp=False)
    self.assertFalse(accelerators.ready())
    self.assertEqual(accelerators.unavailable_reason(), backend.NO_WARP)
    self.configure(enabled=False, warp=False)
    self.assertIsNone(accelerators.unavailable_reason())

  def test_ready(self):
    self.configure(enabled=True, model='m', spec_sha='a' * 64, ready=True)
    self.assertTrue(accelerators.ready())
    self.assertIsNone(accelerators.unavailable_reason())
    self.assertTrue(accelerators.enabled())

  def test_an_old_engine_is_not_the_new_selection(self):
    self.configure(enabled=True, model='m', spec_sha='b' * 64, ready=True)
    self.assertFalse(accelerators.ready())

  def test_a_spec_whose_engine_is_not_built_is_not_ready(self):
    self.configure(enabled=True, model='m', spec_sha='a' * 64, ready=False)
    self.assertFalse(accelerators.ready())

  def test_enabled_is_the_toggle_alone(self):
    # configuration only, never link state or ready(). The model defaults
    # through selected_model(), so it is not part of it either
    self.configure(enabled=True, model=None)
    self.assertTrue(accelerators.enabled())
    self.configure(enabled=True, model='m')
    with mock.patch.object(gadget, 'link_configured', return_value=False):
      self.assertTrue(accelerators.enabled())
    self.configure(enabled=None, model='m')
    with mock.patch.object(gadget, 'link_configured', return_value=True):
      self.assertFalse(accelerators.enabled())

  def test_present_is_usb_independent_while_dormant(self):
    self.configure(enabled=True, model='m')
    with mock.patch.object(gadget, 'dormant', return_value=True), \
         mock.patch.object(gadget, 'CC_ORIENTATION', mock.Mock(read_text=lambda: '1')):
      self.assertTrue(accelerators.present())


class LenderFailureTest(OpenpilotTestCase):
  """Only the owner holds ep0. When its lender cannot listen it keeps the
  gadget, retries, and records why: that line is the offroad alert. modeld
  still prepares, so its join picks the link up once the lender listens."""

  def test_the_owners_lender_error_is_the_alert_not_a_no(self):
    tmp = Path(tempfile.mkdtemp())
    built = tmp / 'jetlink-gadget'
    built.write_text('ok\n')
    self.addCleanup(setattr, backend, '_prepared', False)
    with mock.patch.object(gadget, 'GADGET_STATUS', built), \
         mock.patch.object(gadget, 'LENDER_STATUS', tmp / 'jetlink-lender'), \
         mock.patch.object(gadget, 'link_configured', return_value=True), \
         mock.patch.object(backend, 'enabled', return_value=True), \
         mock.patch.object(backend.warp_cache, 'built', return_value=True), \
         mock.patch.object(backend.warp_cache, 'init_device'):
      gadget.note_lender_error('address in use')
      self.assertEqual(accelerators.unavailable_reason(), 'the lender could not listen: address in use')
      self.assertFalse(accelerators.ready())
      self.assertTrue(accelerators.prepare())
      # cleared once the lender listens again
      gadget.note_lender_error(None)
      self.assertIsNone(accelerators.unavailable_reason())


class LoadTest(OpenpilotTestCase):
  """modeld's two calls: prepare() before it goes realtime, load() once the camera is up."""

  def setUp(self):
    backend._prepared = False
    self.addCleanup(setattr, backend, '_prepared', False)
    self.small = SimpleNamespace(name='small', client=None)

  def prepared(self):
    """prepare() down its yes path, with nothing real behind it."""
    with mock.patch.object(backend, 'enabled', return_value=True), \
         mock.patch.object(gadget, 'link_configured', return_value=True), \
         mock.patch.object(backend.warp_cache, 'built', return_value=True), \
         mock.patch.object(backend.warp_cache, 'init_device') as init_device:
      self.assertTrue(accelerators.prepare())
    init_device.assert_called_once()

  def test_no_warp_is_no_before_the_gpu_comes_up(self):
    with mock.patch.object(backend, 'enabled', return_value=True), \
         mock.patch.object(gadget, 'link_configured', return_value=True), \
         mock.patch.object(backend.warp_cache, 'built', return_value=False), \
         mock.patch.object(backend.warp_cache, 'init_device') as init_device:
      self.assertFalse(accelerators.prepare())
    init_device.assert_not_called()

  def test_the_link_off_says_no_before_any_setup(self):
    with mock.patch.object(backend, 'enabled', return_value=False), \
         mock.patch.object(gadget, 'link_configured') as link_configured:
      self.assertFalse(accelerators.prepare())
    link_configured.assert_not_called()

  def test_nothing_joins_without_prepare(self):
    # the GPU's thread would start on modeld's realtime core
    with mock.patch.object(backend, 'make_model_state') as build:
      self.assertIsNone(accelerators.load(1928, 1208, self.small))
    build.assert_not_called()

  def test_a_later_no_takes_the_yes_back(self):
    self.prepared()
    with mock.patch.object(backend, 'enabled', return_value=False):
      self.assertFalse(accelerators.prepare())
    with mock.patch.object(backend, 'make_model_state') as build:
      self.assertIsNone(accelerators.load(1928, 1208, self.small))
    build.assert_not_called()

  def test_prepared_joins_modeld(self):
    self.prepared()
    joining = SimpleNamespace(client=object())
    with mock.patch.object(backend, 'make_model_state', return_value=joining) as build:
      loaded = accelerators.load(1928, 1208, self.small)
    build.assert_called_once_with(1928, 1208, self.small)
    self.assertIsInstance(loaded, accelerators.Accelerator)
    self.assertIs(loaded.model, joining)
    # the status reads the model's client per send: the link comes and goes mid-drive
    self.assertIs(loaded.status.client, joining.client)
    joining.client = None
    self.assertIsNone(loaded.status.client)

  def test_a_failed_build_drives_the_small_model_and_says_so(self):
    self.prepared()
    with mock.patch.object(backend, 'make_model_state', side_effect=RuntimeError('no warp')), \
         mock.patch.object(backend.cloudlog, 'exception') as log:
      loaded = accelerators.load(1928, 1208, self.small)
    log.assert_called_once_with("jetlink load failed")
    # as modeld always did it: the accelerator's status, over the small model
    self.assertIs(loaded.model, self.small)
    self.assertIs(loaded.status.model, self.small)


class TestProgress(OpenpilotTestCase):
  def test_a_missing_param_is_no_progress(self):
    with mock.patch.object(accelerators, '_params') as params:
      params.return_value.get.return_value = None
      self.assertIsNone(accelerators.progress())

  def test_a_dict_comes_through(self):
    payload = {'stage': 'build', 'frac': 0.5, 'msg': ''}
    with mock.patch.object(accelerators, '_params') as params:
      params.return_value.get.return_value = payload
      self.assertEqual(accelerators.progress(), payload)

  def test_a_non_dict_is_ignored(self):
    with mock.patch.object(accelerators, '_params') as params:
      params.return_value.get.return_value = "build 50%"
      self.assertIsNone(accelerators.progress())

  def test_an_unknown_key_does_not_take_down_the_ui(self):
    # A params library older than this key raises rather than returning None.
    with mock.patch.object(accelerators, '_params') as params:
      params.return_value.get.side_effect = RuntimeError("UnknownKeyName")
      self.assertIsNone(accelerators.progress())

  def test_reporting_never_raises(self):
    # Called from except handlers in the daemons.
    with mock.patch.object(accelerators, '_params') as params:
      params.return_value.put.side_effect = RuntimeError("params gone")
      accelerators.report_progress('build', 0.5)
      params.return_value.remove.side_effect = RuntimeError("params gone")
      accelerators.clear_progress()


class TestShutdown(OpenpilotTestCase):
  def test_disabled_costs_one_param_read_and_nothing_else(self):
    with mock.patch.object(gadget, 'enabled', return_value=False), \
         mock.patch.object(backend, 'shutdown') as request:
      accelerators.shutdown('car battery')
    request.assert_not_called()

  def test_the_request_is_forwarded_with_the_bound(self):
    with mock.patch.object(gadget, 'enabled', return_value=True), \
         mock.patch.object(backend, 'shutdown') as request:
      accelerators.shutdown('car battery', timeout=3.0)
    request.assert_called_once_with('car battery', 3.0)

  def test_a_backend_that_hangs_cannot_hold_hardwared(self):
    release = threading.Event()
    self.addCleanup(release.set)
    with mock.patch.object(gadget, 'enabled', return_value=True), \
         mock.patch.object(backend, 'shutdown', side_effect=lambda *a: release.wait(30)), \
         mock.patch.object(accelerators._log(), 'warning') as warn:
      t0 = time.monotonic()
      accelerators.shutdown('car battery', timeout=0.2)
      self.assertLess(time.monotonic() - t0, 2.0)
    warn.assert_called_once()

  def test_a_backend_that_raises_is_logged_not_propagated(self):
    with mock.patch.object(gadget, 'enabled', return_value=True), \
         mock.patch.object(backend, 'shutdown', side_effect=RuntimeError('no')), \
         mock.patch.object(accelerators._log(), 'exception') as log:
      accelerators.shutdown('car battery', timeout=1.0)
    log.assert_called_once()


if __name__ == '__main__':
  unittest.main()
