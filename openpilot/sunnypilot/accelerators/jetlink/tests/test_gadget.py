"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from openpilot.sunnypilot.accelerators.jetlink import gadget

# what the gadget owner must never end up importing. swaglog pulls all three in
# to publish a log line and costs 28 MB; params imports swaglog. Measured on the
# comma: this module plus the transport is 10.4 MB, against 47.5 MB for the
# daemon that imported the world
HEAVY = ('numpy', 'capnp', 'zmq', 'cereal')


class TestNothingHeavyIsReachable(unittest.TestCase):
  """The owner is only small while this holds, so it is a test and not a note."""

  def imported_by(self, module: str) -> set[str]:
    """Top-level packages a fresh interpreter has after importing `module`."""
    roots = 'sorted({m.split(".")[0] for m in sys.modules})'
    code = f'import sys, json; __import__("{module}"); print(json.dumps({roots}))'
    root = Path(__file__).resolve().parents[5]
    # this runner's own path, so the jetlink package is found wherever it is
    # checked out; a bare PYTHONPATH found the tree but not the submodule
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(root), *sys.path])}
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env=env, cwd=str(root), timeout=120)
    self.assertEqual(out.returncode, 0, out.stderr)
    import json
    return set(json.loads(out.stdout))

  def test_the_gadget_core_stays_out_of_the_heavy_half(self):
    found = self.imported_by('openpilot.sunnypilot.accelerators.jetlink.gadget')
    self.assertEqual(sorted(found & set(HEAVY)), [],
                     'the gadget owner has to stay small; see the module docstring')

  def test_the_owner_itself_stays_out_of_the_heavy_half(self):
    found = self.imported_by('openpilot.sunnypilot.accelerators.jetlink.owner')
    self.assertEqual(sorted(found & set(HEAVY)), [],
                     'everything the owner imports runs for the whole drive')

  def test_the_transport_the_owner_opens_is_light_too(self):
    try:
      import jetlink  # noqa: F401
    except ImportError:
      self.skipTest('the jetlink package is not checked out')
    found = self.imported_by('jetlink.transport.ffs')
    self.assertEqual(sorted(found & set(HEAVY)), [])


class TestParamsOffTheFilesystem(unittest.TestCase):
  """params.cc writes a value to a temp file, fsyncs it, renames it over the key
  and fsyncs the directory, so a plain read never sees a torn value."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    (self.tmp / 'd').mkdir()
    self.enterContext(unittest.mock.patch.dict(os.environ, {'PARAMS_ROOT': str(self.tmp)}))
    os.environ.pop('OPENPILOT_PREFIX', None)

  def write(self, key: str, value: bytes) -> None:
    (self.tmp / 'd' / key).write_bytes(value)

  def test_the_path_follows_the_prefix(self):
    self.assertEqual(gadget.params_dir(), self.tmp / 'd')
    with unittest.mock.patch.dict(os.environ, {'OPENPILOT_PREFIX': 'abc123'}):
      self.assertEqual(gadget.params_dir(), self.tmp / 'abc123')

  def test_a_missing_param_is_not_a_false(self):
    # None and False are different answers: offroad treats an unwritten param
    # as parked, and enabled treats it as off
    self.assertIsNone(gadget.param_bool(gadget.P_ENABLED))
    self.assertFalse(gadget.enabled())
    self.assertTrue(gadget.offroad())

  def test_the_toggle_is_true_and_nothing_else(self):
    for raw, expected in ((b'1', True), (b'0', False), (b'', False), (b'true', True)):
      self.write(gadget.P_ENABLED, raw)
      self.assertIs(gadget.enabled(), expected, raw)

  def test_an_unreadable_store_is_not_an_error(self):
    with unittest.mock.patch.dict(os.environ, {'PARAMS_ROOT': '/nonexistent'}):
      self.assertIsNone(gadget.raw_param(gadget.P_ENABLED))
      self.assertFalse(gadget.enabled())


class TestLinkKind(unittest.TestCase):
  """What carries the link. The explicit endpoint wins, then the setting's
  iOS, which is the phone's cable; else USB. The owner's record only says
  which phone dialed."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    for name, value in (('LINK', self.tmp / 'link'), ('UDC_PATH', self.tmp / 'udc'),
                        ('NET_STATUS', self.tmp / 'net'),
                        ('ios', unittest.mock.Mock(return_value=False)),
                        ('link_endpoint', unittest.mock.Mock(return_value=None))):
      p = unittest.mock.patch.object(gadget, name, value)
      self.addCleanup(p.stop)
      p.start()

  def test_nothing_recorded_is_usb(self):
    self.assertEqual(gadget.link_kind(), 'usb')
    self.assertIsNone(gadget.link_peer())
    self.assertFalse(gadget.over_tcp())

  def test_ios_is_the_cable_before_any_dial(self):
    gadget.ios.return_value = True
    self.assertEqual(gadget.link_kind(), 'cable')
    self.assertTrue(gadget.over_tcp())
    self.assertIsNone(gadget.link_peer())

  def test_the_owners_record_decides_over_the_setting(self):
    # a setting moved while somebody borrowed waits for the car to park; until
    # the owner rebuilds, the gadget is what it built
    gadget.ios.return_value = True
    gadget.note_link('usb')
    self.assertEqual(gadget.link_kind(), 'usb')
    gadget.ios.return_value = False
    gadget.note_link('cable', '192.168.60.3')
    self.assertEqual(gadget.link_kind(), 'cable')
    gadget.clear_link()
    self.assertEqual(gadget.link_kind(), 'usb', 'no owner yet: the setting')

  def test_a_dial_is_recorded_with_the_phone_and_cleared(self):
    gadget.note_link('cable', '192.168.60.3')
    self.assertEqual(gadget.link_peer(), '192.168.60.3')
    gadget.clear_link()
    self.assertIsNone(gadget.link_peer())
    gadget.clear_link()   # twice is not an error

  def test_the_endpoint_param_wins_over_the_record(self):
    gadget.note_link('cable', '192.168.60.3')
    gadget.link_endpoint.return_value = ('10.0.0.5', 5599)
    self.assertEqual(gadget.link_kind(), 'ethernet')
    self.assertTrue(gadget.over_tcp())

  def test_over_tcp_there_is_no_host_to_wait_for(self):
    # the connect already reached the phone or the Jetson; on the cable the
    # UDC is configured by a phone, and it is the dial that proved it
    with unittest.mock.patch.object(gadget, 'udc_state', return_value='powered'), \
         unittest.mock.patch.object(gadget.time, 'sleep', side_effect=AssertionError('waited')):
      gadget.ios.return_value = True
      self.assertTrue(gadget.wait_for_host(5.0, report=lambda: self.fail('reported a wait')))
      gadget.ios.return_value = False
      gadget.link_endpoint.return_value = ('10.0.0.5', 5599)
      self.assertTrue(gadget.wait_for_host(5.0))
      gadget.link_endpoint.return_value = None
      self.assertFalse(gadget.wait_for_host(0.0))

  def test_only_the_endpoint_bypasses_the_gadget(self):
    # a phone on the cable is on the gadget's own network interface
    with unittest.mock.patch.object(gadget, 'FFS_MOUNT', self.tmp / 'ffs'), \
         unittest.mock.patch.object(gadget, 'GADGET_STATUS', self.tmp / 'status'):
      gadget.note_link('cable', '192.168.60.3')
      self.assertFalse(gadget.link_configured())
      gadget.link_endpoint.return_value = ('10.0.0.5', 5599)
      self.assertTrue(gadget.link_configured())

  def test_the_bus_speed_is_read_off_the_bound_udc(self):
    with unittest.mock.patch.object(gadget, 'bound_udc', return_value=None):
      self.assertIsNone(gadget.usb_speed())
    with unittest.mock.patch.object(gadget, 'bound_udc', return_value='a600000.dwc3'):
      self.assertIsNone(gadget.usb_speed())
      (gadget.UDC_PATH / 'a600000.dwc3').mkdir(parents=True)
      (gadget.UDC_PATH / 'a600000.dwc3' / 'current_speed').write_text('super-speed\n')
      self.assertEqual(gadget.usb_speed(), 'super-speed')

  def test_the_network_status_is_the_scripts(self):
    self.assertIsNone(gadget.net_status())
    gadget.NET_STATUS.write_text('ok 192.168.60.1\n')
    self.assertEqual(gadget.net_status(), 'ok 192.168.60.1')


if __name__ == '__main__':
  unittest.main()


class TestTheTwoGadgets(unittest.TestCase):
  """The setting picks the gadget: the plain one for USB, the composite one
  with a network interface for iOS."""

  def test_setup_passes_the_setting(self):
    with unittest.mock.patch.object(gadget.subprocess, 'run') as run, \
         unittest.mock.patch.object(gadget, 'link_configured', return_value=True):
      self.assertTrue(gadget.setup_gadget(True))
      self.assertEqual(run.call_args.args[0][-1], '--ios')
      self.assertTrue(gadget.setup_gadget(False))
      self.assertNotIn('--ios', run.call_args.args[0])

  def test_the_built_gadget_is_read_from_its_config(self):
    tmp = Path(tempfile.mkdtemp())
    config = tmp / 'configs' / 'c.1'
    config.mkdir(parents=True)
    (config / 'ffs.jetlink').touch()
    with unittest.mock.patch.object(gadget, 'GADGET_PATH', tmp):
      self.assertFalse(gadget.built_for_ios())
      (config / 'ncm.usb0').touch()
      self.assertTrue(gadget.built_for_ios())
