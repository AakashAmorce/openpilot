"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The seam with jetlink.comma, the comma's device layer in the jetlink
submodule: the owner shim manager runs, the openpilot names that layer reads
without importing openpilot, and the API without a jetlink checkout. The
layer's own tests are jetlink's (tests/test_comma_*.py there).
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from jetlink.comma import gadget, owner as comma_owner, port

from openpilot.common.basedir import BASEDIR
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.accelerators.jetlink import owner

ROOT = Path(__file__).resolve().parents[5]
# what the gadget owner must never end up importing. swaglog pulls all three in
# to publish a log line and costs 28 MB; params imports swaglog. Measured on the
# comma: the owner plus the transport is 10.4 MB, against 47.5 MB for the
# daemon that imported the world
HEAVY = ('numpy', 'capnp', 'zmq', 'cereal')


def run_fresh(code: str) -> subprocess.CompletedProcess:
  # this runner's own path, so the jetlink package is found wherever it is
  # checked out; a bare PYTHONPATH found the tree but not the submodule
  env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), *sys.path])}
  return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                        env=env, cwd=str(ROOT), timeout=120)


class TestTheShim(OpenpilotTestCase):
  def test_the_owner_stays_out_of_the_heavy_half(self):
    roots = 'sorted({m.split(".")[0] for m in sys.modules})'
    out = run_fresh(f'import sys, json; import openpilot.sunnypilot.accelerators.jetlink.owner; print(json.dumps({roots}))')
    self.assertEqual(out.returncode, 0, out.stderr)
    found = set(json.loads(out.stdout))
    self.assertEqual(sorted(found & set(HEAVY)), [], 'everything the owner imports runs for the whole drive')

  def test_it_hands_the_owner_the_provisioning_run(self):
    with mock.patch.object(comma_owner, 'main') as main:
      owner.main()
    main.assert_called_once_with([sys.executable, '-m', 'openpilot.sunnypilot.accelerators.jetlink.provision'],
                                 cwd=BASEDIR, env={'PYTHONPATH': BASEDIR})
    self.assertIsNotNone(importlib.util.find_spec(owner.WORKER))

  def test_manager_runs_the_shim_as_jetlinkd_while_the_link_is_on(self):
    from openpilot.sunnypilot import accelerators
    [daemon] = accelerators.daemons()
    self.assertEqual((daemon.name, daemon.module), ('jetlinkd', owner.__name__))
    for enabled in (False, True):
      with mock.patch.object(gadget, 'enabled', return_value=enabled):
        self.assertEqual(daemon.should_run(False, None, None), enabled)


class TestTheNamesTheLayerReads(OpenpilotTestCase):
  """jetlink.comma reads openpilot's params as files and its USB ids as
  constants, since importing either would cost the owner 28 MB."""

  def test_every_param_it_reads_is_declared(self):
    from openpilot.common.params import Params
    params = Params()
    for key in (gadget.P_SPEC, gadget.P_LINK,
                gadget.P_OFFROAD, gadget.P_BIG_MODEL, *comma_owner.WATCHED):
      params.check_key(key)

  def test_the_names_are_the_forks(self):
    from openpilot.sunnypilot.models.helpers import ACTIVE_BUNDLE_KEYS
    self.assertEqual(gadget.P_BIG_MODEL, ACTIVE_BUNDLE_KEYS['chestnut'])

  def test_the_usb_constants_are_the_hardware_modules(self):
    from openpilot.common.hardware.usb import CHESTNUT_ROM_USB_IDS, CHESTNUT_USB_IDS, TYPEC_CC_ORIENTATION_PATH
    self.assertEqual(port.CHESTNUT_IDS, frozenset(CHESTNUT_USB_IDS + CHESTNUT_ROM_USB_IDS))
    self.assertEqual(gadget.CC_ORIENTATION, TYPEC_CC_ORIENTATION_PATH)


class TestWithoutAJetlinkCheckout(OpenpilotTestCase):
  def test_the_api_answers_the_negative_default(self):
    # an empty jetlink_repo: the backend stands on jetlink.comma, so nothing
    # past the API can import, and manager, the UI and hardwared must not care
    code = '''
import sys
sys.modules['jetlink'] = None
from openpilot.sunnypilot import accelerators as a
assert a.LINK_MODES == ('off', 'usb', 'ios') and a.LINK_PARAM == 'JetlinkLink', (a.LINK_MODES, a.LINK_PARAM)
assert a.link_mode() == 'off'
assert a.link_transport() == 'USB'
assert not (a.installed() or a.present() or a.ready() or a.enabled() or a.prepare() or a.extends_catalog())
assert a.unavailable_reason() is None and a.load(1, 1, None) is None
assert a.big_catalog({'bundles': []}) == {'bundles': []}
assert a.selected_model_name() is None and a.active_model_name() is None
a.shutdown('test', timeout=0.1)
assert not a.daemons()[0].should_run(False, None, None)
'''
    out = run_fresh(code)
    self.assertEqual(out.returncode, 0, out.stderr)

  def test_the_setting_is_jetlinks(self):
    # written out in the API so the panels build it without a checkout
    from openpilot.sunnypilot import accelerators
    self.assertEqual((accelerators.LINK_MODES, accelerators.LINK_PARAM), (gadget.LINK_MODES, gadget.P_LINK))
