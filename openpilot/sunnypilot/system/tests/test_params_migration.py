"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.system.params_migration import _migrate_jetlink_link, _migrate_model_bundle_slots


class TestModelBundleSlotMigration(OpenpilotTestCase):
  """Pre-split, a chestnut user's big-model selection lived in the single ActiveBundle.
  The migration seeds both slots; per-source validation later drops whichever does not
  match its own manifest."""

  def test_seeds_chestnut_slot_from_active_bundle(self):
    params = Params()
    bundle = {"ref": "big", "minimumSelectorVersion": 18}
    params.put("ModelManager_ActiveBundle", bundle, block=True)
    _migrate_model_bundle_slots(params)
    assert params.get("ModelManager_ActiveBundleChestnut") == bundle
    assert params.get("ModelManager_ActiveBundle") == bundle

  def test_noop_when_chestnut_slot_already_set(self):
    params = Params()
    params.put("ModelManager_ActiveBundle", {"ref": "small"}, block=True)
    params.put("ModelManager_ActiveBundleChestnut", {"ref": "big"}, block=True)
    _migrate_model_bundle_slots(params)
    assert params.get("ModelManager_ActiveBundleChestnut") == {"ref": "big"}

  def test_noop_when_no_selection(self):
    params = Params()
    _migrate_model_bundle_slots(params)
    assert params.get("ModelManager_ActiveBundleChestnut") is None


class TestJetlinkLinkMigration(OpenpilotTestCase):
  """JetlinkEnabled became Accelerator Link. manager writes JetlinkLink's default
  of off right after the migrations run, so a comma with the toggle on has to be
  moved to USB here or it comes up with the link off."""

  def test_the_toggle_on_is_usb(self):
    from openpilot.sunnypilot.accelerators import LINK_MODES
    params = Params()
    params.put_bool("JetlinkEnabled", True, block=True)
    _migrate_jetlink_link(params)
    assert LINK_MODES[params.get("JetlinkLink")] == "usb"

  def test_the_toggle_off_or_unset_is_left_to_the_default(self):
    params = Params()
    _migrate_jetlink_link(params)
    assert params.get("JetlinkLink") is None
    params.put_bool("JetlinkEnabled", False, block=True)
    _migrate_jetlink_link(params)
    assert params.get("JetlinkLink") is None

  def test_a_setting_already_made_wins(self):
    params = Params()
    params.put_bool("JetlinkEnabled", True, block=True)
    params.put("JetlinkLink", 2, block=True)
    _migrate_jetlink_link(params)
    assert params.get("JetlinkLink") == 2
