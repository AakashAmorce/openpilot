"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from unittest import mock

from openpilot.cereal import custom
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot import model_info
from openpilot.selfdrive.ui.ui_state import ChestnutState
from openpilot.sunnypilot.models.helpers import REQUIRED_JSON_VERSION
from openpilot.sunnypilot.models.model_name import DEFAULT_BIG_MODEL, DEFAULT_MODEL

V3_REF = "bf3e3631b3f91d92a1020a5e0dd4298b93ff4244"


def _raw_bundle(ref: str) -> dict:
  bundle = custom.ModelManagerSP.ModelBundle.new_message()
  bundle.ref = ref
  bundle.minimumSelectorVersion = REQUIRED_JSON_VERSION
  bundle.internalName = ref
  bundle.displayName = ref
  bundle.runner = custom.ModelManagerSP.Runner.tinygrad
  return bundle.to_dict()


class TestCarryingModel(OpenpilotTestCase):
  """What the UI names as driving has to be what manager runs. The small model is
  the model manager's whatever the accelerator link says: the stored qcom bundle
  drives until the accelerator joins, and again if it goes."""

  def setUp(self):
    super().setUp()
    self.ui_state = mock.MagicMock()
    self.ui_state.chestnut_present = False
    self.ui_state.chestnut_state = ChestnutState.DISCONNECTED
    self.ui_state.chestnut_active = False
    self.ui_state.chestnut_loading = False
    self.ui_state.is_offroad.return_value = True
    self.ui_state.params.get.side_effect = lambda key: {"ModelManager_ActiveBundle": _raw_bundle("custom_small")}.get(key)
    patcher = mock.patch.object(model_info, "ui_state", self.ui_state)
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_the_link_on_still_names_the_stored_bundle(self):
    for enabled in (True, False):
      with mock.patch("openpilot.sunnypilot.accelerators.enabled", return_value=enabled):
        assert model_info.carrying_model() == ("qcom", "custom_small", "custom_small")
        source, active, _ = model_info.model_info()
      assert (source, active) == ("qcom", "custom_small")

  def test_an_empty_slot_names_the_default_small_model(self):
    self.ui_state.params.get.side_effect = lambda key: None
    assert model_info.carrying_model() == ("qcom", f"{DEFAULT_MODEL} (Default)", f"{DEFAULT_MODEL} (Default)")

  def test_link_active_names_the_accelerator_model(self):
    self.ui_state.chestnut_state = ChestnutState.ACTIVE
    with mock.patch("openpilot.sunnypilot.accelerators.enabled", return_value=True), \
         mock.patch("openpilot.sunnypilot.accelerators.active_model_name", return_value="big"):
      assert model_info.carrying_model() == ("accelerator", "big", "big")

  def test_fitted_board_is_chestnut_not_jetlink(self):
    # a real chestnut ACTIVE keeps comma's semantics whatever the accelerator says
    self.ui_state.chestnut_present = True
    self.ui_state.chestnut_state = ChestnutState.ACTIVE
    self.ui_state.params.get.side_effect = lambda key: {"ModelManager_ActiveBundleChestnut": _raw_bundle("big_custom")}.get(key)
    with mock.patch("openpilot.sunnypilot.accelerators.active_model_name", return_value="jetlink_big"):
      assert model_info.carrying_model() == ("chestnut", "big_custom", "big_custom")


class TestDefaultBigModelName(OpenpilotTestCase):
  """An empty big slot runs the chestnut's in-tree model when a board is fitted,
  and the accelerator's default when none is: jetlink's, named from its catalog."""

  def setUp(self):
    super().setUp()
    self.ui_state = mock.MagicMock()
    patcher = mock.patch.object(model_info, "ui_state", self.ui_state)
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_a_fitted_chestnut_names_the_in_tree_model(self):
    self.ui_state.chestnut_present = True
    with mock.patch("openpilot.sunnypilot.accelerators.default_big_model_name",
                    side_effect=AssertionError("a chestnut never asks the accelerator")):
      assert model_info.default_model_name("chestnut") == f"{DEFAULT_BIG_MODEL} (Default)"
      assert model_info.default_model_name("qcom") == f"{DEFAULT_MODEL} (Default)"

  def test_without_a_chestnut_the_accelerator_names_its_default(self):
    from openpilot.sunnypilot.accelerators.jetlink import helpers
    self.ui_state.chestnut_present = False
    rows = [{"name": "Cinque Terre V3 Model (September 17, 2026)", "ref": V3_REF, "oid": None, "size": None},
            {"name": "BMRLNAP Model v4 (August 30, 2026)", "ref": "f877d7a0ccc3cce943c76e285214c020cd65c899",
             "oid": None, "size": None}]
    # jetlink's default ref, from the pinned jetlink, not the fork's model_name
    with mock.patch.object(helpers, "_index", return_value=(rows, {})):
      assert model_info.default_model_name("chestnut") == "Cinque Terre V3 Model (Default)"
    assert model_info.default_model_name("qcom") == f"{DEFAULT_MODEL} (Default)"

  def test_an_accelerator_with_no_catalog_falls_back_to_the_in_tree_name(self):
    self.ui_state.chestnut_present = False
    with mock.patch("openpilot.sunnypilot.accelerators.default_big_model_name", return_value=None):
      assert model_info.default_model_name("chestnut") == f"{DEFAULT_BIG_MODEL} (Default)"
