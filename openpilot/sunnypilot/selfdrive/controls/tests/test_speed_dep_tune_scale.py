"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

A flat torque tune (the CP tune, torqued's global fit, the manual override) on a platform whose
STEER_MAX moves with speed: rescaled to the carcontroller's scale each frame, so it asks for the
counts per m/s^2 a flat 800-count build would, on both sides of the CX-5's 1200 -> 800 step.
"""
import numpy as np
import pytest

from opendbc.car.mazda.values import MazdaFlags
from opendbc.car.structs import car
import openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override as override_module
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext import LatControlTorqueExt
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride
from openpilot.sunnypilot.selfdrive.controls.tests.speed_dep_helpers import FakeParams, activate_speed_dep, make_torque_params

LAF, FRICTION = 1.76, 0.177  # the CX-5 2022's CP tune (MAZDA_CX9_2021 in params.toml), fitted at 800 counts
BELOW, ABOVE = 10.0, 20.0     # m/s, either side of the 14.2 -> 14.5 step


def mazda_cp():
  CP = car.CarParams.new_message()
  CP.brand = 'mazda'
  CP.carFingerprint = 'MAZDA_CX5_2022'
  CP.flags = int(MazdaFlags.GEN1 | MazdaFlags.STEER_TO_ZERO_EPS)
  CP.lateralTuning.init('torque')
  CP.lateralTuning.torque.latAccelFactor = LAF
  CP.lateralTuning.torque.friction = FRICTION
  return CP


@pytest.fixture
def make_mazda_override(monkeypatch):
  def _make(**params):
    fake = FakeParams(**params)
    monkeypatch.setattr(override_module, "Params", lambda: fake)
    return LatControlTorqueExtOverride(mazda_cp())
  return _make


def run(ovr, tp, v):
  ovr._last_vego = v
  return ovr.update_override_torque_params(tp)


def counts_per_lat_accel(tp, steer_max):
  return steer_max / tp.latAccelFactor


class TestFlatTuneScale:
  def test_counts_match_a_flat_800_build_on_both_sides(self, make_mazda_override):
    ovr = make_mazda_override()
    tp = make_torque_params(LAF, friction=FRICTION)
    run(ovr, tp, BELOW)
    assert tp.latAccelFactor == pytest.approx(LAF * 1.5, rel=1e-6)
    assert counts_per_lat_accel(tp, 1200) == pytest.approx(800 / LAF, rel=1e-6)
    assert tp.friction * 1200 == pytest.approx(FRICTION * 800, rel=1e-6)
    run(ovr, tp, ABOVE)
    assert tp.latAccelFactor == pytest.approx(LAF, rel=1e-6)
    assert tp.friction == pytest.approx(FRICTION, rel=1e-6)

  def test_own_write_is_not_compounded(self, make_mazda_override):
    ovr = make_mazda_override()
    tp = make_torque_params(LAF, friction=FRICTION)
    assert run(ovr, tp, BELOW)
    for _ in range(5):
      assert not run(ovr, tp, BELOW)  # float32-exact: no update_limits every frame
    assert tp.latAccelFactor == pytest.approx(LAF * 1.5, rel=1e-6)

  def test_host_write_becomes_the_new_base(self, make_mazda_override):
    # controlsd's update_torque_parameters: torqued's global fit, learned above 15 m/s at 800
    ovr = make_mazda_override()
    tp = make_torque_params(LAF, friction=FRICTION)
    run(ovr, tp, BELOW)
    tp.latAccelFactor, tp.friction = 2.0, 0.15
    assert run(ovr, tp, BELOW)
    assert tp.latAccelFactor == pytest.approx(3.0, rel=1e-6)
    assert tp.friction == pytest.approx(0.1, rel=1e-6)

  def test_manual_override_rides_the_scale_and_hands_back_the_cp_tune(self, make_mazda_override):
    ovr = make_mazda_override(enforce=True, manual_override=True, manual_lat_accel_factor='2.5', manual_friction='0.12')
    tp = make_torque_params(LAF, friction=FRICTION)
    run(ovr, tp, BELOW)
    assert tp.latAccelFactor == pytest.approx(3.75, rel=1e-6)
    assert tp.friction == pytest.approx(0.08, rel=1e-6)
    ovr.torque_override_enabled = False
    ovr.params.manual_override = False
    run(ovr, tp, BELOW)
    assert tp.latAccelFactor == pytest.approx(LAF * 1.5, rel=1e-6)
    assert tp.friction == pytest.approx(FRICTION / 1.5, rel=1e-6)

  def test_speed_bins_own_the_params_when_active(self, make_mazda_override):
    # the bins carry their own per-count scaling; the flat rescale must not stack on top
    ovr = make_mazda_override()
    activate_speed_dep(ovr, speed_bp=[5.0, 30.0], lat_accel_factor_bp=[2.4, 2.4], friction_bp=[0.1, 0.1])
    tp = make_torque_params(LAF, friction=FRICTION)
    run(ovr, tp, BELOW)
    assert tp.latAccelFactor == pytest.approx(2.4, rel=1e-6)

  def test_bins_switched_off_mid_drive_return_to_the_rescaled_cp_tune(self, make_mazda_override):
    ovr = make_mazda_override()
    ovr.lac_torque = SimpleHost(tp := make_torque_params(LAF, friction=FRICTION))
    activate_speed_dep(ovr, speed_bp=[5.0, 30.0], lat_accel_factor_bp=[2.4, 2.4], friction_bp=[0.1, 0.1])
    run(ovr, tp, BELOW)
    ovr.CP = mazda_cp().as_reader()
    LatControlTorqueExt.disable_speed_dep_torque(ovr)
    run(ovr, tp, BELOW)
    assert tp.latAccelFactor == pytest.approx(LAF * 1.5, rel=1e-6)
    assert tp.friction == pytest.approx(FRICTION / 1.5, rel=1e-6)

  def test_rescale_is_continuous_through_the_step(self, make_mazda_override):
    ovr = make_mazda_override()
    tp = make_torque_params(LAF, friction=FRICTION)
    lafs = []
    for v in np.arange(14.0, 14.8, 0.05):
      run(ovr, tp, float(v))
      lafs.append(tp.latAccelFactor)
    assert max(abs(np.diff(lafs))) < 0.2 * LAF


class SimpleHost:
  """The host disable_speed_dep_torque restores: its torque_params and update_limits."""

  def __init__(self, torque_params):
    self.torque_params = torque_params

  def update_limits(self):
    pass
