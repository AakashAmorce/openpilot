"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

sunnypilot's modeld_tinygrad joins the accelerator the way stock modeld does.

The small model is whatever bundle the user picked, and a custom bundle runs
on modeld_v2, so the link has to be reachable from that process too. This
reads modeld_v2/modeld.py rather than importing it (that costs tinygrad and a
vision stream) and pins the same seam test_native_equivalence pins for stock
modeld: the two calls, the decision before the process goes realtime, the
fallback opening with the re-raise, and the modelDataV2SP field the UI reads.
"""
import ast
import re
import unittest
from pathlib import Path

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.accelerators.tests.test_native_equivalence import (ACCELERATOR_CALLS, MODELD, _accelerator_calls, _assigns,
                                                                             _assert_chestnut_blocks_ignore_the_accelerator, _index,
                                                                             _parse, _prepares, _realtime_index, _tests_name)

MODELD_V2 = Path(__file__).resolve().parents[2] / 'modeld_v2' / 'modeld.py'
# what the loop reads off the model, which the joining state proxies per model
FACE = ('constants', 'desire_key', 'numpy_inputs', 'LAT_SMOOTH_SECONDS', 'LONG_SMOOTH_SECONDS',
        'PLANPLUS_CONTROL', 'get_action_from_model')


class ModeldV2Seam(OpenpilotTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.src = MODELD_V2.read_text()
    cls.tree = ast.parse(cls.src)
    cls.body = _parse(cls.src)

  def test_the_module_is_reachable_from_the_same_two_calls(self):
    calls = _accelerator_calls(self.tree)
    self.assertEqual({attr for _, attr in calls}, ACCELERATOR_CALLS, f"the seam differs from stock modeld's: {calls}")
    self.assertEqual(len(calls), 2, calls)

  def test_both_modelds_join_the_accelerator_with_the_same_lines(self):
    # the accelerator's end of the join lives behind load(); what is left in
    # modeld must not drift apart between the two copies
    def lines(src):
      return [line.strip() for line in src.splitlines() if 'accelerator' in line.lower()]
    self.assertEqual(lines(self.src), lines(MODELD.read_text()))

  def test_the_link_is_decided_before_the_process_goes_realtime(self):
    # prepare() starts tinygrad's device thread; after config_realtime_process
    # it would inherit SCHED_FIFO 54 on core 7 and preempt the frame loop
    decide = _index(self.body, _prepares, 'the accelerators.prepare() call')
    realtime = _realtime_index(self.body)
    chestnut = _index(self.body, lambda s: _assigns(s, 'CHESTNUT'), 'the CHESTNUT assignment')
    self.assertLess(chestnut, decide)
    self.assertLess(decide, realtime, "accelerators.prepare() moved after config_realtime_process")

  def test_a_fitted_chestnut_never_reaches_the_accelerator(self):
    _assert_chestnut_blocks_ignore_the_accelerator(self, self.tree)

  def test_the_fallback_opens_with_the_re_raise(self):
    handler = next(h for n in self.body for x in ast.walk(n) if isinstance(x, ast.Try)
                   for h in x.handlers if any('ChestnutActive' in ast.dump(s) for s in ast.walk(h)))
    guard = handler.body[0]
    self.assertTrue(_tests_name(guard, 'accelerator') and isinstance(guard.body[0], ast.Raise),
                    "the fallback no longer opens with `if accelerator: raise`")

  def test_the_ui_field_is_published(self):
    self.assertIn('modelDataV2SP.acceleratorState', self.src, "modeld_v2 no longer publishes acceleratorState")

  def test_the_loop_reads_the_per_model_face_off_the_model(self):
    for attr in FACE:
      self.assertIn(f'model.{attr}', self.src, f"the loop no longer reads model.{attr}")

  def test_the_joining_state_has_all_either_modeld_reads_off_the_model(self):
    # read out of both modelds rather than kept by hand: a missing attribute is
    # an AttributeError on the frame loop. `model` is the loop's; load_big's `m`
    # is a chestnut's and never the joining state
    from openpilot.sunnypilot.accelerators.jetlink.joining import JoiningModelState
    for path in (MODELD, MODELD_V2):
      names = set(re.findall(r'\bmodel\.([a-zA-Z_][a-zA-Z0-9_]*)', path.read_text()))
      self.assertIn('run', names, f"{path.name}: this no longer finds what the loop reads off the model")
      missing = sorted(n for n in names if not hasattr(JoiningModelState, n))
      self.assertEqual(missing, [], f"JoiningModelState is missing {missing}, which {path.parent.name}/{path.name} reads")


if __name__ == '__main__':
  unittest.main()
