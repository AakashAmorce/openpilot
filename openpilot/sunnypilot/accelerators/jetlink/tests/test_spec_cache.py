"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The spec record: the shapes the server answered with, and whether the engine
for the sha it names is built. One param, so the two cannot name different
models.
"""
import unittest

from jetlink.spec import ModelSpec

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.accelerators.jetlink import spec_cache


def spec(sha256: str) -> ModelSpec:
  return ModelSpec(sha256=sha256, nbytes=4096, frame_skip=4, input_shapes={'features_buffer': (1, 24, 512)},
                   output_shapes={'outputs': (1, 16)}, output_slices={'plan': slice(0, 16)}, checkpoint=None)


class TestSpecRecord(OpenpilotTestCase):
  def test_a_stored_spec_is_ready_for_its_own_model_only(self):
    spec_cache.store(spec('a' * 64))
    self.assertTrue(spec_cache.engine_ready_for('a' * 64))
    self.assertFalse(spec_cache.engine_ready_for('b' * 64))
    self.assertFalse(spec_cache.engine_ready_for(None))
    self.assertEqual(spec_cache.load().sha256, 'a' * 64)

  def test_nothing_recorded_is_not_ready(self):
    self.assertIsNone(spec_cache.load())
    self.assertFalse(spec_cache.engine_ready_for('a' * 64))

  def test_clearing_keeps_the_spec(self):
    # it still sizes the warp; only the engine has to be asked for again
    spec_cache.store(spec('a' * 64))
    spec_cache.clear_ready()
    self.assertFalse(spec_cache.engine_ready_for('a' * 64))
    self.assertEqual(spec_cache.load(), spec('a' * 64))
    spec_cache.store(spec('a' * 64))
    self.assertTrue(spec_cache.engine_ready_for('a' * 64))


if __name__ == '__main__':
  unittest.main()
