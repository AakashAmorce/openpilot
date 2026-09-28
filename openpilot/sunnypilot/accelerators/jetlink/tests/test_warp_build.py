"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The warps accelerators/SConscript builds, and that only the build compiles one.

The cameras are modeld/SConscript's, exported to it: a source build's own, or
every one a prebuilt release installs on. Nothing compiles a missing warp at
runtime, so a source build shrugs off a failed compile (the offroad alert
says so) and a release fails on it. The SConscript runs here against a
stand-in for SCons, and the targets it declares are compared with what
load_warp opens.
"""
import ast
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from openpilot.common.basedir import BASEDIR
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
from openpilot.sunnypilot.accelerators.jetlink import warp_cache

SCONSCRIPT = Path(BASEDIR) / 'openpilot' / 'sunnypilot' / 'accelerators' / 'SConscript'
MODEL = (512, 256)
CAMERAS = [(c.width, c.height) for c in (_ar_ox_fisheye, _os_fisheye)]


class FakeNode:
  def __init__(self, root: str, path: str):
    self.abspath = os.path.join(root, path[1:]) if path.startswith('#') else path
    self.relpath = os.path.relpath(self.abspath, root)


class FakeEnv:
  """The parts of an SCons environment the SConscript touches."""

  def __init__(self, root: str):
    self.root = root
    self.commands: list[tuple[str, str]] = []

  def Clone(self):
    return self

  def Dir(self, path: str) -> FakeNode:
    return FakeNode(self.root, path)

  def Command(self, target, source, action):
    self.commands.append((target, action))


def run_sconscript(camera_configs=CAMERAS, prebuilt: bool = False, arch: str = 'comma_arm64') -> dict[str, str]:
  """{target file name: command} for every warp the SConscript declares."""
  with tempfile.TemporaryDirectory() as root:
    # a checkout with the jetlink submodule in it, and the real sources
    (Path(root) / 'openpilot').symlink_to(Path(BASEDIR) / 'openpilot')
    (Path(root) / 'jetlink_repo' / 'jetlink').mkdir(parents=True)
    (Path(root) / 'tinygrad_repo').mkdir()

    env = FakeEnv(root)
    script = types.ModuleType('SCons.Script')
    script.Action = lambda cmd, msg=None: cmd
    scons = types.ModuleType('SCons')
    scons.Script = script
    namespace = {'Import': lambda *names: None, 'env': env, 'arch': arch, 'camera_configs': camera_configs,
                 'Dir': env.Dir, 'File': env.Dir}

    with mock.patch.dict(sys.modules, {'SCons': scons, 'SCons.Script': script}), mock.patch.dict(os.environ):
      os.environ.pop('PREBUILT_ALL_CAMERAS', None)
      if prebuilt:
        os.environ['PREBUILT_ALL_CAMERAS'] = '1'
      exec(compile(SCONSCRIPT.read_text(), str(SCONSCRIPT), 'exec'), namespace)
  return {Path(target).name: cmd for target, cmd in env.commands}


class TestWarpTargets(OpenpilotTestCase):
  def test_every_camera_modeld_builds_for_gets_a_warp(self):
    # the names load_warp opens for each camera
    targets = run_sconscript()
    self.assertEqual(set(targets), {warp_cache.warp_path(w, h, *MODEL).name for w, h in CAMERAS})

  def test_each_command_compiles_the_camera_its_target_names(self):
    for target, cmd in run_sconscript().items():
      cam = target.split('_')[1]
      self.assertIn(f'--camera-resolution {cam} ', cmd)
      self.assertIn(f'--model-size {MODEL[0]}x{MODEL[1]} ', cmd)
      self.assertTrue(cmd.endswith(target), cmd)

  def test_a_failed_compile_fails_a_release_but_not_a_source_build(self):
    # a device building from source stays on the small model and tries again
    # on the next build; a release installs on devices that never build
    for cmd in run_sconscript(CAMERAS[:1]).values():
      self.assertTrue(cmd.startswith('-'), cmd)
    for cmd in run_sconscript(prebuilt=True).values():
      self.assertFalse(cmd.startswith('-'), cmd)

  def test_nothing_is_built_off_the_comma(self):
    self.assertEqual(run_sconscript(arch='Darwin'), {})


class TestOnlyTheBuildCompiles(OpenpilotTestCase):
  def test_no_runtime_module_imports_the_compiler(self):
    """compile_warp.py and compile_modeld are for scons. A runtime import would
    bring the ~9 s compile back to modeld or a provisioning run, where it was lost to
    ignition."""
    pkg = Path(warp_cache.__file__).parent
    for path in sorted(pkg.glob('*.py')):
      if path.name == 'compile_warp.py':
        continue
      for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
          names = [node.module or ''] + [f'{node.module}.{a.name}' for a in node.names]
        elif isinstance(node, ast.Import):
          names = [a.name for a in node.names]
        else:
          continue
        for imported in names:
          self.assertFalse(imported.endswith(('compile_warp', 'compile_modeld')), f"{path.name} imports {imported}")


if __name__ == "__main__":
  unittest.main()
