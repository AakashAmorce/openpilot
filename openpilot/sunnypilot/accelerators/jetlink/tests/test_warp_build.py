"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Which warps accelerators/SConscript builds, and that only the build compiles one.

A prebuilt release is built on a comma four and installed on a 3X too, so it
carries a warp for both cameras; a source build needs only its own. Nothing
compiles a missing one at runtime, so a release without the 3X's warp put
every 3X on the small model. The SConscript runs here against a stand-in for
SCons, and the targets it declares are compared with what load_warp opens.
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
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
from openpilot.sunnypilot.accelerators.jetlink import warp_cache

SCONSCRIPT = Path(BASEDIR) / 'openpilot' / 'sunnypilot' / 'accelerators' / 'SConscript'
MODEL = (512, 256)


class FakeNode:
  def __init__(self, root: str, path: str):
    path = os.path.join(root, path[1:]) if path.startswith('#') else path
    self.abspath = path
    self.relpath = os.path.relpath(path, root)


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


def run_sconscript(device: str, all_cameras: bool = False, arch: str = 'comma_arm64') -> dict[str, str]:
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
    namespace = {'Import': lambda *names: None, 'env': env, 'arch': arch, 'Dir': env.Dir, 'File': env.Dir}

    with mock.patch.dict(sys.modules, {'SCons': scons, 'SCons.Script': script}), \
         mock.patch.dict(os.environ), \
         mock.patch("openpilot.common.hardware.HARDWARE.get_device_type", return_value=device):
      os.environ.pop('PREBUILT_ALL_CAMERAS', None)
      if all_cameras:
        os.environ['PREBUILT_ALL_CAMERAS'] = '1'
      exec(compile(SCONSCRIPT.read_text(), str(SCONSCRIPT), 'exec'), namespace)
  return {Path(target).name: cmd for target, cmd in env.commands}


def name(camera) -> str:
  return warp_cache.warp_path(camera.width, camera.height, *MODEL).name


class TestCameraList(unittest.TestCase):
  def test_a_release_carries_both_cameras(self):
    # built on the comma four runner, installed on the 3X as well. These are
    # the names zoompilot-prebuilt.yaml checks for after the build
    for device in ('mici', 'tizi'):
      with self.subTest(runner=device):
        targets = run_sconscript(device, all_cameras=True)
        self.assertEqual(set(targets), {'warp_1928x1208_512x256_tinygrad.pkl',
                                        'warp_1344x760_512x256_tinygrad.pkl'})

  def test_a_source_build_makes_only_its_own_camera(self):
    for device, camera in (('mici', _os_fisheye), ('tizi', _ar_ox_fisheye)):
      with self.subTest(device=device):
        targets = run_sconscript(device)
        self.assertEqual(list(targets), [name(camera)])
        # and it is the one backend.prepare looks for on that device
        with mock.patch("openpilot.common.hardware.HARDWARE.get_device_type", return_value=device):
          self.assertEqual(name(camera), warp_cache.warp_path(*warp_cache.device_geometry()).name)

  def test_each_command_compiles_the_camera_its_target_names(self):
    for target, cmd in run_sconscript('mici', all_cameras=True).items():
      cam = target.split('_')[1]
      self.assertIn(f'--camera-resolution {cam} ', cmd)
      self.assertIn(f'--model-size {MODEL[0]}x{MODEL[1]} ', cmd)
      self.assertTrue(cmd.endswith(target), cmd)

  def test_a_failed_compile_does_not_fail_a_source_build(self):
    # the release checks its pickles instead; a device building from source
    # stays on the small model and tries again on the next build
    for cmd in run_sconscript('tizi').values():
      self.assertTrue(cmd.startswith('-'), cmd)

  def test_nothing_is_built_off_the_comma(self):
    self.assertEqual(run_sconscript('pc', all_cameras=True, arch='Darwin'), {})


class TestOnlyTheBuildCompiles(unittest.TestCase):
  def test_no_runtime_module_imports_the_compiler(self):
    """compile_warp.py and compile_modeld are for scons. A runtime import would
    bring the ~9 s compile back to modeld or jetlinkd, where it was lost to
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
