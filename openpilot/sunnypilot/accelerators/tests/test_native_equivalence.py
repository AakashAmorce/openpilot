"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

A device with comma's board fitted must behave exactly as it does on develop.

The upstreaming plan rests on one claim: chestnut is native, at its own lines,
and jetlink is five sites beside it a chestnut device never reaches. This reads
modeld.py rather than importing it (that costs tinygrad, usb1 and a vision
stream) and runs the decide, load and status-publisher statements verbatim
under fakes.

Two joins that must not happen: a fitted board with PCIe trained is never
prepared, because `if not CHESTNUT` guards the call, so its load() answers
None; no board and the link off stops at prepare(), which answers from the
toggle and opens no link. The rest pins the footprint: chestnut statements
are develop's byte for byte, and the module is reachable from two call sites
in four hunks.
"""
import ast
import subprocess
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from jetlink.comma import gadget

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot import accelerators

MODELD = Path(__file__).resolve().parents[3] / 'selfdrive' / 'modeld' / 'modeld.py'

# the zoompilot parent carries comma's chestnut unmodified, so it is the
# baseline rather than any older upstream tag
BASELINE = 'develop'

# everything modeld may call on the module: prepare() before the process goes
# realtime, load() once the camera is up. A name added here without a plan
# entry is a widened seam
ACCELERATOR_CALLS = {'prepare', 'load'}


def _parse(src: str) -> list[ast.stmt]:
  tree = ast.parse(src)
  main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
  return main.body


def _assigns(stmt: ast.stmt, name: str) -> bool:
  return isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)


def _index(body: list[ast.stmt], pred, what: str) -> int:
  for i, stmt in enumerate(body):
    if pred(stmt):
      return i
  raise AssertionError(f"modeld.py main() no longer has {what}; this test is describing a file that moved on")


def _tests_name(stmt: ast.stmt, name: str) -> bool:
  return isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == name


def _prepares(stmt: ast.stmt) -> bool:
  """The statement that asks accelerators.prepare()."""
  return any(isinstance(n, ast.Attribute) and n.attr == 'prepare' and isinstance(n.value, ast.Name)
             and n.value.id == 'accelerators' for n in ast.walk(stmt))


def _accelerator_calls(tree: ast.AST) -> list[tuple[int, str]]:
  """Every `accelerators.<attr>` in the module, as (line, attr)."""
  return [(n.lineno, n.attr) for n in ast.walk(tree)
          if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'accelerators']


def _realtime_index(body: list[ast.stmt]) -> int:
  return _index(body, lambda s: isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                and isinstance(s.value.func, ast.Name) and s.value.func.id == 'config_realtime_process',
                'the config_realtime_process call')


def _assert_chestnut_blocks_ignore_the_accelerator(case: unittest.TestCase, tree: ast.AST) -> None:
  for stmt in ast.walk(tree):
    if not _tests_name(stmt, 'CHESTNUT'):
      continue
    # the whole statement: the accelerator loads after the chestnut load, not
    # as its `elif`
    names = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    case.assertNotIn('accelerators', names, f"an `if CHESTNUT:` block at line {stmt.lineno} reaches the accelerator module")
    case.assertNotIn('accelerator', names, f"an `if CHESTNUT:` block at line {stmt.lineno} reads the loaded accelerator")


def _run(src: str, stmts: list[ast.stmt]) -> str:
  """The verbatim source of a run of statements, dedented so it can be exec'd."""
  lines = src.splitlines()
  return textwrap.dedent('\n'.join(lines[stmts[0].lineno - 1:stmts[-1].end_lineno]))


def _git_show(ref: str, path: str) -> str | None:
  """The file at a ref, or None off a checkout that has it. Never raises."""
  try:
    out = subprocess.run(['git', 'show', f'{ref}:{path}'], cwd=MODELD.parents[3],
                         capture_output=True, text=True, timeout=30, check=False)
  except (OSError, subprocess.SubprocessError):
    return None
  return out.stdout if out.returncode == 0 else None


class FakeAccelerators:
  """The accelerators module as modeld sees it, recording every call.

  prepare() answers the most permissive thing it can by default, so a call that
  should never have happened fails twice over: on the recording, and on an
  accelerator joining where the plan says a chestnut owns the drive. load()
  joins only after a yes from prepare(), as the backend's does.
  """

  def __init__(self, prepare=None):
    self.calls: list[str] = []
    self._prepare = prepare if prepare is not None else (lambda: True)
    self.prepared = False
    self.small = None

  def prepare(self) -> bool:
    self.calls.append('prepare')
    self.prepared = self._prepare()
    return self.prepared

  def load(self, cam_w, cam_h, small):
    self.calls.append('load')
    self.small = small
    if not self.prepared:
      return None
    model = SimpleNamespace(chestnut=True, big_model_available=True, big_model_state='joining')
    return accelerators.Accelerator(model, SimpleNamespace(send=lambda *a: None))


class FakeParams:
  """Params for the block under test only. The real ones belong to a car."""

  def __init__(self):
    self.store: dict[str, bool] = {}

  def put_bool(self, key, value):
    self.store[key] = bool(value)

  def get_bool(self, key):
    return bool(self.store.get(key))

  def remove(self, key):
    self.store.pop(key, None)


class FakeMessaging:
  """One chestnutState and then silence, which is the poller wait's fast path."""

  def __init__(self):
    self.polled = 0

  # Capitalised because messaging's is.
  def Poller(self):
    return self

  def poll(self, timeout_ms):
    self.polled += 1
    return [object()] if self.polled == 1 else []

  def sub_sock(self, name, poller=None, conflate=False):
    return object()

  def recv_one_or_none(self, sock):
    return SimpleNamespace(valid=True, chestnutState=object())


class FakeModelState:
  def __init__(self, cam_w, cam_h, chestnut):
    self.chestnut = chestnut

  def warmup(self):
    pass


class ModeldSeam:
  """modeld's own statements, lifted out of main() and runnable with no hardware."""

  def __init__(self):
    self.src = MODELD.read_text()
    self.body = _parse(self.src)

    decide_start = _index(self.body, lambda s: _assigns(s, 'chestnut_available'), 'the chestnut_available assignment')
    self.decide_end = _index(self.body, _prepares, 'the accelerators.prepare() call')
    self.decide = _run(self.src, self.body[decide_start:self.decide_end + 1])

    load_start = _index(self.body, lambda s: _assigns(s, 'model') and isinstance(s.value, ast.Constant) and s.value.value is None,
                        'the `model = None` that opens the load')
    load_end = _index(self.body, lambda s: isinstance(s, ast.Assert) and s.lineno > self.body[load_start].lineno,
                      'the `assert model is not None` that closes the load')
    self.load = _run(self.src, self.body[load_start:load_end + 1])

    pub_start = _index(self.body, lambda s: _assigns(s, 'chestnut_state'), 'the chestnut_state assignment')
    self.publish = _run(self.src, self.body[pub_start:pub_start + 2])

    self.realtime = _realtime_index(self.body)

  def decide_and_load(self, accel, present: bool, compiled: bool, trained: bool) -> dict:
    """Run the three blocks in order, exactly as main() does, and hand back its locals."""
    env: dict[str, str] = {}
    scope = {
      'chestnut_present': lambda: present,
      'chestnut_compiled': lambda: compiled,
      'chestnut_ready': lambda state: trained,
      'messaging': FakeMessaging(),
      # A short deadline: a board that never trains would otherwise wait out
      # deviceState's real period for a message the fake stops sending.
      'SERVICE_LIST': {'deviceState': SimpleNamespace(frequency=20)},
      'time': time,
      'threading': threading,
      'os': SimpleNamespace(environ=env),
      'Params': FakeParams,
      'accelerators': accel,
      'cloudlog': SimpleNamespace(warning=lambda *a, **k: None, exception=lambda *a, **k: None),
      'ModelState': FakeModelState,
      'ChestnutState': lambda pm, chestnut: SimpleNamespace(send=lambda *a: None, big=chestnut),
      'BIG_MODEL_TIMEOUT': 5,
      'vipc_client_main': SimpleNamespace(width=1928, height=1208),
      'pm': object(),
    }
    for block in (self.decide, self.load, self.publish):
      # in a function: the chestnut load declares `nonlocal big_model`. What
      # each block binds becomes a global for the next, which is how CHESTNUT
      # reaches the load
      wrapped = 'def _block():\n' + textwrap.indent(block, '  ') + '\n  return locals()\n'
      # exec of modeld's own source is the point of this file.
      exec(compile(wrapped, str(MODELD), 'exec'), scope)
      scope.update(scope.pop('_block')())
    scope['environ'] = env
    return scope


class NativeEquivalence(OpenpilotTestCase):
  """What a device gets asked, for the two configurations that must ask nothing."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.seam = ModeldSeam()

  def test_a_trained_chestnut_is_never_prepared(self):
    accel = FakeAccelerators()
    scope = self.seam.decide_and_load(accel, present=True, compiled=True, trained=True)

    self.assertTrue(scope['CHESTNUT'])
    # prepare() is the only call that does anything, and `if not CHESTNUT`
    # guards it; load() then answers None, as the backend's does unprepared
    self.assertEqual(accel.calls, ['load'], "a fitted chestnut was prepared for the accelerator")
    self.assertIsNone(scope['accelerator'])
    self.assertEqual(scope['environ'].get('HCQDEV_WAIT_TIMEOUT_MS'), '3000')
    self.assertTrue(scope['model'].chestnut)
    self.assertIsNotNone(scope['chestnut_state'])

  def test_no_board_and_the_link_off_stops_at_prepare(self):
    # the real module with the link off: prepare() reads one param and nothing
    # opens a gadget
    Params().put(accelerators.LINK_PARAM, accelerators.LINK_MODES.index('off'), block=True)
    self.assertFalse(accelerators.enabled())

    accel = FakeAccelerators(prepare=accelerators.prepare)
    with mock.patch.object(gadget, 'link_configured') as link_configured:
      scope = self.seam.decide_and_load(accel, present=False, compiled=False, trained=False)
    link_configured.assert_not_called()

    self.assertFalse(scope['CHESTNUT'])
    self.assertEqual(accel.calls, ['prepare', 'load'])
    self.assertIsNone(scope['accelerator'], "the disabled link joined")
    # The small model drives, and nothing publishes accelerator status.
    self.assertFalse(scope['model'].chestnut)
    self.assertIsNone(scope['chestnut_state'])

  def test_a_board_that_never_trains_falls_through_to_the_question(self):
    # a chestnut device does ask, once, when the board is fitted but PCIe never
    # trains inside the poller wait: CHESTNUT is false, same as a bare device
    accel = FakeAccelerators(prepare=lambda: False)
    scope = self.seam.decide_and_load(accel, present=True, compiled=True, trained=False)

    self.assertFalse(scope['CHESTNUT'])
    self.assertEqual(accel.calls, ['prepare', 'load'])
    self.assertIsNone(scope['accelerator'])

  def test_a_prepared_link_is_one_load(self):
    # no board and the link on: the small model is built as develop builds it
    # and handed over, and what load() answers is everything modeld keeps
    accel = FakeAccelerators()
    scope = self.seam.decide_and_load(accel, present=False, compiled=False, trained=False)

    self.assertFalse(scope['CHESTNUT'])
    self.assertEqual(accel.calls, ['prepare', 'load'])
    self.assertIsNotNone(accel.small)
    self.assertFalse(accel.small.chestnut)
    self.assertIs(scope['small_model'], accel.small)
    self.assertIs(scope['model'], scope['accelerator'].model)
    self.assertIs(scope['chestnut_state'], scope['accelerator'].status)

  def test_the_link_is_decided_before_the_process_goes_realtime(self):
    # prepare() starts tinygrad's device thread; after config_realtime_process
    # it would inherit SCHED_FIFO 54 on core 7 and preempt the frame loop
    self.assertLess(self.seam.decide_end, self.seam.realtime,
                    "accelerators.prepare() moved after config_realtime_process")


class UpstreamFootprint(OpenpilotTestCase):
  """modeld.py's jetlink path stays four hunks wide, and chestnut's lines stay develop's."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.src = MODELD.read_text()
    cls.tree = ast.parse(cls.src)
    cls.body = _parse(cls.src)

  def _baseline_body(self) -> list[ast.stmt] | None:
    src = _git_show(BASELINE, 'openpilot/selfdrive/modeld/modeld.py')
    if src is None:
      return None
    self.baseline_src = src
    return _parse(src)

  def _skip_without_baseline(self, body):
    if body is None:
      self.skipTest(f"local only: needs the {BASELINE} branch to compare with, and CI's shallow checkout has none")

  def test_the_module_is_reachable_from_two_calls_in_four_hunks(self):
    lines = self.src.splitlines()
    calls = _accelerator_calls(self.tree)
    sites = [f"  modeld.py:{lineno} {lines[lineno - 1].strip()}" for lineno, _ in calls]

    # the four hunks that name it: decision, load, status publisher and the
    # fallback re-raise. The fifth site, modelDataV2SP.acceleratorState, reads
    # the model. Lines closer than a hunk's context are one hunk
    jetlink = sorted({n.lineno for n in ast.walk(self.tree) if isinstance(n, ast.Name) and n.id == 'accelerator'}
                     | {lineno for lineno, _ in calls})
    hunk_starts = [line for i, line in enumerate(jetlink) if i == 0 or line - jetlink[i - 1] > 8]
    detail = '\n'.join(sites + [f"  hunks start at lines {hunk_starts}"])

    self.assertEqual({attr for _, attr in calls}, ACCELERATOR_CALLS, f"the seam widened:\n{detail}")
    self.assertEqual(len(calls), 2, f"expected prepare/load and nothing else:\n{detail}")
    self.assertEqual(len(hunk_starts), 4, f"modeld.py's jetlink path is no longer four hunks:\n{detail}")

  def test_the_chestnut_block_never_mentions_the_accelerator_module(self):
    _assert_chestnut_blocks_ignore_the_accelerator(self, self.tree)

  def test_chestnut_state_is_the_baselines(self):
    body = self._baseline_body()
    self._skip_without_baseline(body)
    ours = next((n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == 'ChestnutState'), None)
    theirs = next((n for n in ast.parse(self.baseline_src).body if isinstance(n, ast.ClassDef) and n.name == 'ChestnutState'), None)
    self.assertIsNotNone(ours, "ChestnutState left modeld.py again")
    self.assertIsNotNone(theirs)
    self.assertEqual(ast.get_source_segment(self.src, ours), ast.get_source_segment(self.baseline_src, theirs),
                     f"class ChestnutState differs from {BASELINE}")

  def test_the_poller_wait_and_the_chestnut_load_are_the_baselines(self):
    body = self._baseline_body()
    self._skip_without_baseline(body)

    def wait(b):
      return next(s for s in b if _tests_name(s, 'chestnut_available'))

    def load(b):
      return next(s for s in b if _tests_name(s, 'CHESTNUT') and len(s.body) > 2)

    def small(b):
      # the chestnut load, then the small model and its fallback, as develop has them
      i = b.index(load(b))
      return b[i:i + 3]

    self.assertEqual(_run(self.src, [wait(self.body)]), _run(self.baseline_src, [wait(body)]),
                     f"the chestnutState poller wait differs from {BASELINE}")
    # the whole statement, orelse included, and the small model after it: the
    # accelerator loads below them rather than as a branch of them
    self.assertEqual(_run(self.src, small(self.body)), _run(self.baseline_src, small(body)),
                     f"the `if CHESTNUT:` load or the small model differs from {BASELINE}")

  def test_the_fallback_is_the_baseline_behind_one_guard(self):
    body = self._baseline_body()
    self._skip_without_baseline(body)

    def handler(b):
      return next(h for n in b for x in ast.walk(n) if isinstance(x, ast.Try)
                  for h in x.handlers if any('ChestnutActive' in ast.dump(s) for s in ast.walk(h)))

    ours = handler(self.body)
    theirs = handler(body)
    guard = ours.body[0]
    self.assertTrue(_tests_name(guard, 'accelerator') and isinstance(guard.body[0], ast.Raise),
                    "the fallback no longer opens with `if accelerator: raise`")
    # everything after the guard is develop's handler: a small-model fault is
    # still fatal and a chestnut still demotes itself
    self.assertEqual(_run(self.src, ours.body[1:]), _run(self.baseline_src, theirs.body),
                     f"the big-model fallback differs from {BASELINE}")


if __name__ == '__main__':
  unittest.main()
