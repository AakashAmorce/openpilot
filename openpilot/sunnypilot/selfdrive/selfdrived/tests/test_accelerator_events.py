from openpilot.cereal import custom, messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.selfdrived.events import Events, EventName, ET
from openpilot.selfdrive.selfdrived.state import SOFT_DISABLE_TIME, State, StateMachine
from openpilot.sunnypilot.selfdrive.selfdrived.accelerator_events import AcceleratorEvents
from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP, EventsSP

EventNameSP = custom.OnroadEventSP.EventName


class TestAcceleratorEvents(OpenpilotTestCase):
  """The adapter alone; the drives through SelfdriveD are traced in
  test_selfdrived_traces.py beside this one."""

  def setUp(self):
    super().setUp()
    self.sm = messaging.SubMaster(['modelV2', 'modelDataV2SP'])
    self.events = Events()
    self.events_sp = EventsSP()
    self.accel = AcceleratorEvents()
    for service in self.sm.services:
      self.sm.data[service] = self.sm[service].as_builder()
      self.sm.seen[service] = self.sm.alive[service] = self.sm.valid[service] = True

  def step(self, state='none', big=False, alive=True, enabled=False, standstill=False):
    self.sm['modelDataV2SP'].acceleratorState = state
    self.sm['modelV2'].big = big
    self.sm.alive['modelV2'] = alive
    self.events.clear()
    self.events_sp.clear()
    self.accel.update(self.sm, enabled, standstill, self.events, self.events_sp)
    return (EventName.bigModelLoading in self.events.names, EventName.bigModelFailed in self.events.names,
            EventNameSP.bigModelLinkLost in self.events_sp.names)

  def test_event_ordinals(self):
    # logs store the ordinal, and 28-31 shipped before these existed
    self.assertEqual(int(EventNameSP.stockEcuReady), 31)
    self.assertEqual(int(EventNameSP.bigModelAvailable), 32)
    self.assertEqual(int(EventNameSP.bigModelLinkLost), 33)

  def test_joining_blocks_only_while_modelv2_is_held_back(self):
    self.assertEqual(self.step(state='joining', alive=False), (True, False, False))
    self.assertEqual(self.step(state='joining', alive=False), (True, False, False))
    # a late join onto a publishing modelV2 never keeps the driver out
    self.assertEqual(self.step(state='joining', alive=True), (False, False, False))
    self.assertEqual(self.step(state='none', alive=False), (False, False, False))
    self.assertEqual(self.step(state='running', alive=False), (False, False, False))

  def test_link_lost_while_engaged_latches_until_disengage(self):
    self.assertEqual(self.step(state='running', big=True, enabled=True), (False, False, False))
    # both events, every tick: the native one drives the main state machine's
    # soft disable, the sunnypilot one drives MADS and carries the guidance
    for _ in range(10):
      self.assertEqual(self.step(state='retrying', big=False, enabled=True), (False, True, True))
    # a rejoin while still engaged does not clear it; only a disengage does
    self.assertEqual(self.step(state='running', big=True, enabled=True), (False, True, True))
    self.assertEqual(self.step(state='running', big=True, enabled=False), (False, False, False))
    # and the latch rearms for the next fall
    self.assertEqual(self.step(state='running', big=True, enabled=True), (False, False, False))
    self.assertEqual(self.step(state='unavailable', big=False, enabled=True), (False, True, True))

  def test_link_lost_while_disengaged_is_silent(self):
    self.assertEqual(self.step(state='running', big=True, enabled=False), (False, False, False))
    self.assertEqual(self.step(state='retrying', big=False, enabled=False), (False, False, False))
    # the fall was consumed: engaging afterwards does not replay it
    self.assertEqual(self.step(state='retrying', big=False, enabled=True), (False, False, False))

  def test_link_lost_ignores_a_chestnut_fall(self):
    # comma's board has the native bigModelFailed for this; the adapter must not double it
    self.assertEqual(self.step(state='none', big=True, enabled=True), (False, False, False))
    self.assertEqual(self.step(state='none', big=False, enabled=True), (False, False, False))

  def drive_state_machine(self, latched: bool) -> State:
    """ENABLED, then the adapter's events fed to the real state machine for SOFT_DISABLE_TIME."""
    machine = StateMachine()
    machine.state = State.enabled
    self.step(state='running', big=True, enabled=True)
    for tick in range(int(SOFT_DISABLE_TIME / DT_CTRL) + 2):
      self.step(state='retrying', big=False, enabled=True)
      if not latched and tick > 0:
        # what a one-tick event looks like to the machine on every tick after the first
        self.events.clear()
        self.events_sp.clear()
      enabled, _ = machine.update(self.events)
      if not enabled:
        break
    return machine.state

  def test_a_latched_loss_takes_the_real_state_machine_to_disabled(self):
    self.assertEqual(self.drive_state_machine(latched=True), State.disabled)

  def test_a_one_tick_loss_would_not_have(self):
    # SOFT_DISABLING returns to ENABLED the tick its SOFT_DISABLE event goes
    # away, which is why the loss has to be latched rather than edge-raised.
    self.assertEqual(self.drive_state_machine(latched=False), State.enabled)

  def test_link_lost_is_a_soft_disable_that_does_not_ask_for_a_restart(self):
    alerts = EVENTS_SP[EventNameSP.bigModelLinkLost]
    self.assertEqual(set(alerts), {ET.SOFT_DISABLE, ET.PERMANENT})
    self.assertEqual(alerts[ET.PERMANENT].alert_text_2, 'Small model is driving,\nreconnecting if it comes back')


class TestOptionalProcesses(OpenpilotTestCase):
  def test_a_dead_link_owner_never_blocks_engagement(self):
    # manager does not restart a process that died, and selfdrived's
    # processNotRunning is NO_ENTRY: the accelerator's daemon has to be one
    # selfdrived ignores, or losing it costs the drive instead of the big model
    from openpilot.sunnypilot import jetlink_adapter
    from openpilot.system.manager.process_config import managed_processes
    self.assertEqual(AcceleratorEvents.OPTIONAL_PROCESSES, {jetlink_adapter.OWNER})
    self.assertIn('jetlinkd', managed_processes)
