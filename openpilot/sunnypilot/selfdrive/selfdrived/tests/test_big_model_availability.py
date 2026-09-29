from openpilot.cereal import custom, messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.sunnypilot.selfdrive.selfdrived.accelerator_events import AcceleratorEvents
from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP, EventsSP

EventName = custom.OnroadEventSP.EventName


class TestBigModelAvailability(OpenpilotTestCase):
  """The adapter's offer to switch; the drive through SelfdriveD is traced in
  test_selfdrived_traces.py beside this one."""

  def setUp(self):
    super().setUp()
    self.sm = messaging.SubMaster(['modelV2', 'modelDataV2SP'])
    self.events = Events()
    self.events_sp = EventsSP()
    self.accel = AcceleratorEvents()
    for service in self.sm.services:
      self.sm.data[service] = self.sm[service].as_builder()
      self.sm.seen[service] = True
      self.sm.alive[service] = True
      self.sm.valid[service] = True

  def update(self, available=False, big=False, standstill=False):
    # ready is the joining state connected and waiting for a window to switch
    self.sm['modelDataV2SP'].acceleratorState = 'ready' if available else 'none'
    self.sm['modelV2'].big = big
    self.events.clear()
    self.events_sp.clear()
    self.accel.update(self.sm, False, standstill, self.events, self.events_sp)
    return EventName.bigModelAvailable in self.events_sp.names

  def test_late_boot_chimes_once_then_can_rejoin(self):
    self.assertFalse(self.update())
    self.assertTrue(self.update(available=True))
    for _ in range(100):
      self.assertFalse(self.update(available=True))
    self.assertFalse(self.update(big=True))
    self.assertFalse(self.update())  # fallback, waiting to reconnect
    self.assertTrue(self.update(available=True))

  def test_every_stop_repeats_the_offer_while_it_is_still_waiting(self):
    # the swap window only opens at a standstill, so the alert that tells the
    # driver to open it is worth repeating at each one
    self.assertTrue(self.update(available=True))
    self.assertFalse(self.update(available=True))
    self.assertTrue(self.update(available=True, standstill=True))
    for _ in range(50):
      self.assertFalse(self.update(available=True, standstill=True))
    self.assertFalse(self.update(available=True))
    self.assertTrue(self.update(available=True, standstill=True))
    # once it is driving, a stop is not an offer
    self.assertFalse(self.update(big=True, standstill=True))

  def test_chestnut_and_old_messages_do_not_announce_availability(self):
    self.assertEqual(custom.ModelDataV2SP.new_message().acceleratorState, 'none')
    self.assertFalse(self.update())
    self.assertFalse(self.update(big=True))
    self.assertFalse(self.update())

  def test_running_big_suppresses_a_pending_status_from_previous_frame(self):
    self.assertFalse(self.update(available=True, big=True))

  def test_missing_invalid_or_stale_messages_never_announce(self):
    for service in ('modelV2', 'modelDataV2SP'):
      for check in ('seen', 'alive', 'valid'):
        with self.subTest(service=service, check=check):
          checks = getattr(self.sm, check)
          checks[service] = False
          self.assertFalse(self.update(available=True))
          checks[service] = True
    self.assertTrue(self.update(available=True))

  def test_stale_gap_does_not_repeat_chime(self):
    self.assertTrue(self.update(available=True))
    self.sm.alive['modelDataV2SP'] = False
    self.assertFalse(self.update())
    self.sm.alive['modelDataV2SP'] = True
    self.assertFalse(self.update(available=True))
    self.assertFalse(self.update())  # an explicit loss rearms it
    self.assertTrue(self.update(available=True))

  def test_notification_has_no_control_effect(self):
    alerts = EVENTS_SP[EventName.bigModelAvailable]
    self.assertEqual(set(alerts), {ET.PERMANENT})
    self.assertEqual(alerts[ET.PERMANENT].alert_text_2, 'Stop with cruise off,\nor turn lateral off')
