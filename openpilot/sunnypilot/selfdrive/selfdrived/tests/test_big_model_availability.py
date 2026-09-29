from openpilot.cereal import custom, messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.sunnypilot.selfdrive.selfdrived.accelerator_events import AcceleratorEvents, OFFER_TICKS
from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP, EventsSP

EventName = custom.OnroadEventSP.EventName


class TestBigModelAvailability(OpenpilotTestCase):
  """The adapter's offer to switch; the drive through SelfdriveD is traced in
  test_selfdrived_traces.py beside this one.

  The large model swaps in only while nothing is in control (the adapter's
  engagement poller: openpilot or MADS engaged, MADS even with its lateral
  paused). Ready while something is, the driver is told once to re-engage;
  ready while nothing is, it swaps at once and bigModelReady says so."""

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

  def update(self, available=False, big=False, enabled=False, mads=False):
    # ready is the joining state connected and waiting for a window to switch
    self.sm['modelDataV2SP'].acceleratorState = 'ready' if available else ('running' if big else 'none')
    self.sm['modelV2'].big = big
    self.events.clear()
    self.events_sp.clear()
    self.accel.update(self.sm, enabled or mads, self.events, self.events_sp)
    return EventName.bigModelAvailable in self.events_sp.names

  def offered_for(self, **kwargs) -> int:
    """Ticks in a row the offer is raised from here on."""
    for n in range(10 * OFFER_TICKS):
      if not self.update(**kwargs):
        return n
    return 10 * OFFER_TICKS

  def test_ready_while_engaged_offers_once_for_three_seconds(self):
    self.assertFalse(self.update(enabled=True))
    self.assertEqual(self.offered_for(available=True, enabled=True), OFFER_TICKS)
    for _ in range(1000):
      self.assertFalse(self.update(available=True, enabled=True))

  def test_mads_alone_is_in_control(self):
    # steering, or paused at a stop: either way MADS steers again on its own
    self.assertTrue(self.update(available=True, mads=True))

  def test_ready_while_nothing_is_in_control_never_offers(self):
    # the swap happens at once, and bigModelReady is what the driver hears
    for _ in range(100):
      self.assertFalse(self.update(available=True))
    self.assertFalse(self.update(big=True))

  def test_engaging_while_it_still_waits_offers(self):
    # it became ready in the gap before an engagement took the window away
    self.assertFalse(self.update(available=True))
    self.assertTrue(self.update(available=True, enabled=True))

  def test_no_repeat_at_every_stop(self):
    self.assertTrue(self.update(available=True, mads=True))
    for _ in range(2 * OFFER_TICKS):
      self.update(available=True, mads=True)
    # stopping and moving off with MADS on changes nothing: it is engaged throughout
    for _ in range(3):
      self.assertFalse(self.update(available=True, mads=True))

  def test_the_swap_ends_it_at_once(self):
    # 2026-09-29: the offer came back for ~0.9 s after "Big Model Ready" had
    # expired, reading as if the switch had not happened
    self.assertTrue(self.update(available=True, mads=True))
    self.assertTrue(self.update(available=True, mads=True))
    # the driver turns MADS off and it swaps
    self.assertFalse(self.update(big=True))
    for _ in range(OFFER_TICKS):
      self.assertFalse(self.update(big=True))

  def test_on_screen_it_never_outlives_the_swap(self):
    am = AlertManager()
    shown = []
    for frame in range(4 * OFFER_TICKS):
      big = frame >= 50
      self.update(available=not big, big=big, enabled=True)
      am.add_many(frame, self.events_sp.create_alerts([ET.PERMANENT], []))
      am.process_alerts(frame, set())
      shown.append((am.current_alert.alert_text_1, am.current_alert.alert_text_2))
    offer = ("Big Model Ready", "Re-engage to switch")
    self.assertIn(offer, shown[:50])
    self.assertNotIn(offer, shown[int(0.2 / 0.01) + 50:])

  def test_it_rearms_after_the_link_goes_and_comes_back(self):
    self.assertTrue(self.update(available=True, enabled=True))
    self.assertFalse(self.update(big=True, enabled=True))   # swapped at a window, then...
    self.assertFalse(self.update(enabled=True))             # lost, waiting to reconnect
    self.assertTrue(self.update(available=True, enabled=True))

  def test_chestnut_and_old_messages_do_not_announce_availability(self):
    self.assertEqual(custom.ModelDataV2SP.new_message().acceleratorState, 'none')
    self.assertFalse(self.update(enabled=True))
    self.assertFalse(self.update(big=True, enabled=True))
    self.assertFalse(self.update(enabled=True))

  def test_running_big_suppresses_a_pending_status_from_previous_frame(self):
    self.sm['modelDataV2SP'].acceleratorState = 'ready'
    self.sm['modelV2'].big = True
    self.accel.update(self.sm, True, self.events, self.events_sp)
    self.assertNotIn(EventName.bigModelAvailable, self.events_sp.names)

  def test_missing_invalid_or_stale_messages_never_announce(self):
    for service in ('modelV2', 'modelDataV2SP'):
      for check in ('seen', 'alive', 'valid'):
        with self.subTest(service=service, check=check):
          checks = getattr(self.sm, check)
          checks[service] = False
          self.assertFalse(self.update(available=True, enabled=True))
          checks[service] = True
    self.assertTrue(self.update(available=True, enabled=True))

  def test_stale_gap_does_not_repeat_the_offer(self):
    self.assertTrue(self.update(available=True, enabled=True))
    self.accel.offer = 0
    self.sm.alive['modelDataV2SP'] = False
    self.assertFalse(self.update(enabled=True))
    self.sm.alive['modelDataV2SP'] = True
    self.assertFalse(self.update(available=True, enabled=True))
    self.assertFalse(self.update(enabled=True))  # an explicit loss rearms it
    self.assertTrue(self.update(available=True, enabled=True))

  def test_notification_has_no_control_effect(self):
    alerts = EVENTS_SP[EventName.bigModelAvailable]
    self.assertEqual(set(alerts), {ET.PERMANENT})
    alert = alerts[ET.PERMANENT]
    self.assertEqual((alert.alert_text_1, alert.alert_text_2), ('Big Model Ready', 'Re-engage to switch'))
