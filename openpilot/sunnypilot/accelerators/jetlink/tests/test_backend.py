"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Holding the gadget while the Jetson comes up.

The comma is the USB device: the link exists only while a process holds ep0
with the UDC bound, and every unbind is an unplug the far end has to recover
from. This module is about not doing that, about the single case that still
needs an edge, and about modeld borrowing the endpoints rather than the gadget.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jetlink.comma import gadget

from openpilot.sunnypilot.accelerators.jetlink import backend, helpers


class FakeClock:
  """monotonic and sleep, so a 45 s wait costs no wall clock."""

  def __init__(self, now: float = 1000.0):
    self.now = now
    self.slept = 0.0

  def monotonic(self) -> float:
    return self.now

  def sleep(self, seconds: float) -> None:
    step = max(seconds, 0.01)
    self.now += step
    self.slept += step


class ClockedTest(unittest.TestCase):
  def setUp(self):
    self.clock = FakeClock()
    for module in (backend, helpers, gadget):
      p = mock.patch.object(module, 'time', self.clock)
      self.addCleanup(p.stop)
      p.start()
    # USB unless a test says otherwise, whatever the machine's params or a
    # previous owner's record hold
    for name, value in (('link_endpoint', mock.Mock(return_value=None)),
                        ('LINK', Path(tempfile.mkdtemp()) / 'link')):
      p = mock.patch.object(gadget, name, value)
      self.addCleanup(p.stop)
      p.start()

  def bus(self, udc: str, cc: bool = True):
    # gadget, not helpers: the primitives live there and everything that reads
    # them resolves them there, helpers included through its forward
    for name, value in (('udc_state', udc), ('port_has_host', cc)):
      p = mock.patch.object(gadget, name, return_value=value)
      self.addCleanup(p.stop)
      p.start()


class WaitForHost(ClockedTest):
  def setUp(self):
    super().setUp()
    self.bounced = []

  def wait(self, seconds: float = backend.CONNECT_TIMEOUT) -> bool:
    return helpers.wait_for_host(seconds, bounce=lambda: self.bounced.append(True))

  def test_a_host_that_is_already_there_is_not_waited_for(self):
    self.bus('configured')
    assert self.wait() is True
    assert self.clock.slept == 0.0

  def test_a_bus_that_stalls_half_enumerated_is_bounced_once(self):
    # A jetson whose hubs are not armed for remote wakeup answers the bind
    # with a bus reset and stops there; only another connect moves it.
    self.bus('default')
    assert self.wait() is False
    assert len(self.bounced) == 1

  def test_the_bounce_waits_out_a_normal_enumeration(self):
    self.bus('addressed')
    self.wait(helpers.STALLED_ENUMERATION / 2)
    assert self.bounced == []

  def test_nothing_on_the_cable_is_not_a_stall(self):
    # No host on the CC pin: there is nobody to enumerate us and bouncing the
    # gadget would only cost the next one its bind.
    self.bus('not attached', cc=False)
    assert self.wait() is False
    assert self.bounced == []

  def test_a_jetson_still_booting_is_left_alone(self):
    # Powered but not yet driving the bus: the UDC never leaves powered.
    self.bus('powered')
    assert self.wait() is False
    assert self.bounced == []

  def test_a_bounce_that_fails_does_not_end_the_wait(self):
    self.bus('default')
    with mock.patch.object(gadget, 'udc_state', return_value='default'):
      assert helpers.wait_for_host(backend.CONNECT_TIMEOUT,
                                   bounce=mock.Mock(side_effect=OSError('no such device'))) is False

  def test_a_host_that_turns_up_late_is_still_joined(self):
    states = ['powered'] * 3 + ['configured']
    with mock.patch.object(gadget, 'udc_state', side_effect=lambda: states.pop(0) if states else 'configured'), \
         mock.patch.object(gadget, 'port_has_host', return_value=True):
      assert self.wait() is True

  def test_a_caller_that_is_going_away_is_not_kept_waiting(self):
    self.bus('powered')
    assert helpers.wait_for_host(backend.CONNECT_TIMEOUT, should_stop=lambda: True) is False
    assert self.clock.slept == 0.0

  def test_the_wait_says_once_that_it_is_waiting(self):
    self.bus('powered')
    said = []
    helpers.wait_for_host(2.0, report=lambda: said.append(True))
    assert said == [True]

  def test_over_tcp_there_is_no_host_to_wait_for(self):
    # The connect already reached the phone or Jetson; no gadget of ours is
    # bound, so the UDC would never say configured.
    self.bus('not attached', cc=False)
    gadget.link_endpoint.return_value = ('10.0.0.5', 5599)
    said = []
    assert helpers.wait_for_host(backend.CONNECT_TIMEOUT, report=lambda: said.append(True)) is True
    assert self.clock.slept == 0.0
    assert said == []

  def test_a_join_over_tcp_returns_the_client_at_once(self):
    self.bus('not attached', cc=False)
    gadget.link_endpoint.return_value = ('10.0.0.5', 5599)
    link = mock.Mock()
    link.open.return_value = client = mock.Mock()
    assert backend._connect_patiently(link) is client
    assert self.clock.slept == 0.0

  def test_a_join_over_the_cable_never_bounces_the_gadget(self):
    # the phone configured the UDC and its dial is the proof it is there; an
    # unbind would only take its network interface down
    self.bus('configured')
    link = mock.Mock()
    link.open.return_value = client = mock.Mock()
    with mock.patch.object(gadget, 'link_kind', return_value='cable'):
      assert backend._connect_patiently(link) is client
    client.rebind.assert_not_called()
    assert self.clock.slept == 0.0


class HoldingTheLink(ClockedTest):
  """A link the attempt could not use stays open for the next one.

  Closing it unbinds the UDC, and while a Jetson boots the join loop asks
  again every few seconds: that was an unplug every cycle, and one of them
  landed on the enumeration.
  """

  def setUp(self):
    super().setUp()
    self.client = mock.Mock(dead=False)
    self.link = backend._Link()
    self.link.client = self.client
    self.bus('powered', cc=False)

  def test_a_link_nobody_enumerated_is_kept(self):
    with self.assertRaises(TimeoutError):
      backend._connect_patiently(self.link)
    assert self.link.client is self.client, 'unbound the gadget between attempts'
    assert self.client.close.call_count == 0

  def test_a_gadget_that_will_not_open_is_still_reported(self):
    # jetlinkd still finishing an exchange, or a gadget boot never created:
    # there is no link to hold on to and the join loop should hear why
    self.link.client = None
    with mock.patch.object(backend.helpers, 'connect', side_effect=OSError('ep0 busy')):
      with self.assertRaises(OSError):
        backend._connect_patiently(self.link)

  def test_no_model_picked_yet_keeps_the_gadget_presented(self):
    # The panel says what is going on; a closed gadget would take the whole
    # link off the bus for the drive instead.
    with mock.patch.object(backend.helpers, 'selected_model', return_value=None):
      with self.assertRaises(RuntimeError):
        backend._open_link(self.link)
    assert self.link.client is self.client
    assert self.client.close.call_count == 0

  def test_closing_the_link_keeps_the_lease(self):
    # jetlinkd should hold the gadget for the whole drive, however many times
    # the join has to start over
    self.link.loan = mock.Mock(closed=False)
    self.link.close()
    assert self.link.client is None
    self.client.close.assert_called_once()
    assert self.link.loan.close.call_count == 0


class BorrowingTheGadget(unittest.TestCase):
  """modeld does not bring the gadget up any more.

  jetlinkd holds ep0 and the bind for as long as the link is enabled, so the
  comma stays enumerated across the ignition edge; modeld asks for the endpoint
  files and gives them back by exiting.
  """

  def setUp(self):
    from jetlink.comma import lending
    self.lending = lending
    self.link = backend._Link()

  def test_the_lease_is_what_the_link_is_opened_over(self):
    loan = mock.Mock(closed=False)
    with mock.patch.object(self.lending, 'borrow', return_value=loan), \
         mock.patch.object(backend.helpers, 'connect') as connect:
      self.link.open()
    assert connect.call_args.kwargs['loan'] is loan
    assert connect.call_args.kwargs['name'] == 'modeld'

  def test_the_lease_is_borrowed_once_and_renewed_every_attempt(self):
    # a phone may have dialed since the last attempt, or its dial be spent
    loan = mock.Mock(closed=False)
    loan.renew.return_value = True
    with mock.patch.object(self.lending, 'borrow', return_value=loan) as borrow, \
         mock.patch.object(backend.helpers, 'connect') as connect:
      self.link.open()
      self.link.client = None
      self.link.open()
    borrow.assert_called_once()
    loan.renew.assert_called_once()
    assert connect.call_args.kwargs['loan'] is loan

  def test_a_dead_client_is_replaced_not_reused(self):
    # a big model retired after a link loss closes its client; the next
    # attempt reused it and failed on EBADF, a whole retry after every loss
    dead = mock.Mock(dead=True)
    loan = mock.Mock(closed=False)
    loan.renew.return_value = True
    self.link.client, self.link.loan = dead, loan
    with mock.patch.object(backend.helpers, 'connect') as connect:
      assert self.link.open() is connect.return_value
    dead.close.assert_called_once()
    loan.renew.assert_called_once()

  def test_a_renewal_still_on_hold_is_not_an_open_of_our_own(self):
    # the owner is holding for a phone; opening the endpoints here would
    # write a hello to it
    loan = mock.Mock(closed=False)
    loan.renew.return_value = False
    self.link.loan = loan
    with mock.patch.object(self.lending, 'borrow') as borrow, \
         mock.patch.object(backend.helpers, 'connect') as connect:
      with self.assertRaises(TimeoutError):
        self.link.open()
    borrow.assert_not_called()
    connect.assert_not_called()

  def test_a_renewal_that_finds_the_owner_gone_borrows_afresh(self):
    loan = mock.Mock(closed=False)
    def gone(timeout):
      loan.closed = True
      return False
    loan.renew.side_effect = gone
    self.link.loan = loan
    with mock.patch.object(self.lending, 'borrow', return_value=None) as borrow, \
         mock.patch.object(backend.helpers, 'connect') as connect:
      self.link.open()
    borrow.assert_called_once()
    assert connect.call_args.kwargs['loan'] is None

  def test_a_lease_that_ended_is_asked_for_again(self):
    with mock.patch.object(self.lending, 'borrow', return_value=mock.Mock(closed=True)) as borrow, \
         mock.patch.object(backend.helpers, 'connect'):
      self.link.open()
      self.link.client = None
      self.link.open()
    assert borrow.call_count == 2

  def test_no_daemon_to_ask_still_opens_the_gadget(self):
    # the link was only just turned on, or jetlinkd died: a drive must not lose
    # the large model to a daemon fault
    with mock.patch.object(self.lending, 'borrow', return_value=None), \
         mock.patch.object(backend.helpers, 'connect') as connect:
      self.link.open()
    assert connect.call_args.kwargs['loan'] is None

  def test_a_daemon_that_throws_is_not_a_lost_drive(self):
    with mock.patch.object(self.lending, 'borrow', side_effect=OSError('no socket')), \
         mock.patch.object(backend.helpers, 'connect') as connect:
      self.link.open()
    assert connect.call_args.kwargs['loan'] is None

  def test_the_early_present_does_not_spend_its_whole_budget_asking(self):
    # PRESENT_TIMEOUT blocks modeld's main thread, and a borrow that outlasts
    # it leaves nothing to open the link with
    with mock.patch.object(self.lending, 'borrow', return_value=None) as borrow, \
         mock.patch.object(backend.helpers, 'connect'):
      self.link.open(deadline=backend.time.monotonic() + 1.5)
    assert borrow.call_args.kwargs['timeout'] <= 1.5


class BuildingOnroad(unittest.TestCase):
  """The picked model is built with the small model driving.

  jetlinkd provisions offroad only, so a model picked in the driveway and
  driven off on used to cost the whole drive: modeld would not even present
  the gadget, and the panel said a device was on the USB port.
  """

  ENTRY = {'name': 'CTM v2', 'ref': 'f' * 40, 'oid': 'a' * 64, 'size': 766 << 20}

  def setUp(self):
    from openpilot.sunnypilot.accelerators.jetlink import provision
    self.provision = provision
    self.client = mock.Mock()
    self.spec = mock.Mock(sha256=self.ENTRY['oid'])
    self.link = backend._Link()
    p = mock.patch.object(backend, '_connect_patiently', return_value=self.client)
    self.addCleanup(p.stop)
    p.start()
    for name, value in (('selected_model', dict(self.ENTRY)), ('shipped_model_path', None),
                        ('engine_ready_for', False)):
      p = mock.patch.object(backend.helpers, name, return_value=value)
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(provision, 'ensure', return_value=self.spec)
    self.addCleanup(p.stop)
    self.ensure = p.start()

  def test_the_model_the_picker_names_is_what_gets_built(self):
    client, spec = backend._open_link(self.link)
    assert spec is self.spec and client is self.client
    assert self.ensure.call_args.args[1:3] == (self.ENTRY['oid'], self.ENTRY['size'])

  def test_the_frame_deadline_is_set_before_the_link_is_handed_over(self):
    # ensure_engine waits minutes; the frame path must not inherit that
    client, _ = backend._open_link(self.link)
    assert client.deadline == backend.INFERENCE_TIMEOUT

  def test_the_join_thread_can_be_stopped_through_the_build(self):
    stop = object()
    backend._open_link(self.link, should_stop=stop)
    assert self.ensure.call_args.kwargs['should_stop'] is stop

  def test_progress_reaches_the_panel(self):
    backend._open_link(self.link)
    assert self.ensure.call_args.kwargs['progress'] is self.provision.report_with_eta

  def test_bytes_neither_end_has_are_a_parked_job(self):
    # Downloading a gigabyte is the one part of provisioning that needs the
    # internet, and it is not something to start mid-drive.
    from jetlink.client import EngineMissing
    self.ensure.side_effect = EngineMissing('no engine')
    self.link.client = self.client
    with mock.patch.object(backend.helpers, 'set_engine_ready') as cleared:
      with self.assertRaises(EngineMissing):
        backend._open_link(self.link)
    cleared.assert_called_once_with(None)
    self.client.close.assert_called_once()


if __name__ == '__main__':
  unittest.main()
