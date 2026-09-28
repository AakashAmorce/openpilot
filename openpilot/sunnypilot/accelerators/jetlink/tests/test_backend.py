"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Holding the gadget while the Jetson comes up.

The comma is the USB device: the link exists only while a process holds ep0
with the UDC bound, and every unbind is an unplug the far end has to recover
from. This module is about not doing that and about modeld borrowing the
endpoints rather than the gadget. The one case that still needs an edge, the
bounce in wait_for_host, is jetlink.comma's and tested there.
"""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from jetlink.comma import gadget

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.accelerators.jetlink import backend


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


class ClockedTest(OpenpilotTestCase):
  def setUp(self):
    self.clock = FakeClock()
    for module in (backend, gadget):
      p = mock.patch.object(module, 'time', self.clock)
      self.addCleanup(p.stop)
      p.start()
    # USB unless a test says otherwise, whatever a previous owner's record holds
    p = mock.patch.object(gadget, 'LINK', Path(tempfile.mkdtemp()) / 'link')
    self.addCleanup(p.stop)
    p.start()

  def bus(self, udc: str, cc: bool = True):
    for name, value in (('udc_state', udc), ('port_has_host', cc)):
      p = mock.patch.object(gadget, name, return_value=value)
      self.addCleanup(p.stop)
      p.start()


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

  def test_endpoints_that_will_not_open_are_still_reported(self):
    # the owner lent them, but a provisioning run is still finishing an
    # exchange on them: there is no link to hold on to and the join loop
    # should hear why
    from jetlink.comma import lending
    self.link.client = None
    with mock.patch.object(lending, 'borrow', return_value=mock.Mock(closed=False)), \
         mock.patch.object(backend.helpers, 'connect', side_effect=OSError('ep0 busy')), \
         self.assertRaisesRegex(OSError, 'ep0 busy'):
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


class BorrowingTheGadget(OpenpilotTestCase):
  """modeld does not bring the gadget up any more.

  jetlinkd holds ep0 and the bind for as long as the link is enabled, so the
  comma stays enumerated across the ignition edge; modeld asks for the endpoint
  files and gives them back by exiting. It never opens the gadget itself: only
  the owner holds ep0.
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
    fresh = mock.Mock(closed=False)
    with mock.patch.object(self.lending, 'borrow', return_value=fresh) as borrow, \
         mock.patch.object(backend.helpers, 'connect') as connect:
      self.link.open()
    borrow.assert_called_once()
    assert connect.call_args.kwargs['loan'] is fresh

  def test_a_lease_that_ended_is_asked_for_again(self):
    with mock.patch.object(self.lending, 'borrow', return_value=mock.Mock(closed=True)) as borrow, \
         mock.patch.object(backend.helpers, 'connect'):
      self.link.open()
      self.link.client = None
      self.link.open()
    assert borrow.call_count == 2

  def test_no_loan_is_no_join_and_no_gadget_of_our_own(self):
    # nobody lent the link: the small model drives and the join loop asks
    # again. Opening the endpoints here would be a second owner of ep0
    with mock.patch.object(self.lending, 'borrow', return_value=None), \
         mock.patch('jetlink.client.JetlinkClient') as client:
      with self.assertRaises(TimeoutError):
        self.link.open()
    assert client.method_calls == []
    assert self.link.client is None and self.link.loan is None

  def test_the_early_present_does_not_spend_its_whole_budget_asking(self):
    # PRESENT_TIMEOUT blocks modeld's main thread, and a borrow that outlasts
    # it leaves nothing to open the link with
    with mock.patch.object(self.lending, 'borrow', return_value=mock.Mock(closed=False)) as borrow, \
         mock.patch.object(backend.helpers, 'connect'):
      self.link.open(deadline=backend.time.monotonic() + 1.5)
    assert borrow.call_args.kwargs['timeout'] <= 1.5


class PresentingEarly(OpenpilotTestCase):
  """The load takes the link from a helper thread and waits PRESENT_TIMEOUT
  for it at most; a helper that finishes later closes what it opened."""

  def setUp(self):
    from jetlink.comma import lending
    for target, name, value in ((lending, 'borrow', mock.Mock(return_value=mock.Mock(closed=False))),
                                (backend, 'PRESENT_TIMEOUT', 0.05)):
      p = mock.patch.object(target, name, value)
      self.addCleanup(p.stop)
      p.start()
    self.link = backend._Link()
    self.client = mock.Mock(dead=False)

  def test_a_link_that_opens_in_time_is_kept(self):
    with mock.patch.object(backend.helpers, 'connect', return_value=self.client):
      backend._present_early(self.link)
    self.assertIs(self.link.client, self.client)
    self.client.close.assert_not_called()

  def test_a_link_that_opens_late_is_closed(self):
    release = threading.Event()
    self.addCleanup(release.set)

    def slow(*args, **kwargs):
      release.wait(5)
      return self.client

    with mock.patch.object(backend.helpers, 'connect', side_effect=slow):
      t0 = time.monotonic()
      backend._present_early(self.link)
      self.assertLess(time.monotonic() - t0, 2.0, 'the load waited on a helper past its budget')
      release.set()
      for _ in range(200):
        if self.client.close.called:
          break
        time.sleep(0.01)
    self.client.close.assert_called_once()


class ShuttingTheJetsonDown(OpenpilotTestCase):
  """hardwared hands the request to the owner only when a Jetson is there to take it."""

  def shutdown(self, enabled=True, present=True, requested=True, taken=True):
    with mock.patch.object(gadget, 'enabled', return_value=enabled), \
         mock.patch.object(backend.helpers, 'gadget_present', return_value=present), \
         mock.patch.object(gadget, 'request_shutdown', return_value=requested) as request, \
         mock.patch.object(backend.helpers, 'await_shutdown', return_value=taken) as wait:
      backend.shutdown('car battery', timeout=3.0)
    return request, wait

  def test_the_link_off_asks_nothing(self):
    request, wait = self.shutdown(enabled=False)
    request.assert_not_called()
    wait.assert_not_called()

  def test_no_jetson_there_asks_nothing(self):
    request, wait = self.shutdown(present=False)
    request.assert_not_called()
    wait.assert_not_called()

  def test_a_jetson_there_is_asked_and_waited_for(self):
    request, wait = self.shutdown()
    request.assert_called_once_with('car battery')
    wait.assert_called_once_with(3.0)

  def test_a_request_that_could_not_be_written_is_not_waited_on(self):
    request, wait = self.shutdown(requested=False)
    request.assert_called_once_with('car battery')
    wait.assert_not_called()


class BuildingOnroad(OpenpilotTestCase):
  """The picked model is built with the small model driving.

  a provisioning run works offroad only, so a model picked in the driveway and
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
    for module, name, value in ((backend.helpers, 'selected_model', dict(self.ENTRY)),
                                (backend.helpers, 'shipped_model_path', None),
                                (backend.spec_cache, 'engine_ready_for', False)):
      p = mock.patch.object(module, name, return_value=value)
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
    with mock.patch.object(backend.spec_cache, 'clear_ready') as cleared:
      with self.assertRaises(EngineMissing):
        backend._open_link(self.link)
    cleared.assert_called_once_with()
    self.client.close.assert_called_once()


if __name__ == '__main__':
  unittest.main()
