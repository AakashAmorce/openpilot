"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The comma's USB-C port: when it is held at sink, when it is let go, and what it
leaves alone. A USB-A host, a chestnut and a device without the lever must see
no change at all; a C-to-C host that lost the toss gets one hold, and only for
as long as it is plugged in.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openpilot.sunnypilot.accelerators.jetlink import gadget, usbport

SWAP = usbport.SWAP_AFTER
RELEASE = usbport.RELEASE_AFTER
CHESTNUT = (0xADD1, 0x0001)
CHESTNUT_ROM = (0x174C, 0x2464)
IPHONE = (0x05AC, 0x12A8)
JETSON_GADGET = (0x0955, 0x7020)


class PortTest(unittest.TestCase):
  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    self.role = self.tmp / 'current_pr'
    self.devices = self.tmp / 'devices'
    self.devices.mkdir()
    self.plug('none')
    self.script = mock.Mock(return_value=True)
    for p in (mock.patch.object(usbport, 'POWER_ROLE', self.role),
              mock.patch.object(usbport, 'USB_DEVICES', self.devices),
              mock.patch.object(usbport, 'run_script', self.script)):
      self.addCleanup(p.stop)
      p.start()
    self.port = usbport.Port()
    self.now = 100.0
    # what is always there: the root hub, and the modem on the other controller
    self.enumerate('usb1', (0x1D6B, 0x0002))
    self.enumerate('1-1', (0x2C7C, 0x6007))
    self.enumerate('1-1:1.0', None)

  def plug(self, role: str) -> None:
    self.role.write_text(role + '\n')

  def enumerate(self, name: str, ids: tuple[int, int] | None) -> None:
    """An entry in the fake /sys/bus/usb/devices; an interface has no ids."""
    d = self.devices / name
    d.mkdir()
    if ids is not None:
      (d / 'idVendor').write_text(f'{ids[0]:04x}\n')
      (d / 'idProduct').write_text(f'{ids[1]:04x}\n')

  def run_for(self, seconds: float, usb: bool = True) -> None:
    end = self.now + seconds
    while self.now < end:
      self.port.update(usb, now=self.now)
      self.now += 0.5

  def commands(self) -> list[str]:
    return [c.args[0] for c in self.script.call_args_list]


class TestHosts(PortTest):
  def test_a_usb_a_host_changes_nothing(self):
    self.plug('sink')
    self.run_for(60)
    self.assertEqual(self.commands(), ['off'], "the comma is already the device on an A-to-C cable")

  def test_a_mac_that_lost_the_toss_is_made_the_host(self):
    self.plug('source')
    self.run_for(SWAP - 0.5)
    self.assertEqual(self.commands(), ['off'], "a chestnut gets its chance to enumerate first")
    self.run_for(1)
    self.assertEqual(self.commands(), ['off', 'hold'])
    # the hold is an unplug and replug, and the Mac comes back as the source
    self.plug('none')
    self.run_for(1)
    self.plug('sink')
    self.run_for(60)
    self.assertEqual(self.commands(), ['off', 'hold'], "held for as long as the host stays plugged in")

  def test_an_iphone_or_a_jetson_that_came_up_as_the_device_is_made_the_host(self):
    for ids in (IPHONE, JETSON_GADGET):
      with self.subTest(ids=ids):
        self.setUp()
        self.plug('source')
        self.enumerate('2-1', ids)
        self.run_for(SWAP + 1)
        self.assertEqual(self.commands(), ['off', 'hold'])

  def test_unplugging_the_host_lets_the_port_go(self):
    self.plug('source')
    self.run_for(SWAP + 1)
    self.plug('sink')
    self.run_for(10)
    self.plug('none')
    self.run_for(RELEASE - 0.5)
    self.assertEqual(self.commands(), ['off', 'hold'], "a replug is not an unplug")
    self.run_for(1)
    self.assertEqual(self.commands(), ['off', 'hold', 'off'])
    self.assertFalse(self.port.held)

  def test_the_next_mac_gets_the_same_treatment(self):
    for _ in range(2):
      self.plug('source')
      self.run_for(SWAP + 1)
      self.plug('sink')
      self.run_for(5)
      self.plug('none')
      self.run_for(RELEASE + 1)
    self.assertEqual(self.commands(), ['off', 'hold', 'off', 'hold', 'off'])


class TestAccessories(PortTest):
  """A chestnut is never taken for a host; anything else that cannot host is
  let go without being cycled."""

  def test_a_chestnut_is_left_alone_and_looked_for_once(self):
    for ids in (CHESTNUT, CHESTNUT_ROM):
      with self.subTest(ids=ids):
        self.setUp()
        self.plug('source')
        self.enumerate('2-1', ids)
        with mock.patch.object(usbport, 'chestnut_attached', wraps=usbport.chestnut_attached) as scan:
          self.run_for(60)
        self.assertEqual(self.commands(), ['off'])
        self.assertEqual(scan.call_count, 1, "one sysfs read a cycle, not a directory walk")

  def test_a_sink_that_cannot_host_is_not_cycled(self):
    self.plug('source')
    self.run_for(SWAP + 1)
    self.plug('none')   # held at sink, a sink-only accessory has nothing to attach to
    self.run_for(RELEASE + 0.5)
    self.assertEqual(self.commands(), ['off', 'hold', 'off'])
    # dual role again, and it is back after a toggle; a poll can land in the gap
    self.run_for(0.5)
    self.plug('source')
    self.run_for(60)
    self.assertEqual(self.commands(), ['off', 'hold', 'off'])

  def test_a_new_plug_is_judged_afresh(self):
    self.plug('source')
    self.enumerate('2-1', CHESTNUT)
    self.run_for(SWAP + 1)
    self.plug('none')
    (self.devices / '2-1' / 'idVendor').unlink()
    self.run_for(usbport.UNPLUGGED + 0.5)
    self.plug('source')
    self.run_for(SWAP + 1)
    self.assertEqual(self.commands(), ['off', 'hold'])

  def test_a_hold_that_failed_is_not_retried_on_the_same_plug(self):
    self.script.side_effect = lambda command: command != 'hold'
    self.plug('source')
    self.run_for(60)
    self.assertEqual(self.commands(), ['off', 'hold'])
    self.assertFalse(self.port.held)

  def test_the_chestnut_ids_are_the_hardware_modules(self):
    from openpilot.common.hardware.usb import CHESTNUT_ROM_USB_IDS, CHESTNUT_USB_IDS
    self.assertEqual(usbport.CHESTNUT_IDS, frozenset(CHESTNUT_USB_IDS + CHESTNUT_ROM_USB_IDS))


class TestTheLink(PortTest):
  def test_ethernet_leaves_the_port_as_it_boots(self):
    self.plug('source')   # the adapter
    self.run_for(60, usb=False)
    self.assertEqual(self.commands(), [])

  def test_moving_to_ethernet_undoes_a_hold_once(self):
    self.plug('source')
    self.run_for(SWAP + 1)
    self.run_for(5, usb=False)
    self.assertEqual(self.commands(), ['off', 'hold', 'off'])
    self.assertFalse(self.port.held)

  def test_stopping_outside_a_hold_runs_nothing(self):
    self.plug('sink')
    self.run_for(5)
    self.port.off()
    self.assertEqual(self.commands(), ['off'])

  def test_without_the_lever_the_port_is_not_watched(self):
    self.script.return_value = False
    self.plug('source')
    with mock.patch.object(usbport, 'power_role') as role:
      self.run_for(60)
    self.assertEqual(self.commands(), ['off'])
    role.assert_not_called()

  def test_no_policy_engine_is_no_role(self):
    self.role.unlink()
    self.run_for(60)
    self.assertEqual(self.commands(), ['off'])


class TestTheScript(unittest.TestCase):
  def run_with(self, returncode: int, agnos: bool = True):
    result = subprocess.CompletedProcess([], returncode, stderr='')
    with mock.patch.object(gadget, 'AGNOS', agnos), \
         mock.patch.object(usbport.subprocess, 'run', return_value=result) as run:
      return usbport.run_script('hold'), run

  def test_it_runs_as_root(self):
    ok, run = self.run_with(0)
    self.assertTrue(ok)
    self.assertEqual(run.call_args.args[0], ['sudo', '-n', 'bash', str(usbport.SCRIPT), 'hold'])

  def test_off_agnos_it_does_nothing(self):
    ok, run = self.run_with(0, agnos=False)
    self.assertFalse(ok)
    run.assert_not_called()

  def test_a_device_without_the_lever_is_not_an_error(self):
    with mock.patch.object(gadget, 'log') as log:
      ok, _ = self.run_with(usbport.UNSUPPORTED)
    self.assertFalse(ok)
    log.warning.assert_called_once()
    log.error.assert_not_called()

  def test_the_script_is_shipped_beside_the_module(self):
    self.assertTrue(usbport.SCRIPT.is_file())


if __name__ == '__main__':
  unittest.main()
