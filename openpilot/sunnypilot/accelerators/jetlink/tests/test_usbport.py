"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The comma's USB-C port: when it is held at sink, when it is let go, and what it
leaves alone. A USB-A host and a chestnut must see no change at all; a C-to-C
host that lost the toss gets one hold, and only for as long as it is plugged in.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openpilot.sunnypilot.accelerators.jetlink import gadget, usbport

SWAP = usbport.SWAP_AFTER
RELEASE = usbport.RELEASE_AFTER


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

  def plug(self, role: str) -> None:
    self.role.write_text(role + '\n')

  def enumerate(self, name: str, controller: str) -> None:
    """A device the comma hosts, linked into the fake /sys/bus/usb/devices."""
    target = self.tmp / 'platform' / controller / 'xhci-hcd.0.auto' / 'usb2' / name
    target.mkdir(parents=True)
    os.symlink(target, self.devices / name)

  def run_for(self, seconds: float, usb: bool = True) -> None:
    end = self.now + seconds
    while self.now < end:
      self.port.update(usb, now=self.now)
      self.now += 0.5

  def commands(self) -> list[str]:
    return [c.args[0] for c in self.script.call_args_list]


class TestHosts(PortTest):
  def test_a_usb_a_host_changes_nothing_but_pd(self):
    self.plug('sink')
    self.run_for(60)
    self.assertEqual(self.commands(), ['link'], "the comma is already the device on an A-to-C cable")

  def test_a_mac_that_lost_the_toss_is_made_the_host(self):
    self.plug('source')
    self.run_for(SWAP - 0.5)
    self.assertEqual(self.commands(), ['link'], "a device gets its chance to enumerate first")
    self.run_for(1)
    self.assertEqual(self.commands(), ['link', 'hold'])
    # the hold is an unplug and replug, and the Mac comes back as the source
    self.plug('none')
    self.run_for(1)
    self.plug('sink')
    self.run_for(60)
    self.assertEqual(self.commands(), ['link', 'hold'], "held for as long as the host stays plugged in")

  def test_unplugging_the_host_lets_the_port_go(self):
    self.plug('source')
    self.run_for(SWAP + 1)
    self.plug('sink')
    self.run_for(10)
    self.plug('none')
    self.run_for(RELEASE - 0.5)
    self.assertEqual(self.commands(), ['link', 'hold'], "a replug is not an unplug")
    self.run_for(1)
    self.assertEqual(self.commands(), ['link', 'hold', 'release'])
    self.assertFalse(self.port.held)

  def test_the_next_mac_gets_the_same_treatment(self):
    for _ in range(2):
      self.plug('source')
      self.run_for(SWAP + 1)
      self.plug('sink')
      self.run_for(5)
      self.plug('none')
      self.run_for(RELEASE + 1)
    self.assertEqual(self.commands(), ['link', 'hold', 'release', 'hold', 'release'])


class TestDevices(PortTest):
  """What the port hosts on purpose is never taken for a host."""

  def test_a_chestnut_is_left_alone_and_looked_for_once(self):
    self.plug('source')
    self.enumerate('2-1', 'a600000.ssusb')
    with mock.patch.object(usbport, 'hosting_a_device', wraps=usbport.hosting_a_device) as scan:
      self.run_for(60)
    self.assertEqual(self.commands(), ['link'])
    self.assertEqual(scan.call_count, 1, "one sysfs read a cycle, not a directory walk")

  def test_a_hold_that_failed_is_not_retried_on_the_same_plug(self):
    self.script.side_effect = lambda command: command != 'hold'
    self.plug('source')
    self.run_for(60)
    self.assertEqual(self.commands(), ['link', 'hold'])
    self.assertFalse(self.port.held)

  def test_a_sink_that_never_enumerates_is_not_cycled(self):
    self.plug('source')
    self.run_for(SWAP + 1)
    self.plug('none')   # held at sink, a sink-only device has nothing to attach to
    self.run_for(RELEASE + 0.5)
    self.assertEqual(self.commands(), ['link', 'hold', 'release'])
    # dual role again, and it is back after a toggle; a poll can land in the gap
    self.run_for(0.5)
    self.plug('source')
    self.run_for(60)
    self.assertEqual(self.commands(), ['link', 'hold', 'release'])

  def test_a_new_plug_is_judged_afresh(self):
    self.plug('source')
    self.enumerate('2-1', 'a600000.ssusb')
    self.run_for(SWAP + 1)
    self.plug('none')
    os.unlink(self.devices / '2-1')
    self.run_for(usbport.UNPLUGGED + 0.5)
    self.plug('source')
    self.run_for(SWAP + 1)
    self.assertEqual(self.commands(), ['link', 'hold'])


class TestTheLink(PortTest):
  def test_ethernet_leaves_the_port_as_it_boots(self):
    self.plug('source')   # the adapter
    self.run_for(60, usb=False)
    self.assertEqual(self.commands(), [])

  def test_moving_to_ethernet_puts_it_all_back(self):
    self.plug('source')
    self.run_for(SWAP + 1)
    self.run_for(5, usb=False)
    self.assertEqual(self.commands(), ['link', 'hold', 'off'], "once, not every cycle")
    self.assertFalse(self.port.held)

  def test_without_the_lever_the_port_is_not_watched(self):
    self.script.side_effect = lambda command: command != 'link'
    self.plug('source')
    with mock.patch.object(usbport, 'power_role') as role:
      self.run_for(60)
    self.assertEqual(self.commands(), ['link'])
    role.assert_not_called()

  def test_no_policy_engine_is_no_role(self):
    self.role.unlink()
    self.run_for(60)
    self.assertEqual(self.commands(), ['link'])


class TestHostingADevice(PortTest):
  def test_only_the_ports_controller_counts(self):
    self.enumerate('1-1', 'a800000.ssusb')   # the modem
    self.assertFalse(usbport.hosting_a_device())
    self.enumerate('2-1', 'a600000.ssusb')
    self.assertTrue(usbport.hosting_a_device())

  def test_root_hubs_and_interfaces_are_not_devices(self):
    self.enumerate('usb2', 'a600000.ssusb')
    self.enumerate('2-0:1.0', 'a600000.ssusb')
    self.assertFalse(usbport.hosting_a_device())


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
