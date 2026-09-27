"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The comma's USB-C port, kept the device end of a USB link.

The port is dual role. It hosts a chestnut or a USB ethernet adapter, and it is
the device for a jetlink host. An A-to-C cable settles which by construction:
the A end only pulls CC up, so the comma can only be the sink and the device.
A C-to-C cable to a Mac does not. Both ends are dual role, and the comma can
come out the source and the host, facing a Mac that cannot be a USB device.
Carrot's Jetson on C-to-C showed the same thing from the other side: the comma
read "Powered cable w/ sink" and enumerated the Jetson's own gadget.

So when the comma finds itself powering the far end and nothing has enumerated
on its host port, the far end is a host that lost the toss, and the port is
held at sink until that cable comes out. A chestnut or an adapter enumerates
within the wait and is left alone, and a USB-A host never gets here: the comma
is already the sink there. Holding for the whole session would be simpler and
would hide a chestnut plugged in while the link is on, since chestnut_present()
needs the comma to host it before jetlink stands aside.

The lever is the charger's DISABLE_POWER_ROLE_SWITCH voter, forced from
debugfs. It is the one that holds. The charger puts the port back to dual role
on every unplug and refuses a role written through the power supply once
nothing is attached, so the policy engine's rev3_sink_only and dual_role/mode
last one plug at most; a forced voter gates all of those writes.

USB PD is off while the link is on USB. An A-to-C cable never has any, and as
a PD sink the comma would ask a Mac for up to 3 A. Apple hosts have also
dropped PD sinks that do not answer their revision 3 messages
(raspberrypi/linux#6569). The policy engine reads this only when the comma
starts up as a sink, so a chestnut the comma powers is untouched.

None of it survives a reboot, and all of it is undone when the link is turned
off or moved to ethernet. Standard library only, like the rest of the owner.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from openpilot.sunnypilot.accelerators.jetlink import gadget

SCRIPT = Path(__file__).with_name('usb_port.sh')
POWER_ROLE = Path('/sys/class/usbpd/usbpd0/current_pr')
# USB_DEVICES_PATH and PRIMARY_USB_CONTROLLER in common/hardware/usb.py, which
# the owner cannot import: the hardware package brings cereal and capnp with it.
# a800000 is host-only and carries the modem
USB_DEVICES = Path('/sys/bus/usb/devices')
PORT_CONTROLLER = 'a600000.ssusb'
# how long the comma powers the far end with nothing enumerated before it takes
# the far end for a host. An adapter or a chestnut enumerates well inside this
SWAP_AFTER = 3.0
# how long the port reads empty before a hold is let go. The hold itself is an
# unplug and replug, so this has to outlast the replug
RELEASE_AFTER = 5.0
# how long the port reads empty before a plug counts as gone. A device let go
# by a hold reattaches in a DRP toggle and a CC debounce, well under this
UNPLUGGED = 1.5
SCRIPT_TIMEOUT = 5.0
UNSUPPORTED = 3


def power_role() -> str | None:
  """'source', 'sink' or 'none' from the policy engine, or None without one."""
  try:
    return POWER_ROLE.read_text().strip() or None
  except OSError:
    return None


def hosting_a_device() -> bool:
  """Has anything enumerated on the port while the comma was its host?

  Devices only: a root hub is named usbN and an interface has a colon. Not
  usb.py's idVendor test, which counts the root hubs host mode brings up.
  """
  try:
    names = os.listdir(USB_DEVICES)
  except OSError:
    return False
  for name in names:
    if name.startswith('usb') or ':' in name:
      continue
    if f'/{PORT_CONTROLLER}/' in os.path.realpath(USB_DEVICES / name):
      return True
  return False


def run_script(command: str) -> bool:
  if not gadget.AGNOS:
    return False
  try:
    result = subprocess.run(['sudo', '-n', 'bash', str(SCRIPT), command], stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, timeout=SCRIPT_TIMEOUT, text=True)
  except Exception:
    gadget.log.exception(f"jetlink: could not run the USB port script ({command})")
    return False
  if result.returncode == UNSUPPORTED:
    gadget.log.warning("jetlink: this device has no USB-C role lever; the port stays as it is")
  elif result.returncode != 0:
    gadget.log.error(f"jetlink: USB port script ({command}) failed: {result.stderr.strip()}")
  return result.returncode == 0


class Port:
  """Called once a cycle by the owner. A cycle reads one sysfs file; sudo only
  runs on a change."""

  def __init__(self):
    self._reset()

  def _reset(self) -> None:
    self.usb = False        # the link is on and over USB
    self.able = False       # the lever is there, so the role is worth watching
    self.held = False       # the voter is forced to sink
    self.took = False       # a host has attached since the hold began
    self.settled = False    # this plug has been judged; leave it until it comes out
    self.role: str | None = None
    self.role_since = 0.0

  def update(self, usb: bool, now: float | None = None) -> None:
    if not usb:
      return self.off()
    now = time.monotonic() if now is None else now
    if not self.usb:
      self.usb = True
      self.able = run_script('link')
    if not self.able:
      return
    role = power_role()
    if role != self.role:
      self.role, self.role_since = role, now
    lasted = now - self.role_since
    if self.held:
      if role == 'sink':
        self.took = True
      elif role != 'source' and lasted >= RELEASE_AFTER:
        self._release(now)
    elif role == 'source':
      if not self.settled and lasted >= SWAP_AFTER:
        if hosting_a_device():
          self.settled = True   # something the comma is meant to host
        else:
          self._hold(lasted)
    elif role == 'sink' or lasted >= UNPLUGGED:
      self.settled = False    # a host, or the plug is gone

  def _hold(self, lasted: float) -> None:
    gadget.log.warning(f"jetlink: nothing enumerated in {lasted:.0f} s on the USB-C port the comma powers; holding it as a device")
    self.took = False
    self.held = run_script('hold')
    # a hold that did not happen leaves the comma powering the same plug; judge
    # it once rather than run sudo every cycle
    self.settled = not self.held

  def _release(self, now: float) -> None:
    if self.took:
      gadget.log.warning("jetlink: the USB-C port is empty; back to dual role")
    else:
      # a sink-only device the comma did not see enumerate: it comes back as a
      # sink on dual role, and holding again would only cycle it
      gadget.log.warning("jetlink: no host came back on the USB-C port; leaving it dual role until the next plug")
    self.held, self.settled = False, not self.took
    # the device reattaches in a moment; the empty time restarts from here
    self.role_since = now
    run_script('release')

  def off(self) -> None:
    """The port as AGNOS boots it. Once: nothing to do if the link never was on USB."""
    if not self.usb:
      return
    self._reset()
    run_script('off')
