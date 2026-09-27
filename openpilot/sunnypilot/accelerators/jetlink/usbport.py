"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The comma's USB-C port, kept the device end of a USB link.

The port is dual role. It hosts a chestnut, and it is the device for a jetlink
host. An A-to-C cable settles which by construction: the A end only pulls CC
up, so the comma can only be the sink and the device. A C-to-C cable does not.
Both ends are dual role, and the comma can come out the source and the host:
facing a Mac, nothing enumerates; facing an iPhone or a Jetson's own USB-C
port, the far end enumerates as a device. Carrot's Jetson on C-to-C read
"Powered cable w/ sink" and enumerated as 0955:7020.

While the link runs over USB, a chestnut is the only thing the comma should
host on this port; ethernet takes the link off USB and this with it. So once
the comma has been the host for a few seconds with no chestnut on the port,
whatever is on the other end is a host that lost the toss, and the port is held
at sink until that cable comes out. Nothing happens anywhere else: the comma
as the device (every USB-A host, a C-to-C host that won), a power supply, a
chestnut, and a device without the lever all leave the port as AGNOS boots it.
Holding for the whole session would be simpler and would hide a chestnut
plugged in while the link is on, since chestnut_present() needs the comma to
host it before jetlink stands aside.

The lever is the charger's DISABLE_POWER_ROLE_SWITCH voter, forced from
debugfs. It is the one that holds. The charger puts the port back to dual role
on every unplug and refuses a role written through the power supply once
nothing is attached, so the policy engine's rev3_sink_only and dual_role/mode
last one plug at most; a forced voter gates all of those writes.

USB PD is off for the length of a hold, so the host that comes back gets what
an A-to-C cable gives it: no contract, and no request from the comma for 3 A.
Apple hosts have also dropped PD sinks that do not answer their revision 3
messages (raspberrypi/linux#6569). None of it survives a reboot.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from openpilot.sunnypilot.accelerators.jetlink import gadget

SCRIPT = Path(__file__).with_name('usb_port.sh')
POWER_ROLE = Path('/sys/class/usbpd/usbpd0/current_pr')
USB_DEVICES = Path('/sys/bus/usb/devices')
# CHESTNUT_USB_IDS and CHESTNUT_ROM_USB_IDS in common/hardware/usb.py, which the
# owner cannot import: the hardware package brings cereal and capnp with it.
# The ROM ones too, so a chestnut being flashed is never taken for a host
CHESTNUT_IDS = frozenset({(0xADD1, 0x0001), (0x3801, 0x0001), (0x174C, 0x2464), (0x174C, 0x2463)})
# how long the comma hosts the far end before judging it. A chestnut enumerates
# well inside this
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


def chestnut_attached() -> bool:
  """Is a chestnut enumerated, running or in its ROM? It can only be on this port."""
  try:
    names = os.listdir(USB_DEVICES)
  except OSError:
    return False
  for name in names:
    try:
      ids = tuple(int((USB_DEVICES / name / f).read_text(), 16) for f in ('idVendor', 'idProduct'))
    except (OSError, ValueError):
      continue   # an interface, or a device going away
    if ids in CHESTNUT_IDS:
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
      # the port as AGNOS boots it, whatever an owner killed mid-hold left
      # behind, and the answer to whether there is a lever at all
      self.usb = True
      self.able = run_script('off')
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
        if chestnut_attached():
          self.settled = True
        else:
          self._hold(lasted)
    elif role == 'sink' or lasted >= UNPLUGGED:
      self.settled = False    # a host, or the plug is gone

  def _hold(self, lasted: float) -> None:
    gadget.log.warning(f"jetlink: hosting something that is not a chestnut for {lasted:.0f} s on the USB-C port; holding it as a device")
    self.took = False
    self.held = run_script('hold')
    # a hold that did not happen leaves the comma hosting the same plug; judge
    # it once rather than run sudo every cycle
    self.settled = not self.held

  def _release(self, now: float) -> None:
    if self.took:
      gadget.log.warning("jetlink: the USB-C port is empty; back to dual role")
    else:
      # a sink-only accessory: it comes back as a sink on dual role, and
      # holding again would only cycle it
      gadget.log.warning("jetlink: no host came back on the USB-C port; leaving it dual role until the next plug")
    self.held, self.settled = False, not self.took
    # the accessory reattaches in a moment; the empty time restarts from here
    self.role_since = now
    run_script('off')

  def off(self) -> None:
    """Undo a hold. Outside one the port is already as AGNOS boots it."""
    held = self.held
    self._reset()
    if held:
      run_script('off')
