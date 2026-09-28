"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

modeld's per-frame status callback for the jetlink path.

The same hook as ChestnutState.send: modeld calls it once the frame is on the
accelerator. It publishes nothing, chestnutState is comma's board on the wire
and a Jetson's telemetry has no message yet, so it goes to swaglog at 1 Hz.
The hook has to exist because passing a callback is what makes the client ask
for telemetry, piggybacked on the previous response at no extra round trip.
It takes no PubMaster: accelerators.load() builds it before modeld has one.
"""
from __future__ import annotations

import time

from openpilot.common.swaglog import cloudlog

LOG_PERIOD = 1.0


class JetlinkStatus:
  def __init__(self, model):
    self.model = model
    self._last_logged = 0.0

  @property
  def client(self):
    # per send: a joining state has no client until the Jetson turns up
    return getattr(self.model, 'client', None)

  def send(self) -> None:
    client = self.client
    telemetry = client.last_state if client is not None else None
    if not telemetry:
      return
    now = time.monotonic()
    if now - self._last_logged < LOG_PERIOD:
      return
    self._last_logged = now
    cloudlog.event("jetlinkTelemetry", dead=bool(client.dead), **telemetry)
