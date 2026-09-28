#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The gadget owner, as manager runs it: jetlink.comma.owner, told which
provisioning worker to start.

The owner lives in jetlink with the rest of the comma's device layer and knows
no openpilot module, so this names the worker, a provisioning run
(provision.py), and where it runs.
It stays resident for the whole drive at about 10 MB, so nothing here may
import swaglog, Params, numpy, capnp or zmq; tests/test_comma_layer.py holds
the line.
"""
from __future__ import annotations

import sys

from jetlink.comma import owner

from openpilot.common.basedir import BASEDIR

WORKER = 'openpilot.sunnypilot.accelerators.jetlink.provision'


def main() -> None:
  owner.main([sys.executable, '-m', WORKER], cwd=BASEDIR, env={'PYTHONPATH': BASEDIR})


if __name__ == "__main__":
  main()
