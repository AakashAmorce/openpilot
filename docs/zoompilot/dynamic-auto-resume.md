# Dynamic Auto Resume

Code: `opendbc/car/mazda/dynamic_auto_resume.py` (the state machine),
`opendbc/car/mazda/carcontroller.py` (`update_dynamic_auto_resume`), `opendbc/car/mazda/carstate.py`
(`distance_setting`). Tests: `car/mazda/tests/test_mazda_dynamic_auto_resume.py`,
`safety/tests/test_mazda.py::test_distance_taps_ride_the_engaged_gate`. Setting: Cruise panel,
"Dynamic Auto Resume" (`MazdaDynamicAutoResume`), Mazda with stock MRCC only.

Status: prototype, not yet validated on a car.

## The problem

Stock MRCC with Stop & Go resumes on its own only if the lead leaves within 3 s of the stop. After that
it holds (HOLD) until RES or the gas, and zoompilot's auto-resume presses RES once the plan stops
asking to stay stopped. Even then, MRCC "does not start moving until the distance between your vehicle
and the vehicle ahead lengthens to the specified distance or farther" (CX-5 owner's manual, Stop Hold
Control). At a long following setting that is a large gap, so the car sits while traffic moves.

## What it does

Through a HOLD it shortens the MRCC following setting to 1 bar, so the pull-away starts sooner, and
puts the driver's own setting back once the gap has opened to what that setting would hold at the
current speed. It only ever uses settings the driver could pick on the wheel; stock MRCC keeps the gas,
the brakes and AEB.

States: `idle` -> `shortening` (held at least 3.5 s, past MRCC's own 3 s resume window) -> `short`
(held, or crawling) -> `restoring` -> `idle`. `restore_pending` covers cruise dropping while short.

- Closed loop: one distance tap, then wait up to 0.6 s for `CRZ_CTRL.DISTANCE_SETTING` to move before
  the next. Taps share CRZ_BTNS pacing with ICBM and the TJA cleanup (`last_button_frame`, at least
  0.2 s apart, the body ECU's registration floor from `icbm.md`). ICBM waits while a sequence runs.
- Never delays a resume: if openpilot starts pressing RES or the car moves mid-sequence, the
  pull-away goes first with whatever setting was reached.
- Restore point: the lead gap reaches the driver's setting's gap at the current speed (time gaps from
  the manual: 50/40/30/25 m at 80 km/h, plus `STOP_GAP` = 4 m), checked after 3 s of moving so
  stop-and-go traffic does not cycle the setting at every start. Backstops: 20 mph, 15 s, or 1 s
  without a lead.
- The driver wins: a physical distance press during an episode ends it and keeps the driver's choice.
  One episode per stop.
- A deaf ECU: if three taps in a row go unconfirmed at a stop with none ever confirmed, it stops arming
  for the rest of the drive. A restore that keeps missing retries every 2 s, five attempts, then logs
  `restore_failed`.

`DISTANCE_SETTING` raw: 1 is 4 bars (longest), 4 is 1 bar (shortest); `DISTANCE_LESS` raises it (see
the cluster comment in `update_longitudinal`).

## Safety

No safety-mode change. The taps are ordinary CRZ_BTNS presses, which the panda already allows only
while controls are allowed (MRCC engaged) and refuses otherwise, like RES. The flag is never set under
openpilot longitudinal. Consequence: after a brake press cancels cruise while short, the restore waits
for the next engagement (`restore_pending`); a narrow "longer is always allowed" exception in the safety
mode would close that gap and is left for later.

## Shadow mode (log only)

`MazdaDynamicAutoResumeShadow` (developer, no UI) runs the same logic against a virtual setting and logs
every decision without sending a frame. It wins over the toggle. Over SSH:

```
echo -n 1 > /data/params/d/MazdaDynamicAutoResumeShadow   # takes effect at the next car start
```

Events are `carlog` INFO lines with `"event": "mazda_dar"` (transitions and taps), forwarded to cloudlog.

## Validation plan

1. By hand, no code: in a HOLD, press distance-shorter to 1 bar. Does MRCC accept it without dropping
   the hold, and does the pull-away start sooner than at the usual setting?
2. Shadow drives: the arm, restore and give-up points look right against the drive.
3. Live, toggle on: time from lead moving to ego moving and the gap at start, against a baseline at
   the usual setting; zero faults, every episode restored.

Open questions: whether MRCC takes distance presses in HOLD at all; whether the resume gap follows the
setting; the real standstill gap behind `STOP_GAP`.

## Known limitations

- A restart (or ignition off) while short loses the driver's setting, and Mazda keeps the short one
  across ignition cycles. The driver sees 1 bar on the dash and can set it back.
- Taps are assumed to register like ICBM's SET taps (one frame, counter offset +1); unconfirmed taps
  are retried, never assumed.
