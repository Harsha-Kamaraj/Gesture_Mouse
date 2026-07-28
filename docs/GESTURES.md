# Gesture reference

Complete listing of every recognised gesture, its default binding and how to
tune it. Finger notation reads **thumb → pinky**, where `1` = extended,
`0` = curled and `–` = ignored.

---

## Static poses

The twelve shapes the classifier recognises. `POSE_LIBRARY` in
`gesture_engine.py` is the source of truth.

| Pose | Pattern | Meaning |
|---|---|---|
| Open Palm | `1 1 1 1 1` | Neutral / sleep / wake |
| Fist | `0 0 0 0 0` | *Unbound by default* |
| Point | `0 1 0 0 0` | Move the cursor · draw shapes |
| Peace | `0 1 1 0 0` | Enter **Scroll** mode |
| Three Fingers | `0 1 1 1 0` | Enter **Brightness** mode |
| Four Fingers | `0 1 1 1 1` | Toggle presentation mode |
| Thumb Up | `1 0 0 0 0` | Context dependent |
| Rock | `0 1 0 0 1` | Enter **Volume** mode |
| Call Me | `1 0 0 0 1` | Screen recording |
| L Shape | `1 1 0 0 0` | **Precision** mode while held |
| Pinky | `0 0 0 0 1` | Toggle whiteboard |
| OK Sign | `– – 1 1 1` + pinch | Screenshot |

> **Why Point requires a curled thumb.** With the thumb wildcarded, `Point` also
> matches `L Shape` perfectly — the two differ by nothing else — and would
> swallow it entirely. This is a genuine constraint of pattern-based
> classification, not an arbitrary choice.

---

## Pinches

Pinch detection is independent of pose classification, and pinch events carry
their own confidence derived from how firmly closed they are. This matters:
a pinching hand matches the L-shape pattern, which the pose library penalises
via `requires_no_pinch`, so pose confidence *collapses* exactly when a pinch
fires.

| Contact | Gesture id | Default action |
|---|---|---|
| Thumb + index, brief | `pinch_tap` | Left click |
| Thumb + index, twice quickly | `pinch_double` | Double click |
| Thumb + index, held ≥ 0.45 s | `drag_start` → `drag_end` | Drag and drop |
| Thumb + middle | `pinch_middle` | Right click |
| Thumb + ring | `pinch_ring` | Middle click |

### The click state machine

```
            distance ≤ pinch_threshold
   OPEN ───────────────────────────────► CLOSED
     ▲                                     │
     │  distance ≥ release_threshold       │  held ≥ drag_hold_time
     │                                     ▼
     └──────────────── DRAGGING ◄──────────┘
              (release → drag_end)
```

Two thresholds, not one. A single threshold makes the pinch chatter open/closed
while held at the boundary, producing a burst of phantom clicks. The release
threshold sits above the close threshold, giving Schmitt-trigger hysteresis.

---

## Modes

A held pose switches mode; the mode reinterprets continuous hand motion. This is
what keeps the gesture vocabulary small — without it, every continuous control
would need to steal a distinct hand shape, and distinguishable shapes run out
fast.

| Mode | Entered by | Continuous input | Exits when |
|---|---|---|---|
| **Navigate** | default | Index tip → cursor | — |
| **Scroll** | ✌️ Peace | Vertical travel → scroll ticks | Pose changes |
| **Volume** | 🤘 Rock | Thumb↔index distance → level | Pose changes |
| **Brightness** | Three fingers | Thumb↔index distance → level | Pose changes |
| **Drag** | Held pinch | Palm centre → cursor | Pinch released |
| **Zoom** | Both hands pinching | Hand separation → zoom | Either releases |
| **Sleeping** | Open palm held | *(nothing)* | Fresh open-palm hold |

During a drag the cursor source switches from the index tip to the **palm
centre**. The fingers are curled while dragging, which degrades fingertip
estimates badly; the palm centre stays steady.

**Waking requires the pose to change first.** Otherwise the same continuous hold
that put the engine to sleep would wake it one hold-duration later.

---

## Holds

Poses held for ~1.2 s (**Settings → Gestures → Mode Hold**). Each fires once
per hold, not repeatedly.

| Pose | Gesture id | Default action |
|---|---|---|
| Open palm | `open_palm_hold` | Sleep / wake tracking |
| OK sign | `ok_sign` | Screenshot |
| Call me 🤙 | `call_hold` | Start / stop screen recording |
| Pinky | `pinky_hold` | Toggle whiteboard |
| Four fingers | `four_hold` | Toggle presentation mode |
| Fist | `fist_hold` | *Unbound by default* |

---

> **Why the fist is unbound.** A relaxed or transitioning hand reads as a fist
> more readily than any other pose, and the obvious binding for it —
> `hold_click` — *latches* the left button, so one accidental fire leaves the
> button held until something toggles it back. Pinch-and-hold already covers
> press-and-hold and is self-limiting, since releasing the pinch releases the
> button. Bind `hold_click` to the fist in **Gestures** if you want it.

## Swipes

Detected while pointing, using net displacement and path straightness rather
than the `$1` recogniser.

| Direction | Gesture id | Default action |
|---|---|---|
| ← | `swipe_left` | Browser back |
| → | `swipe_right` | Browser forward |
| ↑ | `swipe_up` | Next slide |
| ↓ | `swipe_down` | Previous slide |

Thresholds: ≥ 0.22 net displacement, ≥ 0.45 units/s, and displacement/path-length
≥ 0.82 (a circle travels far further than it displaces, so it cannot qualify).

> Swipes are deliberately **not** `$1` templates. `$1` normalises away rotation,
> so a left swipe and a right swipe reduce to the same template — it structurally
> cannot distinguish them.

---

## Air-drawn shapes

Traced with the index finger. Recognition runs once the stroke settles, so a
half-drawn circle cannot fire early as a different shape.

| Shape | Gesture id | Default action |
|---|---|---|
| ○ Circle (either direction) | `circle`, `circle_cw` | Open browser |
| △ Triangle | `triangle` | Open VS Code |
| □ Square | `square` | Open terminal |
| ∿ Wave | `wave` | Toggle mute |
| Z | `z` | Open Spotify |
| S | `s` | Play / pause media |
| ✓ Check | `v_check` | Lock screen |
| ^ Caret | `caret` | App launcher |

Requirements: ≥ 12 samples, ≥ 0.65 normalised path length, and a match score
≥ `dynamic_min_score` (default 0.80).

### Recording your own

**Gestures → name → Record Gesture**, then draw the shape three times with your
index finger.

Two validations run before it is saved:

1. **Self-consistency** — the three takes are compared pairwise; below 72 %
   agreement the gesture is rejected. If your own attempts do not resemble each
   other, it will never work reliably in use, and hearing that at record time
   beats discovering it through misfires later.
2. **Collision** — the new shape is scored against every existing template.
   Above 93 % similarity it is rejected, because two near-identical templates
   make *both* unreliable.

Custom gestures are stored in `data/custom_gestures.json` and can be exported and
shared.

---

## Tuning

All under **Settings**, all per-profile.

| Symptom | Setting | Direction |
|---|---|---|
| Accidental clicks | `min_confidence` | ↑ |
| " | `stability_frames` | ↑ |
| " | `pinch_threshold` | ↓ |
| Clicks not registering | `pinch_threshold` | ↑ |
| " | `min_confidence` | ↓ |
| Drag triggers when clicking | `drag_hold_time` | ↑ |
| Cursor jitters | `smoothing`, `dead_zone` | ↑ |
| Cursor lags | `smoothing` ↓, `prediction_time` | ↑ |
| Too much arm movement | `active_region_margin` ↑ or `sensitivity` | ↑ |
| Gestures fire repeatedly | `global_cooldown` | ↑ |
| Modes toggle on their own | `hold_duration` (Mode Hold) | ↑ |
| Shapes not recognised | `dynamic_min_score` | ↓ |

**Run the calibration wizard first.** It measures your hand, pinch range, reach
and tremor, and derives most of the above automatically. Manual tuning is for
refinement afterwards.

---

## Profiles

| Profile | Tuned for |
|---|---|
| **Default** | Balanced everyday use |
| **Gaming** | Low latency — minimal smoothing, high gain, short cooldowns, model complexity 0 |
| **Office** | Precision — heavy smoothing, conservative thresholds |
| **Presentation** | Air slide control, longer cooldowns |
| **Accessibility** | Amplified cursor, forgiving thresholds, large high-contrast UI, voice feedback |

---

## Safety

| Shortcut | Action |
|---|---|
| `Ctrl+Alt+Q` | Emergency stop — releases all input, disables the cursor |
| `Ctrl+Alt+Space` | Pause / resume tracking |
| 🖐️ held 1 s | Sleep tracking (gesture equivalent) |

Automatic protections:

- Losing hand tracking mid-drag releases the mouse button
- Shutdown releases all three buttons unconditionally
- Optional presence detection pauses tracking when you leave the camera
- Releases (`drag_end`) bypass confidence and cooldown gates entirely — you must
  be confident to *engage* a control, never to *release* one
