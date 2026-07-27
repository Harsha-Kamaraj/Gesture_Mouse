# Architecture

Deeper detail than the README, aimed at someone about to modify the code.

---

## 1. Module dependency graph

Arrows point from dependant to dependency. There are no cycles; every module can
be imported in isolation.

```
                              app.py
        ┌────────────┬───────────┼───────────┬─────────────┐
        ▼            ▼           ▼           ▼             ▼
      ui.py     gesture_    cursor_      actions.py    calibration.py
        │       engine.py  controller.py     │         gesture_recorder.py
        ▼            │           │           ▼
   dashboard.py      │           │      platform_bridge.py
   overlay.py        ▼           │      screen_capture.py
   notifications.py  dynamic_    │      sounds.py
   themes.py         gestures.py │      security.py
        │            │           │           │
        └────────────┴───────────┴───────────┘
                          ▼
              config.py · utils.py · logger.py
                          ▼
                      compat.py
```

`compat.py` sits at the bottom because its shims must execute before MediaPipe,
matplotlib or CustomTkinter are imported anywhere. `config.py` imports it, and
everything imports `config`, which makes the ordering automatic rather than a
rule contributors have to remember.

---

## 2. The domain/effects boundary

This is the single most important structural decision in the project.

**Domain modules are pure.** `gesture_engine.py`, `dynamic_gestures.py` and
`calibration.py` never perform I/O, never move the cursor, never touch the
filesystem, and never import a UI toolkit. They take data in and return data out.

```python
# gesture_engine.py — the entire public contract
output: EngineOutput = engine.update(hands, timestamp)
# EngineOutput carries: mode, pose, confidence, cursor_point,
#                       events[], control_value, trail[]
```

**Effects modules act.** `actions.py` receives `GestureEvent` objects and decides
what they mean; `cursor_controller.py` receives a normalized point and produces
pointer motion.

Three consequences follow directly:

1. **Recognition is testable without a machine to control.** All 34 unit tests
   run against pure functions.
2. **Rebinding is free.** A gesture's meaning lives in a dictionary
   (`DEFAULT_BINDINGS`), not in the classifier.
3. **Plugins are safe to add.** A plugin registers an action; it cannot
   accidentally alter recognition behaviour.

When adding a feature, ask which side it belongs on. "Detect a new pose" is
domain. "Open an application" is effects. If something seems to need both, it is
usually two features.

---

## 3. Threading in detail

### Three threads, one shared object

```python
# app.py
self.state = SharedState()          # plain dataclass
self._state_lock = threading.Lock()
```

Everything crossing a thread boundary goes through `SharedState` under
`_state_lock`. Nothing else is shared. The UI never holds a reference to the
engine, the detector or the cursor controller.

### Why latest-frame-wins matters

`cv2.VideoCapture` maintains an internal buffer. Calling `read()` from a loop
that also does inference means:

```
camera produces at 30 fps  ──►  buffer  ──►  consumer drains at 20 fps
```

The buffer grows. After ten seconds you are processing images from three seconds
ago and the cursor lags visibly behind the hand. Setting `CAP_PROP_BUFFERSIZE=1`
helps but is not honoured by every backend.

`CameraStream` instead runs `read()` on its own thread and keeps exactly one
frame:

```python
with self._new_frame:
    self._frame = frame          # overwrite, never append
    self._frame_index += 1
    self._new_frame.notify_all()
```

A slow consumer now drops frames instead of accumulating latency. Dropped frames
are counted in `PerformanceMonitor.dropped_frames` and surfaced on the dashboard.

### Why actions run on the processing thread

Launching an application or encoding a PNG takes tens to hundreds of
milliseconds. Two options:

- **On the UI thread** — the interface freezes visibly.
- **On the processing thread** — the next gesture is delayed instead.

The second is correct: gestures are already rate-limited by cooldowns, so a
delayed gesture is invisible, whereas a frozen UI is not. Actions that are
genuinely long-running (screen recording) spawn their own thread.

### Tk marshalling

Tk is not thread-safe. Notifications originate on the processing thread, so
`ToastManager.notify` never touches a widget:

```python
self.root.after(0, lambda: self._show(toast))
```

---

## 4. Recognition internals

### 4.1 Feature extraction

`extract_features()` computes everything a gesture rule might need, once per
frame, so classification is pure comparison with no geometry in it.

| Feature | Meaning |
|---|---|
| `extensions` | Per-finger extension, `[0, 1]`, thumb→pinky |
| `pinch_distances` | Thumb tip to each fingertip, palm-normalised |
| `index_tip`, `palm_centre` | Candidate cursor sources |
| `orientation` | Hand roll in degrees |
| `palm_facing` | Front or back of hand, from an edge cross-product |
| `palm_size` | Scale unit and camera-distance proxy |
| `spread` | Finger splay, `[0, 1]` |

### 4.2 Finger extension

Non-thumb fingers combine two signals:

```python
angular = smoothstep(95°, 158°, pip_joint_angle)       # primary
reach   = smoothstep(0.98, 1.22, tip_reach/pip_reach)  # veto for folded fingers
score   = 0.72 * angular + 0.28 * reach
```

The angle alone fails for a finger that is *straight but folded at the knuckle* —
its PIP angle is large yet the tip sits no further from the wrist than the PIP
joint. The reach term catches exactly that case.

The thumb needs a different rule entirely. It has one fewer phalanx and moves by
abduction rather than flexion, so a PIP-style angle barely changes between tucked
and extended. Lateral spread from the index knuckle is the discriminating
measurement:

```python
spread_score = smoothstep(0.45, 0.82, dist(thumb_tip, index_mcp) / palm_size)
score = 0.8 * spread_score + 0.2 * straightness
```

Measured separation on the synthetic model: **0.14 tucked vs 0.87 extended.**

### 4.3 Pose matching

Poses are declarative data:

```python
PoseDefinition(Pose.PEACE, (0, 1, 1, 0, 0), priority=1.05)
#                          thumb ─┘ │ │ │ └─ pinky      1=extended 0=curled None=any
```

Scoring deliberately punishes the worst finger:

```python
score = 0.55 * mean(finger_scores) + 0.45 * min(finger_scores)
```

A plain mean lets one clearly wrong finger hide behind four correct ones —
`FOUR_FINGERS` would score well on an open palm. Weighting the minimum makes the
classifier decisive.

> **A subtlety worth knowing.** `POINT` cannot use a wildcard thumb. With
> `(None, 1, 0, 0, 0)` it also matches the L-shape perfectly and swallows it,
> because the two poses differ by nothing else. It uses `(0, 1, 0, 0, 0)`.

### 4.4 Temporal stabilisation

MediaPipe occasionally emits one bad frame during fast motion. Without
stabilisation that frame becomes a click in the user's document.

```python
stable = stabilizer.update(pose)   # needs N consecutive agreeing frames
```

Release hysteresis is asymmetric: a confirmed pose survives a couple of
contradicting frames before dropping, so a momentary tracking glitch does not
kick you out of scroll mode mid-scroll.

### 4.5 Event gating

Every event passes four gates, in this order:

```
enabled? → confidence ≥ threshold → per-gesture cooldown → global cooldown
```

The global cooldown is checked **last** on purpose: a gesture rejected for low
confidence must not consume the shared budget, or one flickering gesture would
suppress every other gesture.

`RELEASE_GESTURES` bypasses gating entirely. See the README's Safety section.

---

## 5. Cursor signal processing

### One Euro filter

A first-order low-pass filter with a *speed-dependent* cutoff:

```
τ  = 1 / (2π · f_c)
α  = 1 / (1 + τ/Te)
f_c = f_cmin + β · |ẋ̂|          ← the whole idea
```

At rest `|ẋ̂| ≈ 0`, so the cutoff collapses to `f_cmin` and filtering is
aggressive — sensor jitter disappears and the cursor holds still on a button.
During fast movement the cutoff rises and the filter gets out of the way.

The UI's single **Smoothing** slider maps onto `f_cmin`:

```python
min_cutoff = max(0.05, one_euro_min_cutoff * (1 - smoothing) * 2.0)
```

### Velocity prediction

A two-frame finite difference is far too noisy to extrapolate from, so
`VelocityEstimator` least-squares fits position against time over a 5-sample
window and predicts:

```python
predicted = filtered + velocity * prediction_time    # default 35 ms
```

### Precision mode

Precision mode does **not** remap the active region — that would teleport the
cursor when engaged. It scales movement relative to the current position:

```python
target = previous + (target - previous) * gain      # gain ≈ 0.28
```

---

## 6. Configuration and profiles

`AppConfig` is a nested dataclass tree serialised to JSON. Hydration tolerates
both missing and unknown keys:

```python
hints = get_type_hints(cls)          # annotations are strings under
                                     # `from __future__ import annotations`
for f in fields(cls):
    if f.name not in data: continue  # missing  → keep default
    ...                              # unknown  → ignored
```

That is what makes an old profile forward-compatible with a new release, and a
new profile safe to load on an old one.

Profile switching broadcasts to subscribers so the detector, cursor controller,
overlay and UI all re-read settings without restarting:

```python
self.profiles.subscribe(self._on_config_changed)
```

---

## 7. Graceful degradation

No optional dependency is allowed to prevent startup. The pattern is uniform:

```python
class VolumeBackend(ABC):
    available: bool = False

def _build_volume_backend() -> VolumeBackend:
    for factory in candidates:
        try:
            return factory()
        except Exception:
            continue
    return _NullVolume()          # available == False
```

Callers check `.available` and the UI surfaces the result in
**Settings → System Capabilities**, so a user can see *why* brightness gestures
do nothing rather than assuming the app is broken.

The same pattern appears for MediaPipe backends (Solutions → Tasks), face
detection (BlazeFace → Haar), mouse control (pynput → pyautogui) and sound
playback (winsound / afplay / paplay / none).

---

## 8. Adding things

| Task | Where | Notes |
|---|---|---|
| New static pose | `POSE_LIBRARY` in `gesture_engine.py` | Data only |
| New air-drawn shape | `builtin_templates()` in `dynamic_gestures.py` | One exemplar suffices |
| New action | `_builtin_actions()` in `actions.py`, or a plugin | Keep the id stable |
| New setting | The relevant dataclass in `config.py` | Add a control in `_build_settings` |
| New theme | `builtin_themes()` or `assets/themes/*.json` | Tokens only |
| New view | `_build_<name>` in `ui.py` + `NAV_ITEMS` | Built lazily |
| New OS capability | `platform_bridge.py` | ABC + per-platform impls + null stub |

---

## 9. Performance notes

Measured on an Apple M4, 1280×720 capture:

| Stage | Cost |
|---|---|
| Overlay render (full HUD) | ~2.7 ms |
| MediaPipe inference (complexity 1, scale 0.6) | ~8–14 ms |
| Gesture recognition | < 0.5 ms |
| Cursor pipeline | < 0.1 ms |

Levers when frames are tight, in order of impact:

1. `camera.inference_scale` — 0.6 → 0.5 is roughly a 30 % inference saving
2. `detection.model_complexity` — 1 → 0 roughly halves inference
3. `detection.max_num_hands` — 2 → 1
4. `ui.show_landmarks` / `show_skeleton` — minor, but free

Watch **p95/p99** rather than mean frame time on the Performance view. A 60 FPS
average with a 90 ms p99 feels broken to the user, and only the tail shows it.
