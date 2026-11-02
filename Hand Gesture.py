"""
Hand Gesture to Text (webcam) using OpenCV + MediaPipe.

Install:
    pip install opencv-python mediapipe
    (use opencv-python-headless only if you do NOT need a window)

Run:
    python hand_gesture_to_text.py            # default camera 0
    python hand_gesture_to_text.py --camera 1

Keys:
    q = quit      c = clear displayed gesture
"""

import argparse
import logging
import math
import sys
import time
from collections import Counter, deque

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "Could not import OpenCV (cv2). Install it with: 'pip install opencv-python'."
    ) from e

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "Could not import mediapipe. Install it with: 'pip install mediapipe'."
    ) from e

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gesture")

# ---------- Landmark ids ----------
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

NON_THUMB = [(INDEX_PIP, INDEX_TIP), (MIDDLE_PIP, MIDDLE_TIP),
             (RING_PIP, RING_TIP), (PINKY_PIP, PINKY_TIP)]

# ---------- Gesture -> text mapping (edit labels here) ----------
LABELS = {
    "OPEN_PALM": "STOP",
    "FIST": "YES",
    "THUMB_UP": "GOOD",
    "V_SIGN": "PEACE",
    "INDEX": "POINT",
    "ROCK": "GO ON",
    "TWO_TOGETHER": "NO",
}

# ---------- Stability settings ----------
HISTORY_LEN = 10        # frames kept for voting
GESTURE_MIN_VOTES = 7   # votes needed to commit a real gesture
NO_HAND_MIN_VOTES = 4   # votes needed to commit "NO HAND" (clears quickly)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(pts):
    """
    Rotation-independent finger detection.
    pts: list of 21 (x, y) pixel tuples.
    Returns [thumb, index, middle, ring, pinky] booleans.
    A finger is 'up' when its tip is farther from the wrist than its PIP joint.
    """
    palm = dist(pts[WRIST], pts[MIDDLE_MCP]) or 1.0
    fingers = []

    # Thumb: tip must be away from the pinky base and away from the index base.
    thumb_open = (
        dist(pts[THUMB_TIP], pts[PINKY_MCP]) > dist(pts[THUMB_IP], pts[PINKY_MCP])
        and dist(pts[THUMB_TIP], pts[INDEX_MCP]) > 0.55 * palm
    )
    fingers.append(thumb_open)

    for pip_id, tip_id in NON_THUMB:
        up = dist(pts[tip_id], pts[WRIST]) > dist(pts[pip_id], pts[WRIST]) * 1.08
        fingers.append(up)

    return fingers


def classify(fingers, pts):
    """Return a gesture key (see LABELS) or 'UNKNOWN'. Rules are mutually exclusive."""
    t, i, m, r, p = fingers
    palm = dist(pts[WRIST], pts[MIDDLE_MCP]) or 1.0

    if all(fingers):
        return "OPEN_PALM"

    if not any(fingers):
        return "FIST"

    # Thumbs up: only thumb open and pointing upward on screen
    if t and not (i or m or r or p):
        if pts[THUMB_TIP][1] < pts[THUMB_MCP][1]:
            return "THUMB_UP"
        return "UNKNOWN"  # thumbs down / sideways

    # Index + middle up, ring + pinky down, thumb not required
    if i and m and not r and not p:
        gap = dist(pts[INDEX_TIP], pts[MIDDLE_TIP]) / palm  # normalized by hand size
        return "TWO_TOGETHER" if gap < 0.25 else "V_SIGN"

    if i and not (m or r or p) and not t:
        return "INDEX"

    if i and p and not (m or r) and not t:
        return "ROCK"

    return "UNKNOWN"


class Debouncer:
    """Majority-vote smoothing so one noisy frame does not reset everything."""

    def __init__(self):
        self.history = deque(maxlen=HISTORY_LEN)
        self.committed = "NO HAND"

    def update(self, gesture):
        """Returns the newly committed gesture if it changed, else None."""
        self.history.append(gesture)
        top, votes = Counter(self.history).most_common(1)[0]
        needed = NO_HAND_MIN_VOTES if top == "NO HAND" else GESTURE_MIN_VOTES
        if votes >= needed and top != self.committed:
            self.committed = top
            return top
        return None


def parse_args():
    ap = argparse.ArgumentParser(description="Hand Gesture to Text")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    return ap.parse_args()


def draw_text(frame, text, org, scale, color, thick):
    # Dark outline first for readability on any background
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log.error("Could not open camera index %d. Try --camera 1 or check permissions.", args.camera)
        return 1

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    debouncer = Debouncer()
    displayed = ""
    smooth_fps = 0.0
    prev_time = time.time()
    failed_reads = 0

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= 30:
                        log.error("Camera stopped delivering frames. Exiting.")
                        break
                    time.sleep(0.01)
                    continue
                failed_reads = 0

                frame = cv2.flip(frame, 1)  # mirror view (MediaPipe expects selfie-style input)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                finger_str = "-----"
                if results.multi_hand_landmarks:
                    lms = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                    pts = [(lm.x * w, lm.y * h) for lm in lms.landmark]
                    fingers = fingers_up(pts)
                    finger_str = "".join("1" if f else "0" for f in fingers)
                    key = classify(fingers, pts)
                    gesture = LABELS.get(key, "UNKNOWN")
                else:
                    gesture = "NO HAND"

                changed = debouncer.update(gesture)
                if changed is not None:
                    displayed = changed
                    if changed not in ("NO HAND", "UNKNOWN"):
                        log.info("Detected gesture: %s", changed)

                shown = displayed or "NO HAND"
                color = (0, 255, 0) if shown not in ("UNKNOWN", "NO HAND") else (0, 165, 255)
                draw_text(frame, f"Gesture: {shown}", (10, 40), 1.0, color, 2)
                draw_text(frame, f"Fingers: {finger_str}", (10, h - 40), 0.7, (200, 200, 20), 2)
                draw_text(frame, "q: quit   c: clear", (10, h - 15), 0.55, (255, 255, 255), 1)

                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    inst = 1.0 / dt
                    smooth_fps = inst if smooth_fps == 0 else 0.9 * smooth_fps + 0.1 * inst
                draw_text(frame, f"FPS: {smooth_fps:.0f}", (w - 130, 30), 0.7, (100, 255, 100), 2)

                try:
                    cv2.imshow("Hand Gesture to Text", frame)
                except cv2.error as e:
                    log.error("Cannot show window (headless OpenCV build?): %s", e)
                    break

                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("c"):
                    displayed = ""
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())