import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import threading
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import csv
import os

mp_holistic = mp.solutions.holistic

# ---------------- Configurable Parameters ----------------
EAR_WINDOW = 7
ENGAGEMENT_WINDOW = 10
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.21
FRAME_WIDTH = 640  # resize width for speed
LOG_FILE = "engagement_log.csv"

# ---------------- Rolling windows ----------------
ear_history = deque(maxlen=EAR_WINDOW)
emotion_history = deque(maxlen=ENGAGEMENT_WINDOW)
pose_history = deque(maxlen=ENGAGEMENT_WINDOW)
gesture_history = deque(maxlen=ENGAGEMENT_WINDOW)
blink_timestamps = deque(maxlen=200)


# ---------------- Placeholder emotion model ----------------
# Replace predict_emotion with a real model call when available.
def predict_emotion(face_img):
    # face_img: BGR image crop
    # TODO: integrate an emotion classifier
    return "Neutral"


# ---------------- Video Stream Class (threaded) ----------------
class VideoStream:
    def __init__(self, src=0, width=FRAME_WIDTH):
        self.cap = cv2.VideoCapture(src)
        # try set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self._lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self._lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass


# ---------------- EAR Calculation ----------------
def eye_aspect_ratio(landmarks, eye_indices):
    # landmarks: list-like of mp landmark objects (with .x, .y)
    try:
        p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    except Exception:
        return 0.0

    # use vertical / horizontal distances similar to standard EAR formula
    v1 = np.linalg.norm(p2 - p4)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p6)

    denom = (2.0 * h)
    if denom == 0:
        return 0.0
    ear = (v1 + v2) / denom
    return float(ear)


# ---------------- Head Pose (simple heuristic) ----------------
def get_head_pose(landmarks, img_w, img_h):
    # Very simple heuristic based on nose and eye heights
    try:
        nose_tip = (int(landmarks[1].x * img_w), int(landmarks[1].y * img_h))
        left_eye = (int(landmarks[33].x * img_w), int(landmarks[33].y * img_h))
        right_eye = (int(landmarks[263].x * img_w), int(landmarks[263].y * img_h))
    except Exception:
        return "Unknown"

    # if eyes are at very different vertical level -> side tilt
    if abs(left_eye[1] - right_eye[1]) > img_h * 0.03:
        return "Looking Side"
    # if nose x is far to left/right fraction of image
    if nose_tip[0] < img_w * 0.28 or nose_tip[0] > img_w * 0.72:
        return "Looking Away"
    return "Looking Forward"


# ---------------- Gesture Detection ----------------
def detect_gesture(hand_landmarks, is_left=False):
    if hand_landmarks is None:
        return "No Hand"

    lm = hand_landmarks.landmark
    coords = np.array([(p.x, p.y) for p in lm])

    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    hand_height = max_xy[1] - min_xy[1]
    hand_height = max(hand_height, 1e-5)

    fingers_extended = 0

    # Thumb
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]
    thumb_len = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([thumb_mcp.x, thumb_mcp.y])) / hand_height
    thumb_side_margin = 0.02
    thumb_extended = False
    if is_left:
        if thumb_tip.x < thumb_ip.x - thumb_side_margin and thumb_len > 0.16:
            thumb_extended = True
    else:
        if thumb_tip.x > thumb_ip.x + thumb_side_margin and thumb_len > 0.16:
            thumb_extended = True
    if thumb_extended:
        fingers_extended += 1

    # Other fingers
    finger_triplets = [
        (8, 6, 5),
        (12, 10, 9),
        (16, 14, 13),
        (20, 18, 17),
    ]
    for tip_idx, pip_idx, mcp_idx in finger_triplets:
        tip = lm[tip_idx]
        pip = lm[pip_idx]
        mcp = lm[mcp_idx]

        upward = tip.y < pip.y - 0.01
        finger_len = np.linalg.norm(np.array([tip.x, tip.y]) - np.array([mcp.x, mcp.y])) / hand_height
        long_enough = finger_len > 0.28
        if upward and long_enough:
            fingers_extended += 1

    return "Open" if fingers_extended >= 3 else "Closed"


# ---------------- CSV Logging ----------------
def init_log_file(log_path=LOG_FILE):
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "engagement", "ear", "head_pose", "emotion", "gesture", "blink_per_min"])


def log_engagement_row(engagement, ear, head_pose, emotion, gesture, blink_rate, log_path=LOG_FILE):
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            engagement,
            f"{ear:.4f}" if ear is not None else "",
            head_pose if head_pose is not None else "",
            emotion if emotion is not None else "",
            gesture if gesture is not None else "",
            blink_rate
        ])


# ---------------- Rounded Button (UI) ----------------
class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None, width=80, height=30, radius=14, bg="#2563eb", fg="white", hover="#1d4ed8", font=("Segoe UI", 10, "bold")):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0)
        self.command = command
        self.bg = bg
        self.fg = fg
        self.hover = hover
        self.radius = radius
        self.current_color = bg
        self.rounded_rect = self.create_rounded_rect(0, 0, width, height, radius, fill=bg, outline=bg)
        self.text_item = self.create_text(width // 2, height // 2, text=text, fill=fg, font=font)
        self.bind("<Button-1>", lambda e: self._onclick())
        self.bind("<Enter>", lambda e: self._on_enter())
        self.bind("<Leave>", lambda e: self._on_leave())

    def _onclick(self):
        if self.command:
            try:
                self.command()
            except Exception as e:
                print("Button command error:", e)

    @staticmethod
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb):
        return "#%02x%02x%02x" % rgb

    def animate_to(self, target_hex, steps=6, delay=10):
        start_rgb = self._hex_to_rgb(self.current_color)
        end_rgb = self._hex_to_rgb(target_hex)
        def step_animation(step=0):
            if step > steps:
                self.current_color = target_hex
                self.itemconfig(self.rounded_rect, fill=target_hex, outline=target_hex)
                return
            ratio = step / steps
            new_rgb = tuple(int(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * ratio) for i in range(3))
            new_hex = self._rgb_to_hex(new_rgb)
            self.current_color = new_hex
            self.itemconfig(self.rounded_rect, fill=new_hex, outline=new_hex)
            self.after(delay, lambda: step_animation(step + 1))
        step_animation()

    def _on_enter(self):
        self.animate_to(self.hover)

    def _on_leave(self):
        self.animate_to(self.bg)

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)


# ---------------- Tkinter GUI ----------------
class EngagementGUI(tk.Tk):
    def __init__(self, stream, holistic, log_path=LOG_FILE):
        super().__init__()
        self.stream = stream
        self.holistic = holistic
        self.log_path = log_path
        self.title("Real-Time Engagement Detection")
        self.configure(bg="#111827")
        self.is_running = False
        self.last_log_time = 0.0
        self.engagement_counts = {}
        self.session_start_time = None
        init_log_file(self.log_path)

        # Header
        header_frame = tk.Frame(self, bg="#020617")
        header_frame.pack(fill=tk.X)
        header_label = tk.Label(header_frame, text="Real-Time Student Engagement Monitor", font=("Segoe UI", 14, "bold"), fg="#e5e7eb", bg="#020617", pady=6)
        header_label.pack(side=tk.LEFT, padx=(8, 0))
        self.live_label = tk.Label(header_frame, text="● IDLE", font=("Segoe UI", 10, "bold"), fg="#9ca3af", bg="#020617", pady=6)
        self.live_label.pack(side=tk.RIGHT, padx=(0, 12))

        # Main
        main_frame = tk.Frame(self, bg="#111827")
        main_frame.pack(padx=10, pady=10)
        video_container = tk.Frame(main_frame, bg="#020617", bd=2, relief="flat")
        video_container.pack(side=tk.LEFT, padx=(0, 10))
        self.video_label = tk.Label(video_container, bg="#020617")
        self.video_label.pack()

        stats_frame = tk.Frame(main_frame, bg="#020617", bd=1, relief="solid", padx=12, pady=10)
        stats_frame.pack(side=tk.LEFT, fill=tk.Y)
        stats_title = tk.Label(stats_frame, text="Live Metrics", font=("Segoe UI", 11, "bold"), bg="#020617", fg="#e5e7eb")
        stats_title.grid(row=0, column=0, columnspan=2, pady=(0, 8))

        def add_stat(label, row):
            lbl = tk.Label(stats_frame, text=label, font=("Segoe UI", 9), bg="#020617", fg="#9ca3af")
            lbl.grid(row=row, column=0, pady=3, sticky="w")
            val = tk.Label(stats_frame, text="-", font=("Segoe UI", 9, "bold"), bg="#020617", fg="#e5e7eb")
            val.grid(row=row, column=1, pady=3, sticky="w", padx=(6, 0))
            return val

        self.stat_engagement = add_stat("Engagement:", 1)
        self.stat_ear = add_stat("EAR:", 2)
        self.stat_head = add_stat("Head Pose:", 3)
        self.stat_emotion = add_stat("Emotion:", 4)
        self.stat_gesture = add_stat("Gesture:", 5)
        self.stat_blink = add_stat("Blink/min:", 6)

        att_label = tk.Label(stats_frame, text="Attentive %", font=("Segoe UI", 9), bg="#020617", fg="#9ca3af")
        att_label.grid(row=7, column=0, pady=(10, 2), sticky="w")
        self.gauge_canvas = tk.Canvas(stats_frame, width=160, height=10, bg="#020617", highlightthickness=0, bd=0)
        self.gauge_canvas.grid(row=7, column=1, sticky="w", pady=(10, 2))
        self.draw_gauge(0.0)

        control_frame = tk.Frame(self, bg="#111827")
        control_frame.pack(fill=tk.X, pady=8)
        self.timer_label = tk.Label(control_frame, text="⏱ 00:00", font=("Segoe UI", 10), bg="#111827", fg="#9ca3af")
        self.timer_label.pack(side=tk.LEFT, padx=(8, 0))
        RoundedButton(control_frame, "Start", self.start_detection, bg="#22c55e", hover="#16a34a").pack(side=tk.RIGHT, padx=6)
        RoundedButton(control_frame, "Stop", self.stop_detection, bg="#f97316", hover="#ea580c").pack(side=tk.RIGHT, padx=6)
        RoundedButton(control_frame, "Quit", self.on_closing, bg="#3b82f6", hover="#2563eb").pack(side=tk.RIGHT, padx=6)

        self.status_label = tk.Label(self, text="Status: Idle (Press Start)", font=("Segoe UI", 10), bg="#111827", fg="#e5e7eb")
        self.status_label.pack(pady=(0, 8))

        self.update_frame()

    def draw_gauge(self, ratio):
        self.gauge_canvas.delete("all")
        w = int(self.gauge_canvas["width"])
        h = int(self.gauge_canvas["height"])
        gauge_bg = "#0f1724"
        gauge_fill = "#60a5fa"
        self.gauge_canvas.create_rectangle(0, 0, w, h, fill=gauge_bg, outline=gauge_bg)
        fill_w = int(w * max(0.0, min(1.0, ratio)))
        if fill_w > 0:
            self.gauge_canvas.create_rectangle(0, 0, fill_w, h, fill=gauge_fill, outline=gauge_fill)

    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self.status_label.config(text="Status: Running (Logging to CSV)")
            self.last_log_time = 0.0
            self.engagement_counts = {}
            self.session_start_time = time.time()
            self.live_label.config(text="● LIVE", fg="#16a34a")

    def stop_detection(self):
        if self.is_running:
            self.is_running = False
            self.status_label.config(text="Status: Idle (Press Start)")
            self.live_label.config(text="● IDLE", fg="#9ca3af")
            self.show_session_summary()
            ear_history.clear()
            emotion_history.clear()
            pose_history.clear()
            gesture_history.clear()
            blink_timestamps.clear()
            self.session_start_time = None

    def show_session_summary(self):
        total = sum(self.engagement_counts.values())
        if total == 0:
            messagebox.showinfo("Session Summary", "No engagement data recorded for this session.")
            return
        labels_order = ["Attentive", "Distracted", "Drowsy", "Passive", "Not Performing", "No Face Detected"]
        lines = []
        for label in labels_order:
            count = self.engagement_counts.get(label, 0)
            if count > 0:
                perc = (count / total) * 100.0
                lines.append(f"{label}: {perc:.1f}%")
        for label, count in self.engagement_counts.items():
            if label not in labels_order:
                perc = (count / total) * 100.0
                lines.append(f"{label}: {perc:.1f}%")
        msg = "Engagement distribution for this session:\n\n" + "\n".join(lines)
        messagebox.showinfo("Session Summary", msg)

    def update_frame(self):
        try:
            ret, frame = self.stream.read()
            if frame is None or not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            engagement = "Idle - Press Start" if not self.is_running else "No Face Detected"
            ear_value = pose = emotion = gesture = None
            blink_rate = 0

            if self.is_running:
                img_h, img_w = frame.shape[:2]
                # optionally resize for speed
                if img_w != FRAME_WIDTH:
                    scale = FRAME_WIDTH / float(img_w)
                    frame = cv2.resize(frame, (FRAME_WIDTH, int(img_h * scale)))
                    img_h, img_w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.holistic.process(rgb)

                if res.face_landmarks:
                    lm = res.face_landmarks.landmark
                    le = eye_aspect_ratio(lm, LEFT_EYE)
                    re = eye_aspect_ratio(lm, RIGHT_EYE)
                    ear_value = (le + re) / 2.0
                    ear_history.append(ear_value)

                    pose = get_head_pose(lm, img_w, img_h)
                    pose_history.append(pose)

                    pts = np.array([(int(l.x * img_w), int(l.y * img_h)) for l in lm])
                    x, y, w, h = cv2.boundingRect(pts)
                    if w > 0 and h > 0:
                        # add some padding but clamp to frame
                        pad = int(0.08 * max(w, h))
                        x0 = max(0, x - pad)
                        y0 = max(0, y - pad)
                        x1 = min(img_w, x + w + pad)
                        y1 = min(img_h, y + h + pad)
                        face_crop = frame[y0:y1, x0:x1]
                    else:
                        face_crop = frame
                    emotion = predict_emotion(face_crop)
                    emotion_history.append(emotion)

                    if len(ear_history) >= 2 and ear_history[-2] > EAR_THRESHOLD and ear_history[-1] < EAR_THRESHOLD:
                        blink_timestamps.append(time.time())

                    if res.left_hand_landmarks:
                        gesture = detect_gesture(res.left_hand_landmarks, is_left=True)
                    elif res.right_hand_landmarks:
                        gesture = detect_gesture(res.right_hand_landmarks, is_left=False)
                    else:
                        gesture = "No Hand"
                    gesture_history.append(gesture)

                    blink_rate = len([t for t in blink_timestamps if time.time() - t < 60])

                    avg = np.mean(ear_history) if len(ear_history) > 0 else 1.0
                    if avg < EAR_THRESHOLD and len(ear_history) == EAR_WINDOW:
                        engagement = "Drowsy"
                    elif pose != "Looking Forward":
                        engagement = "Distracted"
                    elif emotion in ["angry", "sad", "fear", "disgust"]:
                        engagement = "Not Performing"
                    elif gesture == "Closed":
                        engagement = "Passive"
                    else:
                        engagement = "Attentive"
                else:
                    engagement = "No Face Detected"

                now = time.time()
                if now - self.last_log_time >= 1.0:
                    try:
                        log_engagement_row(engagement, ear_value, pose, emotion, gesture, blink_rate, self.log_path)
                    except Exception as e:
                        print("Logging error:", e)
                    self.last_log_time = now

                self.engagement_counts[engagement] = self.engagement_counts.get(engagement, 0) + 1

            # Display frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

            # Update stats
            self.stat_engagement.config(text=engagement if self.is_running else "-")
            self.stat_ear.config(text=f"{ear_value:.2f}" if (self.is_running and ear_value is not None) else "-")
            self.stat_head.config(text=pose if (self.is_running and pose is not None) else "-")
            self.stat_emotion.config(text=emotion if (self.is_running and emotion is not None) else "-")
            self.stat_gesture.config(text=gesture if (self.is_running and gesture is not None) else "-")
            self.stat_blink.config(text=str(blink_rate) if self.is_running else "-")

            # Timer
            if self.is_running and self.session_start_time:
                elapsed = int(time.time() - self.session_start_time)
                mins = elapsed // 60
                secs = elapsed % 60
                self.timer_label.config(text=f"⏱ {mins:02d}:{secs:02d}")
            else:
                self.timer_label.config(text="⏱ 00:00")

            # Gauge
            total = sum(self.engagement_counts.values())
            ratio = (self.engagement_counts.get("Attentive", 0) / total) if total > 0 else 0.0
            self.draw_gauge(ratio)
            self.status_label.configure(text=f"Status: {engagement}")

        except Exception as e:
            print("Frame update error:", e)

        self.after(30, self.update_frame)

    def on_closing(self):
        try:
            self.stream.release()
        except Exception:
            pass
        try:
            self.holistic.close()
        except Exception:
            pass
        self.destroy()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        stream = VideoStream(0, width=FRAME_WIDTH)
        holistic = mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        app = EngagementGUI(stream, holistic)
        app.mainloop()
    except Exception as e:
        print("Fatal error:", e)
    finally:
        try:
            stream.release()
        except Exception:
            pass
        try:
            holistic.close()
        except Exception:
            pass