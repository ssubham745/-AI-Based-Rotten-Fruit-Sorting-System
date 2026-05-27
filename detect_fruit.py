# detect_fruit.py - SIMPLE GAP-BASED RESET VERSION
# Logic: fruit disappears for 1.5s → reset → next fruit is new
# No complex tracking, no centroid math, no bugs

import cv2
import torch
import serial
import time
import threading
import queue
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import torch.nn as nn

# ── CONFIG ──────────────────────────────────────────────
CNN_WEIGHTS     = "fruit_cnn.pth"
YOLO_MODEL      = "yolov8n.pt"
SERIAL_PORT     = "COM3"
BAUD_RATE       = 9600
CAMERA_INDEX    = 0
CONF_THRESH     = 0.5
FRUIT_CLASS_IDS = [46, 47, 49]   # banana, apple, orange
IMG_SIZE        = 224

# ── ONLY 2 SETTINGS YOU NEED TO TUNE ────────────────────
# How long fruit must be ABSENT before system resets
# Increase if reset triggers too early
# Decrease if system is too slow to reset
NO_FRUIT_RESET_SEC = 1.5

# After classifying a fruit, ignore that same fruit
# for this many seconds (prevents repeat classifications
# of the same fruit still sitting in frame)
SAME_FRUIT_LOCK_SEC = 3.0
# ────────────────────────────────────────────────────────


# ── MODEL LOADING ───────────────────────────────────────
def load_cnn(weights_path):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

cnn_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify_fruit(model, crop_bgr):
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb)
    tensor = cnn_transform(pil).unsqueeze(0)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = probs.argmax().item()
        conf  = probs[pred].item()
    return ("fresh" if pred == 0 else "rotten"), conf


# ── DRAWING ─────────────────────────────────────────────
def draw_box(frame, box, label, conf, count):
    x1,y1,x2,y2 = map(int, box)
    color = (0,210,0) if label=="fresh" else (0,0,220)

    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    tag = f"#{count} {label.upper()} {conf*100:.1f}%"
    (tw,th),_ = cv2.getTextSize(
        tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame,
                  (x1, y1-th-10),(x1+tw+8, y1),
                  color,-1)
    cv2.putText(frame, tag, (x1+4, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,(255,255,255),2)

def draw_banner(frame, label, count):
    h,w = frame.shape[:2]
    if label == "fresh":
        text  = f"#{count}  FRESH — Moving to bin"
        color = (0,130,0)
    else:
        text  = f"#{count}  ROTTEN — Ejecting!"
        color = (0,0,180)

    overlay = frame.copy()
    cv2.rectangle(overlay,(0,h-65),(w,h),color,-1)
    cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)

    (bw,_),_ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, text,
                ((w-bw)//2, h-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(255,255,255),2)

def draw_hud(frame, fps, count, state, lock_remaining):
    h,w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame,(0,0),(w,50),(25,25,25),-1)

    cv2.putText(frame, f"FPS:{fps:.0f}",
                (10,34), cv2.FONT_HERSHEY_SIMPLEX,
                0.75,(0,255,255),2)

    cv2.putText(frame, f"Scanned: {count}",
                (120,34), cv2.FONT_HERSHEY_SIMPLEX,
                0.75,(255,220,0),2)

    # State badge top-right
    state_map = {
        "READY"    : ((0,200,0),   "● READY"),
        "LOCKED"   : ((0,140,255), "● LOCKED"),
        "WAITING"  : ((180,180,0), "● WAITING"),
        "NO FRUIT" : ((120,120,120),"● SCANNING"),
    }
    sc, st = state_map.get(state,
                           ((255,255,255), state))
    cv2.putText(frame, st,
                (w-160,34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, sc, 2)

    # Lock countdown bar
    if lock_remaining > 0:
        bw = int((lock_remaining/SAME_FRUIT_LOCK_SEC)*180)
        cv2.rectangle(frame,(w-190,8),(w-190+bw,22),
                      (0,140,255),-1)
        cv2.putText(frame,
                    f"Next in {lock_remaining:.1f}s",
                    (w-190,7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,(200,200,200),1)


# ════════════════════════════════════════════════════════
#  INFERENCE THREAD
# ════════════════════════════════════════════════════════
class InferenceThread(threading.Thread):

    def __init__(self, yolo, cnn, ser):
        super().__init__(daemon=True)
        self.yolo        = yolo
        self.cnn         = cnn
        self.ser         = ser
        self.frame_q     = queue.Queue(maxsize=1)
        self.result_q    = queue.Queue(maxsize=1)
        self.running     = True

        # ── STATE VARIABLES ──────────────────────────────
        # Total fruits classified this session
        self.count           = 0

        # Time when fruit was LAST seen in frame
        # None = no fruit has been seen yet
        self.last_seen       = None

        # Time when we LAST classified a fruit
        # None = haven't classified anything yet
        self.last_classified = None

        # Is system locked? (waiting for fruit to leave)
        # True  → already classified this fruit, ignoring
        # False → ready to classify next fruit
        self.locked          = False

        # What we last classified (for display)
        self.last_label      = None
        self.last_conf       = 0.0
        self.last_box        = None

    # ── PUSH FRAME FROM MAIN THREAD ─────────────────────
    def push(self, frame):
        try:
            self.frame_q.put_nowait(frame.copy())
        except queue.Full:
            try:    self.frame_q.get_nowait()
            except: pass
            self.frame_q.put_nowait(frame.copy())

    # ── PUSH RESULT TO MAIN THREAD ───────────────────────
    def push_result(self, data):
        try:
            self.result_q.put_nowait(data)
        except queue.Full:
            try:    self.result_q.get_nowait()
            except: pass
            self.result_q.put_nowait(data)

    # ── MAIN INFERENCE LOOP ──────────────────────────────
    def run(self):
        while self.running:

            # ── GET FRAME ────────────────────────────────
            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                self._check_reset()
                continue

            now = time.time()

            # ── YOLO DETECTION ───────────────────────────
            results  = self.yolo(
                frame, conf=CONF_THRESH, verbose=False)
            fruit_found  = False
            best_box     = None
            best_crop    = None

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) not in FRUIT_CLASS_IDS:
                        continue
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    pad = 10
                    x1c = max(0,        int(x1)-pad)
                    y1c = max(0,        int(y1)-pad)
                    x2c = min(frame.shape[1], int(x2)+pad)
                    y2c = min(frame.shape[0], int(y2)+pad)
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size == 0:
                        continue
                    fruit_found = True
                    best_box    = (x1,y1,x2,y2)
                    best_crop   = crop
                    break   # take first detected fruit

            # ── FRUIT FOUND IN FRAME ─────────────────────
            if fruit_found:
                self.last_seen = now   # update last seen time

                lock_rem = 0.0
                if self.last_classified is not None:
                    lock_rem = max(0.0, SAME_FRUIT_LOCK_SEC
                                   - (now - self.last_classified))

                # ── LOCKED: already classified this fruit ─
                if self.locked:
                    # Check if lock has expired
                    if lock_rem <= 0:
                        # Lock expired but fruit still there
                        # This means it's a new fruit shown
                        # immediately after lock — unlock!
                        self.locked = False
                        print("\n  🔓 Lock expired "
                              "— treating as NEW fruit\n")
                    else:
                        # Still locked, just display
                        state = "LOCKED"
                        self.push_result({
                            "box"     : self.last_box,
                            "label"   : self.last_label,
                            "conf"    : self.last_conf,
                            "count"   : self.count,
                            "state"   : state,
                            "lock_rem": lock_rem,
                            "show_box": True
                        })
                        continue

                # ── NOT LOCKED: classify this fruit! ─────
                label, conf = classify_fruit(
                    self.cnn, best_crop)

                self.count          += 1
                self.last_label      = label
                self.last_conf       = conf
                self.last_box        = best_box
                self.last_classified = now
                self.locked          = True   # lock after classify

                # ── PRINT TO COMMAND WINDOW ───────────────
                e = "✅" if label=="fresh" else "🚨"
                print(f"\n{'═'*45}")
                print(f"  {e}  FRUIT #{self.count}")
                print(f"      Result     : {label.upper()}")
                print(f"      Confidence : {conf*100:.1f}%")
                print(f"      Action     : "
                      f"{'➡️  Moving to bin' if label=='fresh' else '⚠️  EJECTING!'}")
                print(f"{'═'*45}")
                print(f"  ⏳ Remove fruit and show next one\n")

                # ── SEND ARDUINO COMMAND ──────────────────
                cmd = 'F' if label=="fresh" else 'R'
                if self.ser:
                    self.ser.write(cmd.encode())
                    print(f"  [SERIAL] → '{cmd}' to Arduino\n")

                self.push_result({
                    "box"     : best_box,
                    "label"   : label,
                    "conf"    : conf,
                    "count"   : self.count,
                    "state"   : "LOCKED",
                    "lock_rem": SAME_FRUIT_LOCK_SEC,
                    "show_box": True
                })

            # ── NO FRUIT IN FRAME ────────────────────────
            else:
                self._check_reset()
                lock_rem = 0.0
                if self.last_classified:
                    lock_rem = max(0.0, SAME_FRUIT_LOCK_SEC
                               - (now - self.last_classified))
                state = "WAITING" if self.locked else "NO FRUIT"
                self.push_result({
                    "box"     : None,
                    "label"   : self.last_label,
                    "conf"    : self.last_conf,
                    "count"   : self.count,
                    "state"   : state,
                    "lock_rem": lock_rem,
                    "show_box": False
                })

    def _check_reset(self):
        """
        If fruit has been absent for NO_FRUIT_RESET_SEC,
        unlock the system so it's ready for the next fruit.
        This is the CORE of the whole logic.
        """
        if self.locked and self.last_seen is not None:
            gap = time.time() - self.last_seen
            if gap >= NO_FRUIT_RESET_SEC:
                self.locked = False
                print(f"\n  ✅ Fruit removed! "
                      f"System RESET — ready for next fruit!")
                print(f"  Show next fruit now...\n")
                print("─"*45)

    def stop(self):
        self.running = False


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════
if __name__ == '__main__':

    print("="*45)
    print("  🍎  Fruit Quality Detection System")
    print("="*45)

    print("Loading YOLO...")
    yolo = YOLO(YOLO_MODEL)

    print("Loading CNN...")
    cnn = load_cnn(CNN_WEIGHTS)

    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✅ Arduino → {SERIAL_PORT}\n")
    except Exception as e:
        print(f"⚠️  No Arduino ({e})\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ Camera error!")
        exit()

    worker = InferenceThread(yolo, cnn, ser)
    worker.start()

    print("✅ System LIVE!")
    print("─"*45)
    #print("  HOW TO USE:")
    #print("  1. Show a fruit → system detects & classifies")
    #print("  2. Remove fruit from camera view")
    #print("  3. Show next fruit → repeat")
    #print("  Press Q to quit")
    #print("─"*45 + "\n")

    fps        = 0.0
    fcount     = 0
    fps_start  = time.time()

    latest = {
        "box":None, "label":None, "conf":0,
        "count":0,  "state":"NO FRUIT",
        "lock_rem":0, "show_box":False
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # FPS
        fcount += 1
        el = time.time() - fps_start
        if el >= 1.0:
            fps       = fcount / el
            fcount    = 0
            fps_start = time.time()

        # Send frame to worker
        worker.push(frame)

        # Get latest result (non-blocking)
        try:
            latest = worker.result_q.get_nowait()
        except queue.Empty:
            pass

        # ── DRAW ─────────────────────────────────────────
        if latest["show_box"] and latest["box"]:
            draw_box(frame,
                     latest["box"],
                     latest["label"],
                     latest["conf"],
                     latest["count"])
            draw_banner(frame,
                        latest["label"],
                        latest["count"])

        elif latest["state"] == "WAITING":
            h,w = frame.shape[:2]
            cv2.putText(frame,
                        "Remove fruit to scan next...",
                        (10, h-15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,(0,165,255),2)

        elif latest["state"] == "NO FRUIT":
            cv2.putText(frame,
                        "Show a fruit to the camera",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(150,150,150),1)

        draw_hud(frame, fps,
                 latest["count"],
                 latest["state"],
                 latest["lock_rem"])

        cv2.imshow("Fruit Quality Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    worker.stop()
    worker.join(timeout=3)
    cap.release()
    if ser: ser.close()
    cv2.destroyAllWindows()
    print(f"\nSession done. "
          f"Total scanned: {latest['count']}")#