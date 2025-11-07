"""
Focus Flow Optimizer — Real-Time Predictor (Final, feature-compatible & stable)
-------------------------------------------------------------------------------
- Loads *latest* model & encoder
- Reads best_threshold AND expected feature list from reports/metrics.json
- Computes engineered features to match training (Total_Input, Key_Mouse_Ratio, Is_Active*, App_Group_Enc)
- Uses predict_proba + learned threshold
- Adds stability: min activity, ignore mouse-only noise, consecutive confirmation,
  never-mute apps, robust active-window fallback
- Logs to data/session_log.csv

Run: python src/flow_optimizer.py
"""

import os
import sys
import time
import csv
import json
import ctypes
import joblib
import psutil
import pandas as pd
import threading
from datetime import datetime
from pynput import keyboard, mouse

# Optional toast notifications (Windows)
try:
    from win10toast import ToastNotifier
    TOASTER = ToastNotifier()
    TOAST_OK = True
except Exception:
    TOASTER = None
    TOAST_OK = False

import win32gui  # Windows active window APIs

# =========================
# Configuration
# =========================
PREDICT_INTERVAL_SECONDS = 5
BASE_COOLDOWN_SECONDS = 180          # min 3 minutes between actions
IDLE_PRINT_EVERY = 3                 # print idle heartbeat every N cycles
USE_SYSTEM_MUTE = False              # set True for real mute/unmute
SESSION_LOG_PATH = os.path.join("data", "session_log.csv")

# Stability controls
MIN_ACTIVITY = 2                     # require at least 2 total events to predict
IGNORE_MOUSE_ONLY = True             # treat pure-mouse as idle noise
REQUIRED_CONSECUTIVE = 2             # need N consecutive decisions before acting

# Never mute on these apps (safety net)
NEVER_MUTE = {"zoom.exe", "teams.exe", "skype.exe", "ms-teams.exe"}

# =========================
# Globals
# =========================
input_lock = threading.Lock()
model_lock = threading.Lock()

KEY_COUNT = 0
MOUSE_COUNT = 0
keyboard_listener = None
mouse_listener = None

IS_MUTED = False
LAST_ACTION_TIME = 0.0
IDLE_COUNTER = 0

# Consecutive decision counters
_CONSEC_FLOW = 0
_CONSEC_DIST = 0

# Track last-known app when WinAPI returns N/A
_LAST_APP = "unknown.exe"

MODEL = None
APP_ENCODER = None
DECISION_THRESHOLD = 0.65
EXPECTED_FEATURES = ["Key_Rate", "Mouse_Rate", "Active_App_Encoded"]  # fallback

# =========================
# Paths
# =========================
def project_root():
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def model_paths():
    root = project_root()
    return (
        os.path.join(root, "models", "focus_model_latest.pkl"),
        os.path.join(root, "models", "app_encoder_latest.pkl"),
    )

def metrics_path():
    return os.path.join(project_root(), "reports", "metrics.json")

# =========================
# Notifications & Audio
# =========================
def notify(title: str, message: str, duration: int = 4):
    if TOAST_OK:
        try:
            TOASTER.show_toast(title=title, message=message, duration=duration)
        except Exception:
            pass

def _sys_mute_windows(mute: bool) -> bool:
    try:
        if mute:
            ctypes.windll.winmm.waveOutSetVolume(0, 0)
        else:
            ctypes.windll.winmm.waveOutSetVolume(0, 0xFFFFFFFF)
        return True
    except Exception:
        return False

def set_audio_state(mute: bool):
    global IS_MUTED
    if IS_MUTED == mute:
        return False
    IS_MUTED = mute
    action = "MUTED" if mute else "UNMUTED"
    if USE_SYSTEM_MUTE:
        ok = _sys_mute_windows(mute)
        print(f"🔊 ACTION: {'System' if ok else '(FAILED) Simulated'} audio {action}")
    else:
        print(f"🔊 ACTION: Simulated audio {action}")
    notify("Focus Flow Optimizer", "Audio muted to help you focus." if mute else "Audio restored. Keep the flow!")
    return True

# =========================
# System info
# =========================
def get_active_window_info():
    """Returns (window_title, process_name) with robust fallbacks."""
    global _LAST_APP
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return "N/A", _LAST_APP
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32gui.GetWindowThreadProcessId(hwnd)
        proc_name = psutil.Process(pid).name()
        if proc_name:
            _LAST_APP = proc_name
        return title or "N/A", proc_name or _LAST_APP
    except Exception:
        # Keep using last known app so features remain consistent
        return "N/A", _LAST_APP

def on_press(key):
    global KEY_COUNT
    with input_lock:
        KEY_COUNT += 1

def on_click(x, y, button, pressed):
    if pressed:
        global MOUSE_COUNT
        with input_lock:
            MOUSE_COUNT += 1

def on_scroll(x, y, dx, dy):
    global MOUSE_COUNT
    with input_lock:
        MOUSE_COUNT += 1

# =========================
# Loaders
# =========================
def load_metrics():
    global DECISION_THRESHOLD, EXPECTED_FEATURES
    try:
        with open(metrics_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        DECISION_THRESHOLD = float(data.get("best_threshold", DECISION_THRESHOLD))
        feats = data.get("features")
        if isinstance(feats, list) and len(feats) > 0:
            EXPECTED_FEATURES = feats
        print(f"✅ Using decision threshold: {DECISION_THRESHOLD:.3f}")
        print(f"✅ Expected features ({len(EXPECTED_FEATURES)}): {EXPECTED_FEATURES}")
    except Exception as e:
        print(f"⚠️ Could not read metrics.json, using defaults. ({e})")

def load_model_and_encoder():
    global MODEL, APP_ENCODER
    mpath, epath = model_paths()
    try:
        MODEL = joblib.load(mpath)
        APP_ENCODER = joblib.load(epath)
        print(f"✅ Loaded model & encoder from: {os.path.dirname(mpath)}")
        return True
    except FileNotFoundError as e:
        print("❌ Model files not found. Train first with: python src/model_trainer.py")
        print(f"Missing: {e.filename}")
        print(f"Expected: {mpath} / {epath}")
        return False
    except Exception as e:
        print(f"❌ Error loading model/encoder: {e}")
        return False

# =========================
# Feature engineering (match trainer)
# =========================
DEV_APPS  = {a.lower() for a in ["code.exe","pycharm64.exe","idea64.exe","notepad++.exe","excel.exe","notepad.exe","sublime_text.exe","devenv.exe"]}
COMM_APPS = {a.lower() for a in ["chrome.exe","msedge.exe","firefox.exe","slack.exe","teams.exe","outlook.exe","discord.exe","whatsapp.exe","telegram.exe"]}

# LabelEncoder in trainer maps classes in sorted order; for {'comm','dev','other'} that’s typically:
APP_GROUP_MAP = {"comm": 0, "dev": 1, "other": 2}

def app_group_of(proc_name: str) -> str:
    p = (proc_name or "").lower()
    if p in DEV_APPS:  return "dev"
    if p in COMM_APPS: return "comm"
    return "other"

def build_feature_row(keys: int, mouse: int, app_proc: str, app_encoded: int) -> dict:
    """Return a dict containing ALL possible features; we will select/arrange later to match EXPECTED_FEATURES."""
    total_input = keys + mouse
    ratio = keys / (mouse + 1)
    ratio = min(max(ratio, 0.0), 50.0)  # clip like trainer

    group = app_group_of(app_proc)
    group_enc = APP_GROUP_MAP.get(group, 2)

    base = {
        "Key_Rate": keys,
        "Mouse_Rate": mouse,
        "Active_App_Encoded": int(app_encoded),
        "Total_Input": total_input,
        "Key_Mouse_Ratio": ratio,
        "Is_ActiveKeys": int(keys > 0),
        "Is_ActiveMouse": int(mouse > 0),
        "App_Group_Enc": group_enc,
    }
    return base

def row_to_dataframe(full_row: dict) -> pd.DataFrame:
    """Build X in the exact column order the model expects; fill missing ones with 0."""
    row = {f: full_row.get(f, 0) for f in EXPECTED_FEATURES}
    return pd.DataFrame([row], columns=EXPECTED_FEATURES)

# =========================
# Logging
# =========================
def log_session_row(ts, app, keys, mouse, pred, conf):
    try:
        os.makedirs(os.path.dirname(SESSION_LOG_PATH), exist_ok=True)
        header_needed = not os.path.exists(SESSION_LOG_PATH)
        with open(SESSION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["Timestamp", "Active_App", "Key_Rate", "Mouse_Rate", "Predicted_State", "Confidence"])
            w.writerow([ts, app, keys, mouse, "Flow" if pred == 1 else "Distracted", round(conf, 3)])
    except Exception:
        pass

# =========================
# Predict loop
# =========================
def predict_once():
    global KEY_COUNT, MOUSE_COUNT, IDLE_COUNTER, LAST_ACTION_TIME, _CONSEC_FLOW, _CONSEC_DIST

    # snapshot
    with input_lock:
        keys, mouse = KEY_COUNT, MOUSE_COUNT
        KEY_COUNT = 0
        MOUSE_COUNT = 0

    total_events = keys + mouse

    # Treat tiny/noisy movement as idle
    if total_events == 0 or total_events < MIN_ACTIVITY or (IGNORE_MOUSE_ONLY and keys == 0 and mouse > 0):
        IDLE_COUNTER += 1
        if IDLE_COUNTER >= IDLE_PRINT_EVERY:
            print(f"[{time.strftime('%H:%M:%S')}] Awaiting meaningful activity…")
            IDLE_COUNTER = 0
        return

    IDLE_COUNTER = 0
    _, app = get_active_window_info()
    app_lc = (app or "unknown.exe").lower()

    # encode app
    try:
        app_encoded = APP_ENCODER.transform([app])[0]
    except Exception:
        app_encoded = 0

    # build features to match training
    features_all = build_feature_row(keys, mouse, app, app_encoded)
    X = row_to_dataframe(features_all)

    # predict proba
    with model_lock:
        proba = MODEL.predict_proba(X)[0]
    flow_prob = float(proba[1])
    pred = 1 if flow_prob >= DECISION_THRESHOLD else 0

    # Guard: never mute on critical comm apps
    if app_lc in NEVER_MUTE:
        pred = 1  # treat as Flow to avoid muting in calls

    # Guard: if last-known app is unknown and we're near threshold, lean safe
    if _LAST_APP == "unknown.exe" and abs(flow_prob - DECISION_THRESHOLD) < 0.1:
        pred = 1

    # Require consecutive consistent decisions before we allow an action
    if pred == 1:
        _CONSEC_FLOW += 1
        _CONSEC_DIST = 0
    else:
        _CONSEC_DIST += 1
        _CONSEC_FLOW = 0

    can_commit = (_CONSEC_FLOW >= REQUIRED_CONSECUTIVE) or (_CONSEC_DIST >= REQUIRED_CONSECUTIVE)

    # cooldown (longer if close to threshold)
    distance = abs(flow_prob - DECISION_THRESHOLD)
    cooldown = BASE_COOLDOWN_SECONDS + (1 - distance) * 45
    can_act = (time.time() - LAST_ACTION_TIME) > cooldown

    # Decision & action
    if pred == 1:
        if IS_MUTED and can_act and can_commit:
            if set_audio_state(False):
                LAST_ACTION_TIME = time.time()
            banner, tip = "✅ FLOW ACHIEVED", "Audio restored. Maintain momentum!"
        else:
            banner, tip = "🟢 FLOW STATE", "Stay focused!"
    else:
        if (not IS_MUTED) and can_act and can_commit:
            if set_audio_state(True):
                LAST_ACTION_TIME = time.time()
            banner, tip = "🛑 DISTRACTION DETECTED", "Audio muted to help you refocus."
        else:
            banner, tip = "🔴 DISTRACTED", f"Muted={IS_MUTED}. Regain focus."

    prob_pct = f"{flow_prob*100:.1f}%"
    print("-" * 55)
    print(f"[{time.strftime('%H:%M:%S')}] {banner}")
    print(f"   App: {app} | Keys: {keys} | Mouse: {mouse} | Flow prob: {prob_pct} | Thr: {DECISION_THRESHOLD:.2f}")
    print(f"   {tip}")
    print("-" * 55)

    log_session_row(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), app, keys, mouse, pred, flow_prob)

# =========================
# Main
# =========================
def start_optimizer():
    global keyboard_listener, mouse_listener

    print("\n--- Focus Flow Optimizer LIVE ---")
    print(f"Predict every {PREDICT_INTERVAL_SECONDS}s | System mute: {USE_SYSTEM_MUTE}\n")

    if not load_model_and_encoder():
        return
    load_metrics()  # threshold + expected features

    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    keyboard_listener.start()
    mouse_listener.start()
    print("Input listeners started.")

    while True:
        predict_once()
        time.sleep(PREDICT_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        start_optimizer()
    except KeyboardInterrupt:
        print("\n--- Optimizer stopping ---")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        try:
            if IS_MUTED:
                _sys_mute_windows(False)
                print("System audio restored.")
        except Exception:
            pass
        if keyboard_listener and getattr(keyboard_listener, "running", False):
            keyboard_listener.stop()
        if mouse_listener and getattr(mouse_listener, "running", False):
            mouse_listener.stop()
        print("✅ Clean exit. Listeners stopped.")
