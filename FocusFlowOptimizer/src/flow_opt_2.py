"""
Focus Flow Optimizer — Windows GUI App
--------------------------------------
Standalone GUI that shows real-time focus/distracted logs.
Double-click to run (once converted to .exe).
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
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk

# Optional notifications
try:
    from win10toast import ToastNotifier
    TOASTER = ToastNotifier()
    TOAST_OK = True
except Exception:
    TOASTER = None
    TOAST_OK = False

import win32gui  # Windows API

# =========================
# Config
# =========================
PREDICT_INTERVAL_SECONDS = 5
BASE_COOLDOWN_SECONDS = 180
USE_SYSTEM_MUTE = False
SESSION_LOG_PATH = os.path.join(os.getcwd(), "session_log.csv")

MIN_ACTIVITY = 2
IGNORE_MOUSE_ONLY = True
REQUIRED_CONSECUTIVE = 2
NEVER_MUTE = {"zoom.exe", "teams.exe", "skype.exe", "ms-teams.exe"}

# =========================
# Globals
# =========================
input_lock = threading.Lock()
model_lock = threading.Lock()
KEY_COUNT = 0
MOUSE_COUNT = 0
IS_MUTED = False
LAST_ACTION_TIME = 0.0
_CONSEC_FLOW = 0
_CONSEC_DIST = 0
_LAST_APP = "unknown.exe"

MODEL = None
APP_ENCODER = None
DECISION_THRESHOLD = 0.5
EXPECTED_FEATURES = [
    "Key_Rate", "Mouse_Rate", "Active_App_Encoded",
    "Total_Input", "Key_Mouse_Ratio",
    "Is_ActiveKeys", "Is_ActiveMouse", "App_Group_Enc"
]

# =========================
# GUI Setup
# =========================
class FocusApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 Focus Flow Optimizer")
        self.geometry("900x600")
        self.configure(bg="#0c5843")

        ttk.Style().configure("TLabel", foreground="white", background="#0c5843", font=("Segoe UI", 11))
        ttk.Style().configure("TButton", font=("Segoe UI", 11, "bold"))

        tk.Label(self, text="🧠 Focus Flow Optimizer — Real-Time Logs",
                 font=("Segoe UI", 16, "bold"), fg="white", bg="#0c5843").pack(pady=10)

        self.text_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=25,
                                                   bg="#002b20", fg="#00ffae", font=("Consolas", 10))
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.btn = ttk.Button(self, text="Start Optimizer", command=self.start_optimizer)
        self.btn.pack(pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, msg):
        self.text_area.insert(tk.END, msg + "\n")
        self.text_area.see(tk.END)
        self.update()

    def start_optimizer(self):
        self.btn.config(state=tk.DISABLED, text="Running...")
        threading.Thread(target=optimizer_main, args=(self,), daemon=True).start()

    def on_close(self):
        global running
        running = False
        self.destroy()


# =========================
# Backend Functions
# =========================
def notify(title, msg):
    if TOAST_OK:
        try:
            TOASTER.show_toast(title, msg, duration=3)
        except:
            pass


def _sys_mute_windows(mute):
    try:
        if mute:
            ctypes.windll.winmm.waveOutSetVolume(0, 0)
        else:
            ctypes.windll.winmm.waveOutSetVolume(0, 0xFFFFFFFF)
        return True
    except Exception:
        return False


def set_audio_state(mute, gui):
    global IS_MUTED
    if IS_MUTED == mute:
        return
    IS_MUTED = mute
    action = "MUTED" if mute else "UNMUTED"
    if USE_SYSTEM_MUTE:
        ok = _sys_mute_windows(mute)
        gui.log(f"🔊 ACTION: {'System' if ok else '(Simulated)'} audio {action}")
    else:
        gui.log(f"🔊 ACTION: Simulated audio {action}")
    notify("Focus Flow Optimizer", f"Audio {action.lower()} to help you focus.")


def get_active_window_info():
    global _LAST_APP
    try:
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32gui.GetWindowThreadProcessId(hwnd)
        proc = psutil.Process(pid)
        name = proc.name()
        if name:
            _LAST_APP = name
        return _LAST_APP
    except:
        return _LAST_APP


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


def load_model_and_encoder(gui):
    global MODEL, APP_ENCODER
    base = os.getcwd()
    model_path = os.path.join(base, "models", "C:\\Users\\swapnilk\\Documents\\FocusFlowOptimizer\\models\\focus_model_latest.pkl")
    enc_path = os.path.join(base, "models", "C:\\Users\\swapnilk\\Documents\\FocusFlowOptimizer\\models\\app_encoder.pkl")
    MODEL = joblib.load(model_path)
    APP_ENCODER = joblib.load(enc_path)
    gui.log(f"✅ Model & encoder loaded from: {model_path}")


def build_features(keys, mouse, app, app_enc):
    total_input = keys + mouse
    ratio = keys / (mouse + 1)
    group = 2
    return {
        "Key_Rate": keys,
        "Mouse_Rate": mouse,
        "Active_App_Encoded": app_enc,
        "Total_Input": total_input,
        "Key_Mouse_Ratio": ratio,
        "Is_ActiveKeys": int(keys > 0),
        "Is_ActiveMouse": int(mouse > 0),
        "App_Group_Enc": group
    }


def log_session_row(ts, app, keys, mouse, pred, conf):
    try:
        header_needed = not os.path.exists(SESSION_LOG_PATH)
        with open(SESSION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["Timestamp", "Active_App", "Key_Rate", "Mouse_Rate", "Predicted_State", "Confidence"])
            w.writerow([ts, app, keys, mouse, "Flow" if pred else "Distracted", round(conf, 3)])
    except:
        pass


running = True


def optimizer_main(gui):
    global KEY_COUNT, MOUSE_COUNT, _CONSEC_FLOW, _CONSEC_DIST, running

    load_model_and_encoder(gui)
    gui.log("🎯 Focus Flow Optimizer started.\n")

    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    keyboard_listener.start()
    mouse_listener.start()

    while running:
        with input_lock:
            keys, mouse_c = KEY_COUNT, MOUSE_COUNT
            KEY_COUNT = 0
            MOUSE_COUNT = 0

        total = keys + mouse_c
        if total < MIN_ACTIVITY:
            gui.log(f"[{time.strftime('%H:%M:%S')}] Awaiting meaningful activity…")
            time.sleep(PREDICT_INTERVAL_SECONDS)
            continue

        app = get_active_window_info()
        try:
            app_enc = APP_ENCODER.transform([app])[0]
        except:
            app_enc = 0

        X = pd.DataFrame([build_features(keys, mouse_c, app, app_enc)])
        proba = MODEL.predict_proba(X)[0]
        flow_prob = float(proba[1])
        pred = flow_prob >= DECISION_THRESHOLD

        if pred:
            _CONSEC_FLOW += 1
            _CONSEC_DIST = 0
        else:
            _CONSEC_DIST += 1
            _CONSEC_FLOW = 0

        if _CONSEC_FLOW >= REQUIRED_CONSECUTIVE:
            banner = "✅ FLOW STATE"
        elif _CONSEC_DIST >= REQUIRED_CONSECUTIVE:
            banner = "🛑 DISTRACTED"
        else:
            banner = "..."

        gui.log(f"[{time.strftime('%H:%M:%S')}] {banner} | {app} | Keys={keys} | Mouse={mouse_c} | Flow={flow_prob:.2f}")

        log_session_row(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), app, keys, mouse_c, pred, flow_prob)
        time.sleep(PREDICT_INTERVAL_SECONDS)


if __name__ == "__main__":
    app = FocusApp()
    app.mainloop()