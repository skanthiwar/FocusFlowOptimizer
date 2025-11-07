"""
Focus Flow Optimizer — Data Monitor
-----------------------------------
Continuously captures user keyboard/mouse activity and active applications,
logging them into data/raw_log.csv for later manual labeling and model training.
"""

import os
import sys
import time
import csv
import psutil
import win32gui
import threading
from pynput import keyboard, mouse
from datetime import datetime

# =========================
# Configuration
# =========================
LOG_INTERVAL_SECONDS = 5           # Snapshot interval
OUTPUT_FILE = os.path.join("data", "raw_log.csv")
AUTOSAVE_BACKUP_INTERVAL = 60 * 5  # Save a backup every 5 minutes
PAUSE_HOTKEY = keyboard.Key.f8     # Press F8 to toggle pause

# =========================
# Globals
# =========================
KEY_COUNT = 0
MOUSE_COUNT = 0
PAUSED = False
STOP_REQUESTED = False
LOCK = threading.Lock()
LAST_BACKUP = time.time()

keyboard_listener = None
mouse_listener = None


# =========================
# Helper Functions
# =========================

def get_project_root():
    """Dynamically resolve project root (works even in packaged apps)."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_active_window_info():
    """Return the current window's title and process name."""
    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32gui.GetWindowThreadProcessId(hwnd)
        proc_name = psutil.Process(pid).name()
        return title, proc_name
    except Exception:
        return "N/A", "N/A"


def init_csv():
    """Ensure data folder and CSV headers exist."""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Key_Rate", "Mouse_Rate", "Active_App", "Window_Title", "Flow_State"])
        print(f"📁 Created new log file: {OUTPUT_FILE}")


def save_snapshot():
    """Append current activity snapshot to CSV."""
    global KEY_COUNT, MOUSE_COUNT, LAST_BACKUP
    title, app = get_active_window_info()

    with LOCK:
        key_rate = KEY_COUNT
        mouse_rate = MOUSE_COUNT
        KEY_COUNT = 0
        MOUSE_COUNT = 0

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        key_rate,
        mouse_rate,
        app,
        title,
        -1  # Default: unlabeled state
    ]

    try:
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print(f"⚠️ Error writing to log: {e}")

    # Periodic backup
    if time.time() - LAST_BACKUP > AUTOSAVE_BACKUP_INTERVAL:
        backup_file = OUTPUT_FILE.replace(".csv", "_backup.csv")
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as src, open(backup_file, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            print(f"💾 Backup saved: {backup_file}")
        except Exception:
            pass
        LAST_BACKUP = time.time()


# =========================
# Input Event Listeners
# =========================

def on_press(key):
    """Keyboard press callback."""
    global KEY_COUNT
    if key == PAUSE_HOTKEY:
        toggle_pause()
    with LOCK:
        KEY_COUNT += 1


def on_click(x, y, button, pressed):
    """Mouse click callback."""
    if pressed:
        global MOUSE_COUNT
        with LOCK:
            MOUSE_COUNT += 1


def on_scroll(x, y, dx, dy):
    """Mouse scroll callback."""
    global MOUSE_COUNT
    with LOCK:
        MOUSE_COUNT += 1


# =========================
# Control Functions
# =========================

def toggle_pause():
    """Toggle paused state via hotkey."""
    global PAUSED
    PAUSED = not PAUSED
    state = "⏸️ Paused" if PAUSED else "▶️ Resumed"
    print(f"\n{state} logging.")


def main_loop():
    """Main monitoring loop."""
    global STOP_REQUESTED

    print("\n--- FOCUS FLOW DATA MONITOR ---")
    print(f"Press {PAUSE_HOTKEY} to pause/resume logging.")
    print(f"Logging every {LOG_INTERVAL_SECONDS}s. Press Ctrl+C to stop.\n")

    init_csv()
    last_snapshot_time = time.time()

    while not STOP_REQUESTED:
        if not PAUSED:
            if time.time() - last_snapshot_time >= LOG_INTERVAL_SECONDS:
                save_snapshot()
                last_snapshot_time = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Logged activity snapshot.")
        time.sleep(0.5)

    print("\n🛑 Stopping Data Monitor... Goodbye!")


def start_listeners():
    """Initialize input listeners."""
    global keyboard_listener, mouse_listener
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    keyboard_listener.start()
    mouse_listener.start()


def stop_listeners():
    """Stop listeners safely."""
    if keyboard_listener and hasattr(keyboard_listener, "running") and keyboard_listener.running:
        keyboard_listener.stop()
    if mouse_listener and hasattr(mouse_listener, "running") and mouse_listener.running:
        mouse_listener.stop()


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    try:
        start_listeners()
        main_loop()
    except KeyboardInterrupt:
        STOP_REQUESTED = True
        print("\nKeyboardInterrupt detected.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        stop_listeners()
        print("✅ Listeners stopped cleanly.")
