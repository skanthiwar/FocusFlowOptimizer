# 🧠 Focus Flow Optimizer

An **AI-powered productivity assistant** that detects when you're in a *Flow* or *Distracted* state —  
by analyzing your keyboard and mouse activity patterns and tracking your active applications in real time.

This intelligent desktop tool helps users enhance focus, minimize distractions, and maintain deep work sessions automatically.

---

## 🚀 Project Overview

Most productivity tools remind you to focus —  
**Focus Flow Optimizer actually *detects* whether you are focused or distracted** in real time using behavioral signals.  

It leverages a **machine learning model** trained on your activity patterns to understand your working behavior.  
When it detects distraction, it can **automatically mute system audio** to reduce interruptions — and **restore it** when you’re back in flow.

---

## 🧩 Features

✅ **Real-time focus detection** using trained Random Forest model  
✅ **Behavioral analysis** — monitors typing/mouse activity & app usage  
✅ **Auto Mute/Unmute** system to boost focus  
✅ **Interactive GUI (Tkinter)** showing live logs  
✅ **CSV-based logging** of every focus event  
✅ **Streamlit Dashboard** for visual data insights  
✅ **One-click desktop EXE build** using PyInstaller  

---

## ⚙️ Tech Stack

**Python · Scikit-learn · Pandas · Psutil · Pynput · Tkinter · Matplotlib · Streamlit · PyInstaller**

---

## 📂 Folder Structure

    FocusFlowOptimizer/
    ├── src/
    │   ├── model_trainer.py           # Trains ML model on labeled data
    │   ├── flow_optimizer.py          # Core real-time predictor (CLI)
    │   ├── flow_opt_2.py              # GUI version with live logging
    ├── models/
    │   ├── focus_model_latest.pkl     # Trained ML model
    │   ├── app_encoder_latest.pkl     # Label encoder for app names
    ├── reports/
    │   ├── metrics.json               # Model thresholds & feature info
    │   ├── feature_importances.csv
    ├── dashboard/
    │   ├── focus_dashboard.py         # Streamlit visualization dashboard
    ├── data/
    │   ├── labeled_data.csv           # Training data
    │   ├── session_log.csv            # Real-time predictions log
    ├── assets/
    │   ├── brain.ico                  # App icon
    │   ├── preview.png                # Screenshot preview
    └── dist/
        ├── FocusFlowOptimizer.exe     # Built executable desktop app

---

## 🧠 How It Works

1. Tracks **keyboard & mouse activity** continuously.  
2. Detects which **application** is currently active (VS Code, Chrome, etc.).  
3. Computes engineered features like:
   - Total input rate  
   - Key/mouse ratio  
   - Active app group (Dev/Comm/Other)  
4. Predicts *Flow* or *Distracted* state using the trained ML model.  
5. Automatically **mutes/unmutes audio** based on your state.  
6. Logs every event into `data/session_log.csv`.  
7. Visualize focus data trends using the Streamlit dashboard.

---

## 💻 Run the Application

### 1️⃣ Clone the Repository

    git clone https://github.com/<your-username>/FocusFlowOptimizer.git
    cd FocusFlowOptimizer

### 2️⃣ Install Dependencies

    pip install pandas scikit-learn joblib pynput psutil pywin32 win10toast matplotlib streamlit

### 3️⃣ (Optional) Train the Model

    python src\model_trainer.py

This generates:
- `models/focus_model_latest.pkl`
- `models/app_encoder_latest.pkl`
- `reports/metrics.json`

---

## ▶️ Run the Focus Optimizer (Console Mode)

    python src\flow_optimizer.py

**Example Output:**

    --- Focus Flow Optimizer LIVE ---
    Predict every 5s | System mute: False

    [18:45:32] 🟢 FLOW STATE
       App: code.exe | Keys: 52 | Mouse: 2 | Flow prob: 91.2% | Thr: 0.50
       Stay focused!
    -------------------------------------------------------

---

## 🪟 Run the GUI Version

    python src\flow_opt_2.py

The GUI version:
- Displays focus predictions live  
- Shows mute/unmute actions  
- Logs all predictions to `data/session_log.csv`

---

## 📊 Visualize Focus Data (Dashboard)

    streamlit run dashboard/focus_dashboard.py

**Includes:**
- Focus probability over time  
- Keyboard & mouse activity graphs  
- Most-used apps during sessions  
- App-based filtering & insights  

---

## 🖥️ Build a Desktop App (EXE)

    pyinstaller --onefile --windowed --icon=assets/brain.ico --name "FocusFlowOptimizer" src/flow_opt_2.py

After build completion:

    dist/FocusFlowOptimizer.exe

Just double-click it to start — no need for Python installed!

---

## 🧾 Output Files

| File | Description |
|------|--------------|
| `data/session_log.csv` | Records every focus event |
| `reports/metrics.json` | Stores model thresholds and feature info |
| `models/*.pkl` | Trained ML model & label encoder |
| `dist/FocusFlowOptimizer.exe` | Built Windows executable |

---

## 📸 Preview

![Focus Flow Optimizer GUI](assets/preview.png)

---

## 🧩 Example Use Case

When working in VS Code and you switch briefly to Chrome —  
the app detects the drop in focus, reduces your flow score, and **mutes system audio**.  
Once you return to deep work, it **unmutes audio** automatically.  

Your session is logged for later analysis and visualization.

---

## 🌟 Future Enhancements

- 🧠 Add facial/emotional context detection  
- ☁️ Cloud syncing & team productivity reports  
- 🔔 Smart motivational reminders  
- 📈 Integrations with Notion, Jira, or Slack  

---

## ❤️ Credits

Developed by **Swapnil**  
Built with **Python**, passion, and purpose — to help people stay focused, productive, and in their *flow state*.

---

## 📬 Connect & Collaborate

💡 Want to collaborate or contribute?  
- Fork the repo & submit a pull request  
- Connect on **[LinkedIn](www.linkedin.com/in/swapnil-kanthiwar-648906176)**  

> “The best way to stay in flow is to remove what breaks it — and automate the rest.” 🧠

---

