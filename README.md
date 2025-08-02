# 🧠 Sign Language Detector (Flask + ML)

A real-time sign language detection web app built with Python, Flask, OpenCV, and Mediapipe.  
This project uses a trained `.pkl` model to classify sign language gestures and display predictions live from a webcam.

---

## 🔧 Technologies & Libraries Used

| Tool / Library    | Purpose                         |
|------------------|---------------------------------|
| `Flask`          | Backend web framework           |
| `OpenCV`         | Capturing webcam frames         |
| `Mediapipe`      | Hand tracking and landmark detection |
| `joblib`         | For loading the trained ML model (`.pkl` file) |
| `sklearn`, `numpy`, `pandas` | Data processing & model training |
| Python Version   | **Python 11 only** (due to compatibility issues) |

---

## 🛠️ Model Info

- Model was trained using scikit-learn
- Saved using **joblib** (which is 🔥 for saving & loading models fast)
- The `.pkl` file used contains gesture classification logic

---

## 🚨 Known Issues

- ❌ **Not Deployable (yet)** — Tried every free hosting platform:
  - Render: failed due to large model + lib size
  - Railway: same issue
  - Vercel: doesn’t support backend-heavy ML projects like this

If **you know a free or affordable platform** where heavy Python-based Flask + ML apps can be hosted (with support for big `.pkl` files), **please open an issue or reach out!**

---

## 💡 Future Plans

- Build a lighter version that’s deployable
- Try converting model to TensorFlow Lite or ONNX
- Add frontend improvements with better webcam UI
- Maybe even mobile-friendly version in future?

---

## 👑 About Me

> I'm Prince. I built this while learning Python, Flask, and ML together.  
> This is a work-in-progress project, but everything works **locally** with Python 11.  
> Still learning — still failing — still pushing forward 🚀

---

## 📦 How to Run Locally

```bash
git clone https://github.com/codedbyyprince/SIGN-DETECTOR--.git
cd SIGN-DETECTOR--
pip install -r requirements.txt
python app.py
