# 📧 Spam Mail Detection — AI/ML Project

> **Internpe AI/ML Internship | Project 2**
> Built by **Gnanendra** · [@Internpe](https://linkedin.com/company/internpe)

---

## 🎯 Project Overview

A machine learning web application that classifies email and SMS messages as **SPAM** or **HAM (safe)** using Logistic Regression and TF-IDF vectorization — achieving **96.2% test accuracy**.

---

## ✨ Features

- 🔍 Real-time spam prediction with confidence score
- 📊 Visual confidence bar and spam signal highlights
- 💡 Built-in example messages to test instantly
- 🚀 Deployed on Render.com — no setup needed

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| ML Model | Logistic Regression (scikit-learn) |
| Feature Extraction | TF-IDF Vectorizer |
| Web Framework | Flask |
| Dataset | SMS Spam Collection (5,572 messages) |
| Deployment | Render.com |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 96.61% |
| Test Accuracy | **96.23%** |
| Precision (Spam) | 99% |
| Recall (Ham) | 100% |

---

## 🗂️ Project Structure

```
spam_detector/
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment config
├── README.md
├── data/
│   └── mail_data.csv         # Dataset (5,572 messages)
├── model/
│   ├── spam_model.pkl        # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl  # Fitted TF-IDF vectorizer
├── notebook/
│   └── spam_detection_notebook.py  # Full EDA + training notebook
└── templates/
    └── index.html            # Web UI
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/spam-mail-detection.git
cd spam-mail-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py

# 5. Open browser → http://localhost:5000
```

---

## 📱 How It Works

```
User Input → TF-IDF Transform → Logistic Regression → SPAM / HAM + Confidence %
```

1. **Input**: User enters an email or SMS message
2. **Preprocessing**: Text cleaned, lowercased, stop words removed
3. **Feature Extraction**: TF-IDF converts text to numerical vectors
4. **Prediction**: Logistic Regression classifies the vector
5. **Output**: SPAM 🚨 or HAM ✅ with confidence percentage

---

## 📸 Screenshots

> Add screenshots here after deployment

---

## 🙏 Acknowledgements

- **Internpe** for providing the internship opportunity and reference code
- UCI ML Repository for the SMS Spam Collection dataset

---

## 📬 Connect

**Gnanendra** — [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)

*Part of Internpe AI/ML + Data Science Internship*
