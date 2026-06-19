"""
RETRAIN SCRIPT — run this once inside your activated venv to fix the
scikit-learn version mismatch (model was pickled with 1.8.0, your venv has 1.5.1).

Usage:
    (venv) > python retrain_model.py

This regenerates model/spam_model.pkl and model/tfidf_vectorizer.pkl
using YOUR installed scikit-learn version, so app.py will work correctly.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sklearn

print(f"scikit-learn version in this environment: {sklearn.__version__}")
print("Retraining model with this version...\n")

# Load data
raw_mail_data = pd.read_csv('data/mail_data.csv')
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Label encoding (spam=0, ham=1)
mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})

X = mail_data['Message']
Y = mail_data['Category'].astype('int')

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_features, Y_train)

# Evaluate
train_acc = accuracy_score(Y_train, model.predict(X_train_features))
test_acc = accuracy_score(Y_test, model.predict(X_test_features))
print(f"Training Accuracy : {train_acc:.4f}")
print(f"Test Accuracy     : {test_acc:.4f}")
print()
print(classification_report(Y_test, model.predict(X_test_features), target_names=['Spam', 'Ham']))

# Save — overwrites the old pickle files with ones matching YOUR sklearn version
joblib.dump(model, 'model/spam_model.pkl')
joblib.dump(feature_extraction, 'model/tfidf_vectorizer.pkl')

print(f"\n✅ Model retrained and saved using scikit-learn {sklearn.__version__}")
print("✅ You can now run: python app.py")
