# ============================================================
#  SPAM MAIL DETECTION — Complete ML Notebook
#  Internpe AI/ML Internship Project
#  Author: Pallavi
# ============================================================

# %% ── Imports ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully")

# %% [markdown]
# ## Step 1 — Data Collection & Pre-Processing

# %% ── Load Data ───────────────────────────────────────────
raw_mail_data = pd.read_csv('data/mail_data.csv')
print("Dataset Shape:", raw_mail_data.shape)
print()
print(raw_mail_data.head())

# %% ── Handle null values ─────────────────────────────────
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
print("\nNull values after handling:")
print(mail_data.isnull().sum())

# %% ── Basic info ─────────────────────────────────────────
print("\nDataset Info:")
print(f"Total messages : {len(mail_data)}")
print(f"Spam messages  : {(mail_data['Category'] == 'spam').sum()}")
print(f"Ham messages   : {(mail_data['Category'] == 'ham').sum()}")

# %% [markdown]
# ## Step 2 — Exploratory Data Analysis (EDA)

# %% ── Class distribution plot ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Spam Mail Detection — EDA', fontsize=14, fontweight='bold')

# Pie chart
counts = mail_data['Category'].value_counts()
colors = ['#ff4757', '#2ed573']
axes[0].pie(counts, labels=['Ham', 'Spam'], colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Class Distribution')

# Bar chart
axes[1].bar(['Ham', 'Spam'], counts, color=colors, edgecolor='white', linewidth=0.5)
axes[1].set_title('Message Count by Class')
axes[1].set_ylabel('Count')
for i, v in enumerate(counts):
    axes[1].text(i, v + 30, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('notebook/eda_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA plots saved")

# %% ── Message length analysis ────────────────────────────
mail_data['msg_length'] = mail_data['Message'].apply(len)
mail_data['word_count'] = mail_data['Message'].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for label, color in [('spam', '#ff4757'), ('ham', '#2ed573')]:
    subset = mail_data[mail_data['Category'] == label]['msg_length']
    axes[0].hist(subset, bins=40, alpha=0.6, color=color, label=label.upper())
axes[0].set_title('Message Length Distribution')
axes[0].set_xlabel('Character count')
axes[0].legend()

avg_len = mail_data.groupby('Category')['word_count'].mean()
axes[1].bar(avg_len.index, avg_len.values, color=['#ff4757', '#2ed573'])
axes[1].set_title('Average Word Count by Category')
axes[1].set_ylabel('Avg word count')

plt.tight_layout()
plt.savefig('notebook/eda_lengths.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 3 — Label Encoding
# - spam → 0
# - ham  → 1

# %%
# Label encoding (matching Internpe convention exactly)
mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})
print("Label encoding done:")
print(mail_data['Category'].value_counts())

X = mail_data['Message']
Y = mail_data['Category']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Labels   (Y) shape: {Y.shape}")

# %% [markdown]
# ## Step 4 — Train / Test Split

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3
)

print(f"Total samples  : {X.shape[0]}")
print(f"Training       : {X_train.shape[0]}")
print(f"Testing        : {X_test.shape[0]}")

Y_train = Y_train.astype('int')
Y_test  = Y_test.astype('int')

# %% [markdown]
# ## Step 5 — Feature Extraction (TF-IDF)

# %%
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features  = feature_extraction.transform(X_test)

print(f"Vocabulary size          : {len(feature_extraction.vocabulary_)}")
print(f"Training feature matrix  : {X_train_features.shape}")
print(f"Test feature matrix      : {X_test_features.shape}")

# %% [markdown]
# ## Step 6 — Model Training (Logistic Regression)

# %%
model = LogisticRegression(max_iter=1000)
model.fit(X_train_features, Y_train)
print("✅ Logistic Regression model trained!")

# %% [markdown]
# ## Step 7 — Evaluation

# %%
# Training accuracy
pred_train = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, pred_train)
print(f"Accuracy on training data : {accuracy_on_training_data:.4f}")

# Test accuracy
pred_test = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, pred_test)
print(f"Accuracy on test data     : {accuracy_on_test_data:.4f}")

print("\nClassification Report:")
print(classification_report(Y_test, pred_test, target_names=['Spam', 'Ham']))

# %% ── Confusion Matrix ────────────────────────────────────
cm = confusion_matrix(Y_test, pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Spam', 'Ham'],
            yticklabels=['Spam', 'Ham'])
plt.title('Confusion Matrix — Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('notebook/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved")

# %% [markdown]
# ## Step 8 — Building a Predictive System

# %%
def predict_mail(input_mail):
    """Predict whether an email/SMS is spam or ham."""
    input_data_features = feature_extraction.transform([input_mail])
    prediction = model.predict(input_data_features)
    proba      = model.predict_proba(input_data_features)[0]

    if prediction[0] == 1:
        result = 'Ham mail ✅'
        conf   = proba[1]
    else:
        result = 'Spam mail 🚨'
        conf   = proba[0]

    print(f"Message  : {input_mail[:80]}...")
    print(f"Result   : {result}")
    print(f"Confidence: {conf*100:.1f}%")
    print()

# Test with Internpe's provided example
predict_mail("I've been searching for the right words to thank you for this breather. "
             "I promise i wont take your help for granted and will fulfil my promise. "
             "You have been wonderful and a blessing at all times")

# Test with a spam example
predict_mail("CONGRATULATIONS! You've WON a £1000 Tesco gift card. "
             "CALL NOW: 08081 565656 to CLAIM your prize!")

# %% [markdown]
# ## Step 9 — Save Model

# %%
joblib.dump(model,              'model/spam_model.pkl')
joblib.dump(feature_extraction, 'model/tfidf_vectorizer.pkl')
print("✅ Model saved to model/spam_model.pkl")
print("✅ Vectorizer saved to model/tfidf_vectorizer.pkl")

print("\n" + "="*50)
print("  PROJECT COMPLETE — Internpe AI/ML Internship")
print(f"  Final Test Accuracy: {accuracy_on_test_data*100:.2f}%")
print("="*50)
