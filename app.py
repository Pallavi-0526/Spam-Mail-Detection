from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model              = joblib.load(os.path.join(BASE_DIR, 'model', 'spam_model.pkl'))
feature_extraction = joblib.load(os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl'))

SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'cash', 'prize', 'claim', 'urgent',
    'click', 'offer', 'limited', 'congratulations', 'selected',
    'call now', 'txt', 'mobile', 'discount', 'buy now'
]

def get_spam_signals(text):
    text_lower = text.lower()
    found = [kw for kw in SPAM_KEYWORDS if kw in text_lower]
    return found[:5]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Please enter a message.'}), 400

    features   = feature_extraction.transform([message])
    prediction = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]

    is_spam    = int(prediction) == 0
    confidence = float(proba[0]) if is_spam else float(proba[1])

    signals = get_spam_signals(message) if is_spam else []

    return jsonify({
        'result':      'SPAM' if is_spam else 'HAM',
        'is_spam':     is_spam,
        'confidence':  round(confidence * 100, 1),
        'signals':     signals,
        'message_len': len(message.split())
    })

@app.route('/examples')
def examples():
    examples_list = [
        {
            'label': 'Spam example',
            'text':  "CONGRATULATIONS! You've WON a £1000 Tesco gift card. CALL NOW: 08081 565656 to CLAIM your prize!"
        },
        {
            'label': 'Ham example',
            'text':  "Hey, are we still on for lunch tomorrow? Let me know what time works for you."
        },
        {
            'label': 'Spam example 2',
            'text':  "FREE ringtone! Txt MUSIC to 87121 to get FREE music on ur mobile every week! 3 POUNDS per week"
        },
        {
            'label': 'Ham example 2',
            'text':  "I've been searching for the right words to thank you for this breather. You have been wonderful and a blessing at all times."
        }
    ]
    return jsonify(examples_list)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
