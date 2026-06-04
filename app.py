"""
Gaming Academic Performance — Flask API
=======================================
Run:  python app.py
URL:  http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load Models & Scalers ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

clf_model  = joblib.load(os.path.join(BASE, 'best_clf_model.joblib'))
reg_model  = joblib.load(os.path.join(BASE, 'best_reg_model.joblib'))
scaler_clf = joblib.load(os.path.join(BASE, 'scaler_clf.joblib'))
scaler_reg = joblib.load(os.path.join(BASE, 'scaler_reg.joblib'))

# ── Encoding Maps (must match training) ─────────────────────────────────────
GENDER_MAP       = {'Male': 0, 'Female': 1, 'Other': 2}
GENRE_MAP        = {'Casual': 0, 'FPS': 1, 'RPG': 2}
STRESS_MAP       = {'High': 0, 'Low': 1, 'Medium': 2}
GRADE_LABEL_MAP  = {0: 'F', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}

MODELS_NEEDING_SCALE_CLF = [
    'LogisticRegression', 'KNeighborsClassifier',
    'SVC', 'GaussianNB', 'MLPClassifier'
]
MODELS_NEEDING_SCALE_REG = [
    'LinearRegression', 'Ridge', 'SVR', 'MLPRegressor'
]

def model_class_name(model):
    return type(model).__name__

def prepare_features(data: dict):
    """
    Convert raw form data → numpy array ready for model.
    Feature order must match training:
    [age, gender, gaming_hours, study_hours, sleep_hours,
     attendance, gaming_genre, social_activity, device_usage,
     reaction_time_ms, addiction_score, stress_level]
    """
    features = np.array([[
        float(data['age']),
        GENDER_MAP.get(data['gender'], 0),
        float(data['gaming_hours']),
        float(data['study_hours']),
        float(data['sleep_hours']),
        float(data['attendance']),
        GENRE_MAP.get(data['gaming_genre'], 0),
        float(data['social_activity']),
        float(data['device_usage']),
        float(data['reaction_time_ms']),
        float(data['addiction_score']),
        STRESS_MAP.get(data['stress_level'], 1),
    ]])
    return features


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON or form data.
    Returns both classification (grade letter) and regression (grade score).
    """
    try:
        # Accept both JSON and form POST
        data = request.get_json(silent=True) or request.form.to_dict()

        if not data:
            return jsonify({'error': 'No input data received'}), 400

        features = prepare_features(data)

        # ── Classification prediction ──
        if model_class_name(clf_model) in MODELS_NEEDING_SCALE_CLF:
            clf_input = scaler_clf.transform(features)
        else:
            clf_input = features

        grade_code  = clf_model.predict(clf_input)[0]
        grade_label = GRADE_LABEL_MAP.get(int(grade_code), str(grade_code))

        # Confidence (probability if available)
        if hasattr(clf_model, 'predict_proba'):
            proba      = clf_model.predict_proba(clf_input)[0]
            confidence = round(float(max(proba)) * 100, 2)
        else:
            confidence = None

        # ── Regression prediction ──
        if model_class_name(reg_model) in MODELS_NEEDING_SCALE_REG:
            reg_input = scaler_reg.transform(features)
        else:
            reg_input = features

        grade_score = round(float(reg_model.predict(reg_input)[0]), 2)
        # Cap score to 0–100
        grade_score = max(0, min(100, grade_score))

        # ── Performance label based on score ──
        if grade_score >= 80:
            performance = 'Excellent'
            performance_color = 'success'
        elif grade_score >= 70:
            performance = 'Good'
            performance_color = 'primary'
        elif grade_score >= 60:
            performance = 'Average'
            performance_color = 'warning'
        elif grade_score >= 50:
            performance = 'Below Average'
            performance_color = 'warning'
        else:
            performance = 'Poor'
            performance_color = 'danger'

        return jsonify({
            'success': True,
            'grade_category':    grade_label,
            'grade_score':       grade_score,
            'confidence':        confidence,
            'performance':       performance,
            'performance_color': performance_color,
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'clf_model': model_class_name(clf_model),
        'reg_model': model_class_name(reg_model),
    })


if __name__ == '__main__':
    print("=" * 55)
    print("  Gaming Academic Performance — ML API")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=8080)
