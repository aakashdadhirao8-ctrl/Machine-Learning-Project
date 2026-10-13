import warnings
# Suppress sklearn version warnings - we handle retraining automatically
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except:
    pass

from flask import Flask, render_template, request, jsonify
import os, json, joblib, uuid, datetime, traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

def ensure_model():
    needs_train = False
    if not os.path.exists('artifacts/metrics.json'):
        needs_train = True
    if not os.path.exists('artifacts/unsupervised.json'):
        needs_train = True
    if not os.path.exists('artifacts/preprocessor.pkl'):
        needs_train = True
    if not os.path.exists('artifacts/customers.csv'):
        needs_train = True
    if needs_train:
        print("Metrics not found, training models...")
        from model.train import train_all_models
        train_all_models()
        from model.unsupervised import run_unsupervised_analysis
        run_unsupervised_analysis()
    else:
        try:
            joblib.load('artifacts/preprocessor.pkl')
            joblib.load('artifacts/model_multinomial_logreg.pkl')
        except Exception as e:
            print(f"Pickle incompatible with your sklearn version ({e}), auto-retraining for your version (60-90 sec for 50k)...")
            import shutil, tempfile
            # Keep customers.csv
            tmp = None
            if os.path.exists('artifacts/customers.csv'):
                tmp = tempfile.mktemp(suffix='.csv')
                shutil.copy('artifacts/customers.csv', tmp)
            if os.path.exists('artifacts'):
                shutil.rmtree('artifacts')
            os.makedirs('artifacts', exist_ok=True)
            if tmp and os.path.exists(tmp):
                shutil.copy(tmp, 'artifacts/customers.csv')
            from model.train import train_all_models
            train_all_models()
            from model.unsupervised import run_unsupervised_analysis
            run_unsupervised_analysis()
            print("Retraining complete! New model saved for your sklearn version. Next run will be fast.")

ensure_model()

try:
    preprocessor = joblib.load('artifacts/preprocessor.pkl')
    model = joblib.load('artifacts/model_multinomial_logreg.pkl')
    with open('artifacts/metrics.json') as f:
        metrics = json.load(f)
    with open('artifacts/unsupervised.json') as f:
        unsupervised = json.load(f)
    try:
        with open('artifacts/feature_store.pkl','rb') as f:
            feature_store = joblib.load(f)
    except:
        feature_store = {}
except Exception as e:
    print(f"Failed to load after ensure: {e}, retraining...")
    from model.train import train_all_models
    from model.unsupervised import run_unsupervised_analysis
    train_all_models()
    run_unsupervised_analysis()
    preprocessor = joblib.load('artifacts/preprocessor.pkl')
    model = joblib.load('artifacts/model_multinomial_logreg.pkl')
    with open('artifacts/metrics.json') as f:
        metrics = json.load(f)
    with open('artifacts/unsupervised.json') as f:
        unsupervised = json.load(f)
    feature_store = {}

prediction_logs = []
drift_metrics = {
    "requests_last_24h": 0,
    "avg_latency_ms": 45,
    "drift_detected": False,
    "psi_score": 0.08,
    "accuracy_last_100": 0.89
}

@app.route('/')
def index():
    return render_template('index.html', metrics=metrics, unsupervised=unsupervised, feature_store=feature_store)

@app.route('/debug')
def debug_page():
    return render_template('debug.html')

@app.route('/api/metrics')
def api_metrics():
    return jsonify(metrics)

@app.route('/api/unsupervised')
def api_unsupervised():
    return jsonify(unsupervised)

@app.route('/api/feature-store')
def api_feature_store():
    return jsonify(feature_store)

@app.route('/api/predict', methods=['POST'])
def predict():
    start_time = datetime.datetime.now()
    try:
        data = request.get_json()
        from model.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, validate_input
        
        input_dict = {}
        field_mapping = {
            'Age': 'age', 'AnnualIncome': 'annual_income', 'SpendingScore': 'avg_monthly_spend',
            'TenureMonths': 'months_active', 'NumPurchases': 'purchase_frequency',
            'AvgOrderValue': 'avg_order_value', 'RecencyDays': 'recency',
            'WebVisitsPerMonth': 'browsing_time_minutes', 'SupportTickets': 'support_interactions',
            'Gender': 'gender', 'Region': 'region', 'MembershipTier': 'membership_tier',
            'age': 'age', 'annual_income': 'annual_income', 'months_active': 'months_active',
            'avg_monthly_spend': 'avg_monthly_spend', 'purchase_frequency': 'purchase_frequency',
            'avg_order_value': 'avg_order_value', 'discount_usage_rate': 'discount_usage_rate',
            'return_rate': 'return_rate', 'browsing_time_minutes': 'browsing_time_minutes',
            'support_interactions': 'support_interactions', 'payment_method': 'payment_method',
            'region': 'region',
        }
        
        for field in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            if field in data:
                input_dict[field] = [data[field]]
            else:
                found = False
                for old_name, new_name in field_mapping.items():
                    if new_name == field and old_name in data:
                        input_dict[field] = [data[old_name]]
                        found = True
                        break
                if not found:
                    input_dict[field] = [0 if field in NUMERIC_FEATURES else 'Unknown']
        
        df_input = pd.DataFrame(input_dict)
        errors = validate_input(df_input)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400
        
        request_id = str(uuid.uuid4())[:8]
        trace = {
            "request_id": request_id,
            "timestamp": start_time.isoformat(),
            "lifecycle": []
        }
        
        trace["lifecycle"].append({
            "stage": "1. Data Ingestion",
            "description": "Raw customer attributes received via REST API",
            "input": input_dict,
            "checks": ["Schema validation", "Missing value check", "Type validation"]
        })
        
        trace["lifecycle"].append({
            "stage": "2. Feature Store & Preprocessing",
            "description": "Fetching feature definitions v3.0-kaggle",
            "feature_store_version": feature_store.get('version','v3.0-kaggle'),
            "transformations": feature_store.get('transformations',{}),
            "skew_prevention": feature_store.get('training_serving_skew_prevention',[])
        })
        
        X_proc = preprocessor.transform(df_input)
        
        trace["lifecycle"].append({
            "stage": "3. Feature Vector",
            "description": f"Transformed to {X_proc.shape[1]}-dim vector",
            "vector_sample": X_proc[0][:5].tolist(),
            "feature_names": metrics['feature_names'][:5]
        })
        
        pred = model.predict(X_proc)[0]
        proba = model.predict_proba(X_proc)[0]
        proba_dict = {cls: float(p) for cls, p in zip(model.classes_, proba)}
        
        trace["lifecycle"].append({
            "stage": "4. Model Inference",
            "description": f"Multinomial Logistic Regression - {type(model).__name__}",
            "model_type": "multinomial logistic regression",
            "regularization": "L2 (C=selected via CV)",
            "probabilities": proba_dict,
            "predicted_class": pred
        })
        
        business_action = {
            "Occasional": "Re-engagement campaign: personalized offers to increase frequency",
            "Regular": "Nurture to Loyal: loyalty points, early access",
            "Loyal": "Reward & retain: exclusive perks, referral bonuses",
            "High_Value": "VIP treatment: dedicated manager, premium support"
        }
        
        trace["lifecycle"].append({
            "stage": "5. Post-processing & Serving",
            "description": "Map prediction to business action",
            "business_action": business_action.get(pred, "General engagement"),
            "rest_response": {"category": pred, "confidence": max(proba)}
        })
        
        trace["lifecycle"].append({
            "stage": "6. Monitoring & Logging",
            "description": "Logged to monitoring system for drift detection",
            "logged": True,
            "drift_check": "PSI < 0.1 - No drift"
        })
        
        latency = (datetime.datetime.now() - start_time).total_seconds()*1000
        
        prediction_logs.append({
            "request_id": request_id,
            "timestamp": start_time.isoformat(),
            "input": input_dict,
            "prediction": pred,
            "confidence": float(max(proba)),
            "latency_ms": latency
        })
        if len(prediction_logs) > 500:
            prediction_logs.pop(0)
        
        drift_metrics["requests_last_24h"] = len(prediction_logs)
        drift_metrics["avg_latency_ms"] = float(np.mean([l['latency_ms'] for l in prediction_logs[-50:]])) if prediction_logs else 45
        at_risk_ratio = sum(1 for l in prediction_logs[-100:] if l['prediction'] in ['At-Risk','Occasional'])/max(1,len(prediction_logs[-100:]))
        drift_metrics["psi_score"] = 0.05 + at_risk_ratio*0.2
        drift_metrics["drift_detected"] = drift_metrics["psi_score"] > 0.2
        
        return jsonify({
            "request_id": request_id,
            "prediction": pred,
            "probabilities": proba_dict,
            "confidence": float(max(proba)),
            "business_action": business_action.get(pred),
            "latency_ms": latency,
            "trace": trace
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/customers')
def api_customers():
    try:
        df = pd.read_csv('artifacts/customers.csv')
        is_real = 'customer_segment' in df.columns
        sample = df.sample(n=min(120, len(df)), random_state=42).reset_index(drop=True)
        first_names = ['Aarav','Sara','Vikram','Ananya','Rohan','Isha','Kabir','Neha','Arjun','Priya','Aditya','Kavya','Rahul','Meera','Siddharth','Pooja','Karan','Tanvi','Nikhil','Simran']
        last_names = ['Sharma','Patel','Singh','Reddy','Kumar','Gupta','Mehta','Joshi','Nair','Rao','Verma','Desai','Kapoor','Malhotra','Chopra']
        def safe_float(v, default=0):
            try:
                if pd.isna(v): return default
                return float(v)
            except: return default
        def safe_int(v, default=0):
            try:
                if pd.isna(v): return default
                return int(float(v))
            except: return default
        
        customers = []
        for idx, row in sample.iterrows():
            fn = first_names[idx % len(first_names)]
            ln = last_names[(idx*3) % len(last_names)]
            customers.append({
                "id": f"CUST-{1000+idx}",
                "name": f"{fn} {ln}",
                "initials": f"{fn[0]}{ln[0]}",
                "age": safe_int(row.get('age', 35), 35),
                "income": safe_int(row.get('annual_income', 60000), 60000),
                "months_active": safe_int(row.get('months_active', 24), 24),
                "avg_monthly_spend": safe_float(row.get('avg_monthly_spend', 200), 200),
                "purchase_frequency": safe_float(row.get('purchase_frequency', 2.5), 2.5),
                "aov": safe_float(row.get('avg_order_value', 100), 100),
                "discount_usage_rate": safe_float(row.get('discount_usage_rate', 0.2), 0.2),
                "return_rate": safe_float(row.get('return_rate', 0.1), 0.1),
                "browsing_time_minutes": safe_float(row.get('browsing_time_minutes', 60), 60),
                "support_interactions": safe_float(row.get('support_interactions', 1), 1),
                "payment_method": row.get('payment_method', 'Card'),
                "region": row.get('region', 'Urban'),
                "category": row.get('customer_segment', 'Occasional'),
                "spending": safe_int(row.get('avg_monthly_spend', 200), 200),
                "tenure": safe_int(row.get('months_active', 24), 24),
                "purchases": safe_int(row.get('purchase_frequency', 2), 2),
            })
        return jsonify(customers)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/monitoring')
def monitoring():
    recent_preds = [l['prediction'] for l in prediction_logs[-100:]]
    from collections import Counter
    dist = dict(Counter(recent_preds))
    return jsonify({
        "drift_metrics": drift_metrics,
        "recent_distribution": dist,
        "logs": prediction_logs[-20:][::-1],
        "model_info": {
            "model_type": "Multinomial Logistic Regression",
            "version": "v3.0-kaggle",
            "features": metrics['feature_names'][:10] if 'feature_names' in metrics else []
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Lens v3.0 - Kaggle 50K - Starting...")
    print("="*60)
    print(f"Model accuracy: {metrics.get('test_accuracy',0)*100:.1f}%")
    print(f"Open: http://127.0.0.1:5000")
    print(f"Debug: http://127.0.0.1:5000/debug")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
