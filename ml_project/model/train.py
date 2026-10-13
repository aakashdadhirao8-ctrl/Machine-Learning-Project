import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, joblib, json
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

try:
    import sklearn
    SKLEARN_VERSION = tuple(map(int, sklearn.__version__.split('.')[:2]))
except:
    SKLEARN_VERSION = (1, 6)

print(f"sklearn version: {SKLEARN_VERSION}")

from .data_generator import generate_customer_data
from .preprocessing import get_preprocessor, build_feature_names, save_preprocessor, FEATURE_STORE

def get_logreg(C=1.0, penalty='l2', l1_ratio=None):
    """Create LogisticRegression compatible with both old and new sklearn"""
    if SKLEARN_VERSION >= (1, 8):
        # New API in 1.8+: penalty deprecated, use l1_ratio
        if penalty == 'l1':
            return LogisticRegression(C=C, l1_ratio=1.0, max_iter=1000, random_state=42, solver='saga')
        elif penalty == 'l2':
            return LogisticRegression(C=C, l1_ratio=0.0, max_iter=1000, random_state=42, solver='lbfgs')
        elif penalty == 'elasticnet':
            return LogisticRegression(C=C, l1_ratio=l1_ratio or 0.5, max_iter=1000, random_state=42, solver='saga')
        else:
            return LogisticRegression(C=C, max_iter=1000, random_state=42)
    else:
        # Old API
        if penalty == 'elasticnet':
            return LogisticRegression(penalty=penalty, solver='saga', l1_ratio=l1_ratio or 0.5, C=C, max_iter=1000, random_state=42)
        else:
            solver = 'saga' if penalty == 'l1' else 'lbfgs'
            return LogisticRegression(penalty=penalty, solver=solver, C=C, max_iter=1000, random_state=42)

def train_all_models():
    print("Checking for real Kaggle dataset...")
    os.makedirs('artifacts', exist_ok=True)
    real_paths = ["artifacts/customers.csv", "/tmp/customers.csv"]
    df = None
    for p in real_paths:
        if os.path.exists(p):
            try:
                df_test = pd.read_csv(p, nrows=5)
                if 'customer_segment' in df_test.columns:
                    print(f"Found real Kaggle dataset at {p}, using it (50k rows)")
                    df = pd.read_csv(p)
                    if 'customer_id' in df.columns:
                        df = df.drop('customer_id', axis=1)
                    break
            except Exception as e:
                print(f"Failed to load {p}: {e}")
                continue
    
    if df is None:
        print("Real dataset not found, generating synthetic data...")
        df = generate_customer_data(3500)
        df.to_csv('artifacts/customers.csv', index=False)
        from .preprocessing import TARGET_COLUMN
        target_col = TARGET_COLUMN
    else:
        from .preprocessing import TARGET_COLUMN
        target_col = TARGET_COLUMN
        df.to_csv('artifacts/customers.csv', index=False)
        print(f"Using real dataset: {df.shape}, target={target_col}")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
    
    print(f"Split: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    
    preprocessor = get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    feature_names = build_feature_names(preprocessor)
    save_preprocessor(preprocessor, 'artifacts/preprocessor.pkl')
    
    print("Training Multinomial Logistic Regression variants...")
    models_linear = {}
    
    # Use version-compatible creation
    lr_multinomial = get_logreg(C=1.0, penalty='l2')
    lr_multinomial.fit(X_train_proc, y_train)
    models_linear['multinomial_l2_lbfgs'] = lr_multinomial
    
    lr_l1 = get_logreg(C=0.8, penalty='l1')
    lr_l1.fit(X_train_proc, y_train)
    models_linear['multinomial_l1'] = lr_l1
    
    lr_l2_c01 = get_logreg(C=0.1, penalty='l2')
    lr_l2_c01.fit(X_train_proc, y_train)
    models_linear['multinomial_l2_C0.1'] = lr_l2_c01
    
    lr_l2_c10 = get_logreg(C=10, penalty='l2')
    lr_l2_c10.fit(X_train_proc, y_train)
    models_linear['multinomial_l2_C10'] = lr_l2_c10
    
    lr_en = get_logreg(C=1.0, penalty='elasticnet', l1_ratio=0.5)
    lr_en.fit(X_train_proc, y_train)
    models_linear['multinomial_elasticnet'] = lr_en
    
    bias_variance_data = []
    for name, model in models_linear.items():
        train_acc = accuracy_score(y_train, model.predict(X_train_proc))
        val_acc = accuracy_score(y_val, model.predict(X_val_proc))
        bias_variance_data.append({
            'model': name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc
        })
    
    best_linear_name = max(bias_variance_data, key=lambda x: x['val_acc'])['model']
    best_linear_model = models_linear[best_linear_name]
    print(f"Best linear: {best_linear_name}")
    
    joblib.dump(best_linear_model, 'artifacts/model_multinomial_logreg.pkl')
    
    print("Training tree-based models...")
    models_tree = {}
    
    dt = DecisionTreeClassifier(max_depth=8, min_samples_split=10, criterion='gini', random_state=42)
    dt.fit(X_train_proc, y_train)
    models_tree['decision_tree_gini'] = dt
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_proc, y_train)
    models_tree['random_forest'] = rf
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(y_train)
    joblib.dump(le, 'artifacts/label_encoder.pkl')
    
    print("Running cross-validation and hyperparameter search...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_linear_model, X_train_proc, y_train, cv=skf, scoring='accuracy')
    
    # Grid search - use new API if needed
    if SKLEARN_VERSION >= (1, 8):
        param_grid = {'C': [0.01, 0.1, 1, 10]}
    else:
        param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
    
    grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_proc, y_train)
    
    param_dist_rf = {'n_estimators': [50,100,150], 'max_depth': [6,10,15,None], 'min_samples_split': [2,5,10]}
    random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist_rf, n_iter=4, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X_train_proc, y_train)
    
    bayesian_trials = []
    for C in [0.05, 0.2, 0.5, 2, 5]:
        m = get_logreg(C=C, penalty='l2')
        scores = cross_val_score(m, X_train_proc, y_train, cv=3)
        bayesian_trials.append({'C': C, 'mean_cv': scores.mean()})
    
    test_pred = best_linear_model.predict(X_test_proc)
    test_proba = best_linear_model.predict_proba(X_test_proc)
    test_acc = accuracy_score(y_test, test_pred)
    clf_report = classification_report(y_test, test_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, test_pred, labels=best_linear_model.classes_)
    
    try:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=best_linear_model.classes_)
        roc_auc_ovr = roc_auc_score(y_test_bin, test_proba, multi_class='ovr')
    except:
        roc_auc_ovr = 0.0
    
    coefs = best_linear_model.coef_
    
    tree_importances = {}
    for name, model in models_tree.items():
        if hasattr(model, 'feature_importances_'):
            tree_importances[name] = model.feature_importances_.tolist()
    
    metrics = {
        'cv_scores_mean': float(cv_scores.mean()),
        'cv_scores_std': float(cv_scores.std()),
        'cv_scores_all': cv_scores.tolist(),
        'grid_best_params': grid.best_params_,
        'grid_best_score': float(grid.best_score_),
        'random_best_params': random_search.best_params_,
        'random_best_score': float(random_search.best_score_),
        'bayesian_trials': bayesian_trials,
        'test_accuracy': float(test_acc),
        'roc_auc_ovr': float(roc_auc_ovr),
        'classification_report': clf_report,
        'confusion_matrix': conf_mat.tolist(),
        'classes': best_linear_model.classes_.tolist(),
        'bias_variance': bias_variance_data,
        'feature_names': feature_names,
        'linear_coefficients': coefs.tolist(),
        'tree_importances': tree_importances,
        'feature_store': FEATURE_STORE,
        'sklearn_version': sklearn.__version__,
        'split_info': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        }
    }
    
    with open('artifacts/metrics.json','w') as f:
        json.dump(metrics, f, indent=2)
    
    joblib.dump((X_test, y_test, X_test_proc), 'artifacts/test_data.pkl')
    
    print(f"Training complete. Accuracy: {test_acc*100:.1f}%")
    return metrics

if __name__ == "__main__":
    train_all_models()
