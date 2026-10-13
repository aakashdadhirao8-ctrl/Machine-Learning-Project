import pandas as pd
import numpy as np
import joblib, json, os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

def run_unsupervised_analysis():
    df = pd.read_csv('artifacts/customers.csv')
    if 'customer_segment' in df.columns:
        target_col = 'customer_segment'
    elif 'CustomerCategory' in df.columns:
        target_col = 'CustomerCategory'
    else:
        target_col = df.columns[-1]
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    if 'customer_id' in X.columns:
        X = X.drop('customer_id', axis=1)
    
    if len(X) > 5000:
        print(f"Large dataset {len(X)} rows, sampling 3000 for unsupervised...")
        sample_idx_full = np.random.choice(len(X), 3000, replace=False)
        X_sample_full = X.iloc[sample_idx_full]
        y_sample_full = y.iloc[sample_idx_full]
    else:
        X_sample_full = X
        y_sample_full = y
    
    preprocessor = joblib.load('artifacts/preprocessor.pkl')
    try:
        X_proc = preprocessor.transform(X_sample_full)
    except:
        X_proc = preprocessor.fit_transform(X_sample_full)
    
    results = {}
    
    inertias = []
    sil_scores = []
    K_range = range(2,9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_proc)
        inertias.append(km.inertia_)
        try:
            sil = silhouette_score(X_proc, labels)
        except:
            sil = 0
        sil_scores.append(sil)
    
    kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans_final.fit_predict(X_proc)
    results['kmeans'] = {
        'inertias': inertias,
        'sil_scores': sil_scores,
        'labels': kmeans_labels.tolist()[:200],
        'cluster_centers': kmeans_final.cluster_centers_.tolist(),
        'K_range': list(K_range)
    }
    
    agg = AgglomerativeClustering(n_clusters=4)
    agg_labels = agg.fit_predict(X_proc)
    results['hierarchical'] = {
        'labels': agg_labels.tolist()[:200]
    }
    
    dbscan = DBSCAN(eps=2.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_proc)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    results['dbscan'] = {
        'labels': dbscan_labels.tolist()[:200],
        'n_clusters': n_clusters_dbscan,
        'n_noise': n_noise,
        'eps': 2.5,
        'min_samples': 10
    }
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_proc)
    results['pca'] = {
        'transformed': X_pca[:500].tolist(),
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'total_variance': float(sum(pca.explained_variance_ratio_))
    }
    
    sample_idx = np.random.choice(len(X_proc), min(800, len(X_proc)), replace=False)
    X_sample = X_proc[sample_idx]
    y_sample = y_sample_full.iloc[sample_idx] if hasattr(y_sample_full, 'iloc') else y_sample_full[sample_idx]
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
        results['tsne'] = {
            'transformed': X_tsne.tolist(),
            'categories': y_sample.tolist()
        }
    except Exception as e:
        results['tsne'] = {'error': str(e)}
    
    pca_umap = PCA(n_components=2, random_state=42)
    X_umap_sim = pca_umap.fit_transform(X_sample)
    results['umap'] = {
        'transformed': (X_umap_sim + np.random.normal(0,0.1, X_umap_sim.shape)).tolist(),
        'categories': y_sample.tolist(),
        'simulated': True
    }
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    anomaly_labels = iso.fit_predict(X_proc)
    anomaly_scores = iso.decision_function(X_proc)
    n_anomalies = list(anomaly_labels).count(-1)
    results['anomaly'] = {
        'n_anomalies': n_anomalies,
        'anomaly_rate': n_anomalies / len(X_proc),
        'scores_sample': anomaly_scores[:200].tolist(),
        'labels_sample': anomaly_labels[:200].tolist()
    }
    
    with open('artifacts/unsupervised.json','w') as f:
        json.dump(results, f, indent=2)
    
    print("Unsupervised analysis complete")
    return results

if __name__ == "__main__":
    run_unsupervised_analysis()
