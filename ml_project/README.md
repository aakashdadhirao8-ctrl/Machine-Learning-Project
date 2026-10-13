# CustomerLens - Multi-Class Customer Category Prediction (Kaggle 50K Real Data)

**Topic:** Multi-Class Customer Category Prediction Using Multinomial Logistic Regression
**Dataset:** Kaggle Online Retail Customer Segmentation - 50,000 customers, 4 classes (Occasional, Regular, Loyal, High_Value)
**Accuracy:** 66.4% test, 0.86 ROC-AUC, 66.6% CV mean

## Quick Start in VS Code

1. Open folder `ml_project` in VS Code
2. Create venv:
   - Windows: `python -m venv venv` then `.\venv\Scripts\Activate.ps1`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Run: `python app.py`
5. Open: http://127.0.0.1:5000

If clicks don't work:
- Make sure you opened http://127.0.0.1:5000 NOT file://
- Open http://127.0.0.1:5000/debug to test
- Check F12 Console for errors
- Hard refresh Ctrl+Shift+R

## Structure
- app.py - Flask REST, 6-stage trace lineage
- model/preprocessing.py - Feature Store v3.0-kaggle
- model/train.py - 70/15/15 split, LogReg + RF
- model/unsupervised.py - KMeans, PCA, t-SNE
- templates/index.html - Lens v3.0 authentic tool
- templates/debug.html - minimal debug page
- artifacts/customers.csv - 50k real Kaggle data
- CustomerLens_Kaggle50K_BoxOffice_Style.pptx - 17 slides final deck

## API
- GET /api/customers - 120 sampled
- POST /api/predict - 12 features -> 4 classes
- GET /api/metrics - 66.4% accuracy
- GET /debug - debug page
