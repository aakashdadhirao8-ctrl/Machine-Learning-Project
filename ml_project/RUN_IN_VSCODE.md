# Run in VS Code - Step by Step

## 1. Open Folder
File -> Open Folder -> select ml_project (contains app.py)

## 2. Venv
Windows PowerShell:
```
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Mac/Linux:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Run
Option A: Press F5 -> Flask: Lens App
Option B: Terminal -> python app.py

You must see:
* Running on http://127.0.0.1:5000

## 4. Open Browser
http://127.0.0.1:5000
NOT file://, NOT Live Server 5500

## 5. Debug if clicks don't work
Open http://127.0.0.1:5000/debug
- Click Customers/Predict/Test API tabs - do they switch? If yes, nav works.
- Click Test Click Me button - does alert show? If yes, JS works.
- Click Test /api/customers - does it show Status 200? If no, Flask not running.

Then open main page http://127.0.0.1:5000 and press Ctrl+Shift+R hard refresh.

Check F12 Console for red errors.

## 6. Common Issues
- ModuleNotFoundError: No module named flask -> venv not activated
- Port 5000 in use (Mac AirPlay) -> change port in app.py to 5001
- Failed to load customers -> Flask not running or wrong URL
- Chart.js blocked -> charts skip but clicks still work (fixed)

## 7. Retrain (optional)
Artifacts already have trained model. To retrain:
```
python -m model.train
python -m model.unsupervised
```
