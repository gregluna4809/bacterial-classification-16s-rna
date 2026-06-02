# 16S Classifier API

FastAPI inference service for the saved 16S rRNA genus classifier artifacts in:

```text
outputs/models/xgb_svd_smote/
```

## Install

```powershell
pip install -r backend\requirements.txt
```

## Run

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

## Smoke Tests

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

```powershell
$body = @{ sequence = ("ACGT" * 30) } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
```
