# Flask API and model evaluation

From the project root:

```bash
pip install -r requirements.txt
python backend\analytics.py
python backend/app.py
```

Open http://127.0.0.1:5000. The API loads `models\ipl_winner_model.pkl`, serves `/api/options`, `/api/metrics`, `/api/analytics`, `/api/health`, and `/api/predict`, and serves the Power BI assets under `/reports/`.

The analytics and evaluation commands look for `matches.csv` and `deliveries.csv` in `Dataset\` and then Windows `Downloads`. Configure `IPL_MATCHES_CSV` and `IPL_DELIVERIES_CSV` to use other locations. The user identified the CSVs as coming from Kaggle; the exact dataset and URL were not provided, and the license is unknown.

The model artifact was saved with scikit-learn 1.6.1, so the project pins that version for reproducibility. Predictions use the exact seven input features from Module 6 of the notebook.

The evaluator excludes no-result matches for winner classification, evaluates a chronological 80/20 holdout and the latest five seasons with rolling-origin splits, saves metrics and source-file hashes in `models\model_evaluation.json`, then fits the saved pipeline on all labeled matches. The current measured Random Forest is below the toss-winner baseline and remains experimental.

Run the focused checks with `python -m unittest backend.test_app`.
