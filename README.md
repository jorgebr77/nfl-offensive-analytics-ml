# NFL Offensive Analytics: Play-Style Clustering and Next-Season Forecasting

An applied machine-learning portfolio project studying NFL offensive identity and predicting next-season offensive efficiency using play-by-play data from **2018–2024**.

## Key results

- **224 team-season profiles** and **four interpretable offensive clusters** using K-Means.
- **49.48%** year-to-year cluster retention under the project's cluster assignments.
- **2024 holdout:** Ridge MAE **0.4496**, versus persistence MAE **0.5305** (**15.25% lower**).
- **Rolling-origin retrospective evaluation, 2021–2024:** Ridge MAE **0.4308**, versus persistence MAE **0.4969** (**13.31% lower**), across **128 team-season forecasts**; Ridge had lower MAE in **all four years**.
- Forecasting fit remains modest (2024 Ridge R² **0.1443**). The very high same-season regression R² (**0.9937**) is **descriptive**, not evidence of future predictive accuracy.

## Visual results

![Rolling-origin MAE comparison](outputs/figures/rolling_origin_mae_comparison.png)

![PCA view of offensive clusters](outputs/figures/pca_clusters.png)

![Cluster EPA comparison](outputs/figures/cluster_epa_per_drive.png)

## Methods

1. Load NFL play-by-play Parquet files and engineer offensive team-season metrics.
2. Standardize selected features, compare silhouette scores, and interpret a four-cluster K-Means solution; visualize with PCA.
3. Evaluate same-season Ridge regression separately as a descriptive analysis.
4. Forecast season *t+1* EPA per drive from season *t* characteristics using a Ridge pipeline (median imputation, standardization, Ridge with fixed alpha=10).
5. Compare against a **persistence baseline** (next-season efficiency equals previous-season efficiency), with a 2023 validation year and 2024 holdout, plus expanding-window retrospective evaluation for 2021–2024.
6. Describe playoff appearance associations without causal claims.

## Reproduce

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/portfolio_analysis.ipynb
```

Obtain compatible NFL play-by-play data (e.g., from the [nflverse](https://github.com/nflverse/nflverse-data) ecosystem) and place the following seven files in `data/`:

```
data/play_by_play_2018.parquet
...
data/play_by_play_2024.parquet
```

**Important:** The notebook expects the columns specified in `REQUIRED_COLUMNS`, and results depend on dataset version, source, and preprocessing. The original analysis used supplied season Parquet files; downloading another version may require harmonization. Raw data are intentionally not included in this repository.

If your data live elsewhere, set an environment variable before launching Jupyter:

```powershell
$env:NFL_DATA_DIR = 'C:\path\to\your\parquet-folder'
```

Then use **Kernel → Restart & Run All**. Tables are saved to `outputs/tables/` and figures to `outputs/figures/`. Tables are ignored by Git to avoid accidental large or sensitive data commits.

## Limitations

- The sample comprises only seven seasons; rolling-origin test windows each contain 32 team-season observations. Team-seasons may be dependent across years.
- Four clusters were chosen for interpretation; **two clusters had a slightly higher silhouette score** in the reported experiment.
- The same-season EPA-per-play → EPA-per-drive model uses closely related quantities and should not be interpreted as forward prediction.
- Fixed Ridge hyperparameters were not selected through nested time-series tuning. The retrospective 2024 result is not an untouched second holdout after inspection.
- Associations between cluster membership and playoff appearances are descriptive, not causal.
- `fourth_down_go_rate` in the existing features is a fourth-down offensive-play share, not a conventional fourth-down conversion/attempt rate. This should be corrected or renamed in future work.

## Repository contents

- `notebooks/portfolio_analysis.ipynb` — updated notebook with executed figures and evaluation.
- `notebooks/original_academic_analysis.ipynb` — preserved original course notebook.
- `outputs/figures/` — exported visualizations from the updated notebook.
- `docs/` — original academic report and slides.
- `requirements.txt` — Python dependencies.

**Project context:** University of Colorado Boulder MS in Data Science; portfolio refinement performed after the original academic submission.
