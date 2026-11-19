# Retail Store Inventory & Demand Forecasting

An end-to-end retail analytics and machine-learning project focused on daily product demand forecasting, inventory replenishment decisions, sell-signal classification, and a retrieval-augmented analytics assistant.

The project uses the **Retail Store Inventory Forecasting Dataset** from Kaggle:
https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset

> **Data note:** The source dataset is synthetic. Forecasting performance should therefore be interpreted in the context of this dataset and not as a guarantee of performance on live retail operations.

## Project objectives

- Explore demand, revenue, pricing, promotions, seasonality, and inventory patterns.
- Compare baseline forecasting approaches with a global LightGBM model.
- Evaluate forecasts with time-based validation and a held-out test period.
- Translate forecasts into replenishment and service-level trade-offs.
- Classify whether upcoming sales are likely to beat a recent run-rate.
- Provide a SQL-backed RAG assistant for questions about the warehouse and model outputs.

## Repository structure

```text
retail_forecasting/
├── data/                 # Place the source CSV here (not committed)
├── docs/                 # Architecture and design notes
├── models/               # Trained model artifacts
├── notebooks/            # EDA, forecasting, sell model, and RAG experiments
├── outputs/              # Evaluation tables, charts, and report
├── schema/               # SQLite and PostgreSQL schemas
├── src/                  # Reusable Python pipeline modules
├── warehouse/            # Generated SQLite analytical warehouse
└── requirements.txt
```

## Main workflow

1. **Data preparation and EDA** — standardize fields, validate records, derive business measures, and explore trends and drivers.
2. **Demand forecasting** — compare naive, seasonal-naive, moving-average, same-weekday, LightGBM, and Holt-Winters approaches.
3. **Forecast evaluation** — use a chronological train/validation/test design and report MAE, RMSE, WAPE, bias, and sMAPE.
4. **Inventory simulation** — compare replenishment policies at a common service-level target and estimate stock, holding-cost, and lost-margin trade-offs.
5. **Sell-signal model** — estimate the probability that the next seven days' sales exceed the recent run-rate, then map probabilities to HOT / NORMAL / COLD signals.
6. **Analytics assistant** — build a searchable knowledge base and SQL tools to answer questions grounded in warehouse data.

## Selected results

The current experiment reports the following hold-out results:

| Area | Result |
|---|---:|
| LightGBM demand forecast WAPE | 64.6% |
| Naive baseline WAPE | 87.3% |
| LightGBM MAE skill vs. naive | 26.0% |
| Holt-Winters chain-level WAPE | 5.9% |
| Sell-signal ROC-AUC | 0.686 |
| RAG retrieval hit@1 | 0.92 |
| RAG retrieval hit@3 | 1.00 |
| Query-routing accuracy | 16/16 |

These are reported experiment outputs for this dataset and split. They are not independently verified production results.

## Getting started

### 1. Set up the environment

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

Download the CSV from the Kaggle dataset page and place it at:

```text
data/retail_store_inventory.csv
```

The raw data file is intentionally not included in this repository.

### 3. Run the forecasting pipeline

```bash
python src/retail_pipeline.py --data data/retail_store_inventory.csv --out outputs --pdf
```

### 4. Build the warehouse and run the extended workflow

```bash
python src/run_all.py
```

For the RAG assistant:

```bash
python src/rag_assistant.py "Which Electronics products will sell best next week?"
python src/rag_assistant.py -i
```

The first dense knowledge-base build may download the `BAAI/bge-small-en-v1.5` embedding model. Use the project's `--no-dense` option for BM25-only retrieval.

## Key modelling considerations

- The source dataset's `Demand Forecast` field is highly correlated with `Units Sold` and is excluded from model inputs to avoid target leakage.
- `Inventory Level` and `Units Ordered` are also excluded from demand-model features.
- Evaluation uses chronological splits rather than random train/test splitting.
- The sell-signal target is defined relative to a trailing run-rate because a simple zero-sales classification target is highly imbalanced.
- Inventory policy comparisons should be read alongside the chosen service-level target and cost assumptions.

## Limitations

- The dataset is synthetic and may not represent real store operations.
- Series-level demand can be noisy, and some external drivers may have weak relationships with demand.
- Results depend on the dataset, feature definitions, time split, and business-cost assumptions.
- The RAG assistant's answers are limited by the warehouse, knowledge-base content, retrieval quality, and optional language-model configuration.

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, statsmodels, SQLite, SQL, matplotlib, seaborn, Jupyter, and retrieval-augmented generation.

## License

No license has been added yet. Unless a license is added, normal copyright restrictions apply to reuse of this repository.
