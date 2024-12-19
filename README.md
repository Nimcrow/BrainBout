# UFC Event Prediction Model

A machine learning model that predicts the excitement level of UFC fights based on fighter statistics and historical data.

## Project Structure

```
UFCEventPredictionModel/
├── data/
│   ├── raw/                # Raw scraped data files
│   │   ├── large_dataset.csv
│   │   ├── ufc_fights.csv
│   │   ├── ufc_events_and_fights_stats.csv
│   │   └── ufc_fighters_roster_stats_and_records.csv
│   └── processed/          # Processed and merged datasets
│       ├── merged_best_fights.csv
│       ├── Final_merged_fights.csv
│       ├── intersecting_datapoints.csv
│       ├── missing_name_in_large_dataset.csv
│       ├── X_test.csv
│       ├── X_train.csv
│       ├── y_test.csv
│       └── y_train.csv
├── models/                 # Trained models and scalers
│   ├── feature_names.csv
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── notebooks/             # Jupyter notebooks and visualizations
│   └── figures/
│       ├── model_performance.png
│       └── roc_curves.png
├── outputs/               # Model evaluation results
│   └── model_results.txt
├── Project_Iteration_results/  # Results from different iterations
├── reports/              # Project documentation
└── scripts/              # Python source code
    ├── analyze_best_fights.py
    ├── preprocessing.py
    ├── predict_fight_excitement.py
    ├── scrape_ufc_fighters.py
    └── train_models.py
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/UFCEventPredictionModel.git
cd UFCEventPredictionModel
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Collection

1. Scrape UFC fighter data:
```bash
python scripts/scrape_ufc_fighters.py
```

This script will:
- Scrape fighter profiles from UFC.com
- Save fighter statistics to `data/raw/ufc_fighters_roster_stats_and_records.csv`

## Data Processing

1. Analyze best fights:
```bash
python scripts/analyze_best_fights.py
```

2. Preprocess the data:
```bash
python scripts/preprocessing.py
```

This script:
- Loads raw data files
- Creates features and target variables
- Applies SMOTE for class balancing
- Saves processed datasets to `data/processed/`

## Model Training

Train the prediction models:
```bash
python scripts/train_models.py
```

This will:
- Train multiple models (Random Forest, Logistic Regression, SVM)
- Perform hyperparameter tuning
- Save the best model and performance metrics
- Generate visualization plots

## Making Predictions

Predict fight excitement:
```bash
python scripts/predict_fight_excitement.py
```

Features:
- Interactive command-line interface
- Fighter name fuzzy matching
- Detailed matchup analysis
- Excitement probability predictions

## Model Performance

The model evaluates fights based on multiple criteria:
- Historical fight data
- Fighter statistics
- Style matchups
- Win rates and experience
- Championship implications

Performance metrics and visualizations are saved in:
- `notebooks/figures/`
- `outputs/model_results.txt`

## Contact

abrahamalshamsie@gmail.com

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- UFC.com for fighter statistics
- Scikit-learn for machine learning tools
- SMOTE for handling class imbalance