# NBA Game Prediction System

A comprehensive machine learning system for predicting NBA game outcomes, spreads, and betting recommendations.

## 🏀 Project Overview

This system uses advanced statistical analysis and machine learning to predict:
- Game outcomes (win/loss)
- Point spreads
- Over/under totals
- Betting recommendations with confidence levels

## 📊 Features

- **Data Collection & Processing**
  - Automated fetching of NBA statistics
  - Advanced metrics integration
  - Real-time odds tracking
  - Historical data analysis

- **Machine Learning Models**
  - XGBoost-based prediction models
  - Feature engineering for game context
  - Rolling statistics and trends
  - Opponent-adjusted metrics

- **Betting Analysis**
  - Unit-based betting recommendations
  - Confidence scoring
  - Historical performance tracking
  - ROI analysis

## 🛠️ Technical Stack

- **Core Technologies**
  - Python 3.8+
  - XGBoost
  - Pandas
  - Scikit-learn
  - NumPy

- **Data Sources**
  - NBA API
  - Sportsbook odds
  - Advanced statistics
  - Tracking data

## 📋 Project Structure

```
NBA_Prediction/
├── data/                    # Data storage
│   ├── raw/                # Raw data files
│   │   ├── advanced/       # Advanced statistics
│   │   ├── betting/       # Betting odds
│   │   ├── core/          # Core statistics
│   │   ├── defense/       # Defensive metrics
│   │   ├── matchups/      # Matchup data
│   │   └── tracking/      # Player tracking data
│   └── processed/         # Processed data
├── scripts/                # Python scripts
│   ├── fetch/             # Data fetching
│   ├── process/           # Data processing
│   ├── train/             # Model training
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks
├── models/                # Trained models
└── output/                # Prediction outputs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/TechWorksAZ/NBA_Prediction.git
cd NBA_Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

### Usage

1. Fetch data:
```bash
python scripts/fetch/fetch_all.py
```

2. Process features:
```bash
python scripts/process/engineer_features.py
```

3. Train models:
```bash
python scripts/train/train_model.py
```

4. Generate predictions:
```bash
python scripts/predict/generate_predictions.py
```

## 📈 Model Performance

- Win/Loss Prediction Accuracy: [TBD]
- Spread Prediction MAE: [TBD]
- Total Points Prediction MAE: [TBD]
- Betting ROI: [TBD]

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

TechWorksAZ - [GitHub Profile](https://github.com/TechWorksAZ)

## 🙏 Acknowledgments

- NBA API for providing game data
- Sportsbook odds providers
- Open source machine learning community
