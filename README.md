# Customer Churn Predictor

An explainable data-science mini-project that estimates customer churn probability from tenure, monthly charges, support-ticket volume, and usage hours.

## Overview

Subscription businesses use churn signals to prioritize helpful retention outreach. This project demonstrates a complete binary-classification workflow with standardized features and logistic regression implemented from scratch.

## Problem statement

Given recent customer behavior indicators, estimate the likelihood that a customer will churn. The result is a prioritization signal, not a definitive judgment about an individual customer.

## Features

- Standardizes numeric customer features.
- Trains logistic regression with gradient descent.
- Reports probability and a retention-outreach recommendation.
- Includes reproducible CSV data, tests, and CLI.
- Uses only Python's standard library.

## Technology stack

- Python 3.10+
- Logistic regression and gradient descent
- Feature standardization
- CSV, `argparse`, and `math`
- `unittest` for automated testing

## Installation

```bash
git clone https://github.com/mohan843/customer-churn-predictor.git
cd customer-churn-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime packages are required.

## Execution

Lower-risk example:

```bash
python predictor.py --tenure-months 36 --monthly-charges 45 --support-tickets 0 --usage-hours 42
```

Higher-risk example:

```bash
python predictor.py --tenure-months 3 --monthly-charges 88 --support-tickets 5 --usage-hours 12
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training accuracy: 100.0%
Churn probability: 0.0%
Customer likely to remain
```

## Responsible use and limitations

The sample dataset is synthetic and small. Real churn systems should use time-based validation, monitor performance across customer segments, avoid discriminatory proxies, respect privacy, and treat predictions as prompts for supportive service—not automatic cancellation or denial.

## Future improvements

- Add calibration, precision-recall metrics, and time-based holdouts.
- Include product events, plan changes, and customer feedback.
- Compare logistic regression with tree ensembles and survival analysis.
- Add explainability for retention teams.
- Monitor drift and intervention outcomes.
