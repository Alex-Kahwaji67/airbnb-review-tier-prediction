# Airbnb Review Tier Prediction - Scrubbed Report Summary

## Objective

The project predicts review-quality tiers for new or unreviewed Montreal Airbnb listings using listing characteristics available before review history is established.

## Data Preparation

- Dataset: Montreal Airbnb `listings` data from June 2025.
- Cleaned record count: 6,754 listings after preprocessing.
- Rows missing selected fields were dropped.
- Listings with no review score were removed from training.
- Extreme max-night outliers were removed.
- IDs, metadata, redundant columns, open text fields, and review-history variables were removed to reduce leakage and keep the feature set closer to information available before listing performance is realized.

## Target Reframing

The original review-score target was highly skewed. The project first tested regression, then reframed the problem as a three-class classification task:

- Excellent: 4.8 to 5.0
- Good: 4.5 to 4.8
- Poor: below 4.5

The report states the class distribution as 58% Excellent, 29% Good, and 13% Poor.

## Models And Evaluation

The project compared:

- ANN classifier
- L1-regularized logistic workflow
- Random forest classifier

Macro F1 was used because the classes were imbalanced and each tier mattered for ranking decisions.

Reported results:

- ANN macro F1: about 0.515
- L1/logistic workflow macro F1: about 0.516
- Random forest macro F1: about 0.490

## Interpretation

The report selected the interpretable L1/logistic workflow. Positive signals included convenience and host-experience features such as self check-in, Plateau neighborhood, host tenure, instant bookable, and coffee. Negative signals included estimated revenue, keypad, suite, smart lock, and Westmount neighborhood.

## Limitations

- Single city and time snapshot.
- Strong class imbalance.
- Weak recall for Good and Poor classes.
- Open text fields were dropped instead of modeled with NLP.

## Future Work

- Rebalance classes using oversampling or undersampling.
- Add NLP features from listing descriptions or host descriptions.
- Add external demand, transit, event, and neighborhood features.
- Validate the ranking intervention through an A/B test using booking rate, time to first booking, host retention, and realized review score.

