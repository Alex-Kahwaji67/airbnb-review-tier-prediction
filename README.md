# Airbnb Review Tier Prediction

This project predicts review-quality tiers for Montreal Airbnb listings using listing, host, availability, amenity, and location features. It reframes a skewed review-score prediction problem into a three-class classification task: Poor, Good, and Excellent.

The work is based on a Data Mining for Business Analytics coursework project and has been cleaned for portfolio staging.

## Problem

New Airbnb listings do not have review histories, which makes it harder to rank them fairly in search. The project tests whether listing characteristics available before review history accumulates can estimate a review-quality tier and support a more useful ranking signal.

## Data

The original project used Montreal Airbnb listings data from June 2025 in a `listings.csv` file. The report states that the cleaned training dataset contained 6,754 listings after dropping rows with missing selected fields, removing listings without review scores, and excluding extreme max-night outliers.

Raw data is not included in this repo. To run the notebook, place the listings file here:

```text
data/listings.csv
```

See `data/README_DATA.md` for data placement notes.

## Methods

- Data cleaning for missing values, type conversion, and extreme max-night outliers.
- Leakage control by removing IDs, metadata, open text fields, review-history fields, and post-review score columns unavailable to new listings.
- Feature engineering for host location, neighborhood lumping, property-type indicators, bathroom indicators, and top amenity indicators.
- Target reframing from continuous review score to three tiers:
  - Excellent: 4.8 to 5.0
  - Good: 4.5 to 4.8
  - Poor: below 4.5
- Model comparison using ANN, random forest, and an L1-regularized logistic workflow.

## Evaluation

The project used macro F1 because the classes were imbalanced and all tiers mattered. The original report also discusses confusion matrices and class-specific recall, especially weak identification of Good and Poor listings.

## Results

The report selected the interpretable L1/logistic workflow with macro F1 around 0.516. The ANN had similar macro F1 around 0.515, while the random forest model was lower at about 0.490. The strongest business value came from the interpretation: convenience and host-experience features were associated with stronger review tiers.

The notebook implementation also includes an L1 logistic feature-selection/interpretation step feeding an MLP classifier, so the staged notebook preserves the modeling workflow while noting the report's selected interpretable framing.

## How To Run

1. Use Python 3.10+.
2. Install the key libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

3. Place the Montreal Airbnb `listings.csv` file at:

```text
data/listings.csv
```

4. Open and run:

```text
notebooks/airbnb_review_tier_prediction.ipynb
```

## Limitations / Future Work

- The project uses one city and one data snapshot: Montreal, June 2025.
- The target is heavily imbalanced, with the report noting 58% of listings in Excellent, 29% in Good, and 13% in Poor.
- The report notes weaker recall for Good and Poor listings.
- Future work could test oversampling or undersampling, use additional external location/demand features, and apply NLP to host/listing descriptions instead of dropping open text fields.
