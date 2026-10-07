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
2. From the repository root, install the notebook dependencies into your Python environment:

```bash
python -m pip install -r requirements.txt
```

3. Place the Montreal Airbnb `listings.csv` file at:

```text
data/listings.csv
```

4. Open the notebook in Jupyter or a notebook-capable editor using that same Python environment. Run all cells in order from a fresh kernel, with `notebooks/` as the working directory so `../data/listings.csv` resolves correctly:

```text
notebooks/airbnb_review_tier_prediction.ipynb
```

The requirements list only third-party libraries imported by the notebook; install your preferred notebook frontend separately if needed. The pandas and scikit-learn upper bounds preserve the existing categorical-dtype checks and L1 penalty API. These are compatibility requirements, not a lockfile for the original coursework environment. `src/clean_notebook.py` is a staging utility that requires the original source notebook outside this repository; it is not needed to run the staged analysis.

Without the original listings file, import and synthetic-data smoke checks can validate code paths, but they cannot reproduce the reported record count or model scores. The results above are retained from the coursework report; reproducing them requires the original dataset and a full notebook run.

## Limitations / Future Work

- The project uses one city and one data snapshot: Montreal, June 2025.
- The target is heavily imbalanced, with the report noting 58% of listings in Excellent, 29% in Good, and 13% in Poor.
- The report notes weaker recall for Good and Poor listings.
- Future work could test oversampling or undersampling, use additional external location/demand features, and apply NLP to host/listing descriptions instead of dropping open text fields.
