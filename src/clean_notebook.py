import json
from pathlib import Path


SOURCE_NOTEBOOK = Path(
    "../../../01_source_material/data-mining-for-analytics/INSY446 –Group 3 – Python.ipynb"
)
OUTPUT_NOTEBOOK = Path("../notebooks/airbnb_review_tier_prediction.ipynb")


def markdown(title, body=""):
    source = [title + "\n"]
    if body:
        source.append(body + "\n")
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def clean_cell(cell):
    cleaned = dict(cell)
    cleaned["metadata"] = {}
    cleaned["outputs"] = []
    cleaned["execution_count"] = None
    source = "".join(cleaned.get("source", []))
    source = source.replace(
        "pd.read_csv('/Users/weeb/Library/Mobile Documents/com~apple~CloudDocs/*INSY 446/Team proj/Airbnb/listings.csv')",
        "pd.read_csv('../data/listings.csv')",
    )
    source = source.replace("df_cleaner.drop(", "df_clean.drop(")
    cleaned["source"] = source.splitlines(True)
    return cleaned


def main():
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    cells = [
        markdown(
            "# Airbnb Review Tier Prediction",
            "Cleaned portfolio notebook. Place the Montreal Airbnb `listings.csv` file in `../data/` before running.",
        ),
        markdown(
            "## 1) Intro & problem",
            "Predict listing review-quality tiers for Montreal Airbnb listings so new or unreviewed listings can be ranked with a quality signal.",
        ),
        clean_cell(nb["cells"][0]),
        clean_cell(nb["cells"][1]),
        markdown(
            "## 2) Data loading / cleaning",
            "Load the listings data, remove metadata/text fields, drop missing target rows, remove extreme max-night outliers, and convert strings to typed fields.",
        ),
        *[clean_cell(nb["cells"][i]) for i in [4, 10, 21, 23]],
        markdown(
            "## 3) Feature engineering & leakage control",
            "Remove review-history fields unavailable for new listings, lump categories, engineer property and amenity indicators, and create rating tiers.",
        ),
        *[clean_cell(nb["cells"][i]) for i in [33, 34, 35, 36, 38, 39, 41, 43, 45, 60, 62, 73, 74, 75, 76, 77]],
        markdown(
            "## 4) Model training & comparison",
            "Compare ANN, L1-regularized logistic workflow, and random forest approaches using macro F1.",
        ),
        *[clean_cell(nb["cells"][i]) for i in [79, 80, 81, 82, 93, 94]],
        markdown(
            "## 5) Evaluation",
            "Evaluate macro F1, class distribution, predictions, and confusion matrices.",
        ),
        *[clean_cell(nb["cells"][i]) for i in [87, 88, 89, 90]],
        markdown(
            "## 6) Interpretation / business takeaways",
            "The report selected the interpretable L1/logistic workflow and linked stronger review tiers to convenience and host-experience signals. Class imbalance remains the main limitation.",
        ),
    ]

    cleaned_notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(json.dumps(cleaned_notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
