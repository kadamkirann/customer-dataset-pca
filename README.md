# Customer Dataset Dimensionality Reduction using PCA

## Aim
Apply Principal Component Analysis (PCA) to customer spending features, reduce dimensionality, and compare PCA performed on original data with PCA performed after StandardScaler.

## Dataset
UCI Wholesale Customers dataset (440 records). The six numerical annual spending features used are:
- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

`Channel` and `Region` are descriptive categorical columns and are not used as PCA inputs.

Source: UCI Machine Learning Repository, Wholesale Customers. DOI: https://doi.org/10.24432/C5030X (CC BY 4.0).

## Files
- `Customer_Dataset_PCA.ipynb` — complete notebook experiment
- `app.py` — Streamlit application
- `Project_Report.pdf` — project report
- `requirements.txt` — Python dependencies

## Run the notebook
1. Open https://colab.research.google.com/
2. Upload `Customer_Dataset_PCA.ipynb`.
3. Select Runtime > Run all.
4. Internet access is needed to download the dataset.

## Run the Streamlit app
```bash
pip install -r requirements.txt
streamlit run app.py
```
Internet access is needed to download the dataset.

## Method
1. Load and inspect the customer dataset.
2. Select six numerical spending features.
3. Explore distributions and correlations.
4. Apply PCA without scaling.
5. Apply StandardScaler, then PCA.
6. Compare explained variance and cumulative explained variance.
7. Plot customers using the first two principal components.

## Important note
This is an unsupervised dimensionality-reduction experiment, not a customer purchase prediction model. Execute the notebook before submission and use its actual output values when discussing results.
