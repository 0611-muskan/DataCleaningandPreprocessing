# Data Preprocessing & Cleaning – California Housing Dataset 🧹📊

This document explains **only the data preprocessing and data cleaning steps** performed on the California Housing dataset before applying any machine learning model.

---

## 📌 Dataset Loading

* The dataset is loaded using `fetch_california_housing()` from `sklearn.datasets`.
* It is converted into a Pandas DataFrame for easier analysis.
* Feature names are used as column headers.
* A new column `price` is added as the target variable.

---

## 🔍 Initial Data Exploration

The following checks are performed to understand the dataset structure:

* **First 5 rows** using `df.head()`
* **Dataset shape** using `df.shape`
* **Column names, data types, and non-null counts** using `df.info()`

These steps help identify:

* Number of rows and columns
* Presence of missing values
* Data types of each feature

---

## ❓ Handling Missing Values

* Missing values are checked using `df.isnull().sum()`.
* Strategy explanation:

  * **Few missing values** → Drop rows using `dropna()`
  * **Many missing values** → Fill using mean / median / mode
* Since the California Housing dataset has **no significant missing values**, no imputation is applied.

---

## 🔁 Handling Duplicate Records

* Duplicate rows are detected using:

  ```python
  df.duplicated().sum()
  ```
* If duplicates were found, they would be removed using `drop_duplicates()`.
* This step prevents biased learning caused by repeated data.

---

## 📈 Outlier Detection

### Statistical Analysis

* `df.describe()` is used to inspect:

  * Minimum and maximum values
  * Mean
  * Percentiles (25%, 50%, 75%)
* Helps in identifying abnormal values.

### Visualization

* A boxplot is generated for the `price` column.
* Boxplots help detect:

  * Median
  * Interquartile range (IQR)
  * Extreme values (outliers)

---

## 🚫 Outlier Removal Using Z-Score

* Only numeric columns are selected.
* Z-score is calculated for each numeric feature.
* Rows with absolute Z-score ≥ 3 are treated as outliers.
* These rows are remov
