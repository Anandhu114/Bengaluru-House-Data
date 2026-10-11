# Bengaluru House Price Prediction using Machine Learning

## Project Overview

This project focuses on predicting residential property prices in Bengaluru, Karnataka, India, using Machine Learning techniques.

The project uses a housing dataset containing information about property locations, total area, number of bedrooms, bathrooms, and prices. The dataset is cleaned and preprocessed before applying regression algorithms to predict house prices.

The main objective is to explore real-estate data, identify and remove outliers, transform categorical features into numerical values, train regression models, and evaluate their performance.

## Objectives

- Analyze Bengaluru housing data using Python.
- Clean missing values and remove duplicate records.
- Handle inconsistent property-area values and categorical features.
- Detect and remove unrealistic property-price and size outliers.
- Perform exploratory data analysis using visualizations.
- Build and evaluate Machine Learning regression models.
- Compare different algorithms to identify a suitable model for house price prediction.

## Technologies and Libraries

The project uses the following technologies:

- **Python** — Core programming language.
- **Pandas** — Data manipulation and preprocessing.
- **NumPy** — Numerical operations and statistical calculations.
- **Matplotlib** — Data visualization.
- **Scikit-learn** — Model training, cross-validation, hyperparameter tuning, and evaluation.
- **Jupyter Notebook** — Interactive development and analysis environment.

## Dataset Description

The project uses `Bengaluru_House_Data.csv`, containing 13,320 property records and nine original columns.

| Column | Description |
|---|---|
| `area_type` | Type of property area measurement |
| `availability` | Property availability information |
| `location` | Location of the property |
| `size` | Property configuration, such as 2 BHK or 4 Bedroom |
| `society` | Society or residential community information |
| `total_sqft` | Total property area in square feet |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |
| `price` | Property price |

The original dataset contains missing values, duplicate entries, inconsistent area formats, and potential outliers.

**Dataset requirement:** The notebook loads `Bengaluru_House_Data.csv` directly. Place this CSV file in the same directory as the notebook before running the project.

## Project Workflow

### 1. Data Loading

The dataset is loaded into a Pandas DataFrame for analysis and preprocessing.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Bengaluru_House_Data.csv")
```

The initial dataset contains 13,320 rows and nine columns.

### 2. Data Cleaning

The following preprocessing steps are performed:

- Remove the `society`, `balcony`, `availability`, and `area_type` columns.
- Inspect the dataset structure and missing values.
- Remove rows containing missing values.
- Identify and remove duplicate records.

After dropping the selected columns, the dataset contains five columns. After removing missing values and duplicates, 12,365 records remain.

### 3. Feature Engineering

#### Extracting BHK

The original `size` column contains values such as `2 BHK`, `3 BHK`, and `4 Bedroom`.

A new numerical feature named `BHK` is created by extracting the number from the `size` column.

```python
df["BHK"] = df["size"].apply(
    lambda x: int(x.split(" ")[0])
)
```

This allows the number of bedrooms to be used directly by regression models.

#### Converting Total Square Feet

The `total_sqft` column contains numeric values as well as ranges such as `1133 - 1384`.

A custom function converts numeric values into floats and replaces a range with its midpoint.

For example, `1133 - 1384` becomes `1258.5`.

#### Calculating Price per Square Foot

A new feature called `price_per_sqft` is calculated to help analyze property prices.

```python
df["price_per_sqft"] = (
    df["price"] * 100000 / df["total_sqft"]
)
```

This feature is primarily used for identifying unusual prices and removing outliers.

### 4. Location-Based Data Processing

The dataset contains many unique property locations.

The notebook:

- Removes extra whitespace from location names.
- Counts the number of properties in each location.
- Identifies locations with ten or fewer records.
- Groups these less-frequent locations into an `other` category.

This reduces the number of separate location categories and helps simplify categorical feature encoding.

### 5. Outlier Removal

Outlier removal is an important part of the project because unusual property sizes and prices can affect regression performance.

#### A. Minimum Area per Bedroom

Records with less than 300 square feet per bedroom are removed.

```python
df = df[~(df.total_sqft / df.BHK < 300)]
```

This is used as a heuristic to identify potentially unrealistic property records.

#### B. Price-per-Square-Foot Outliers

For each location, the notebook calculates the mean and standard deviation of the price per square foot.

Records outside one standard deviation from the location-specific mean are removed.

#### C. BHK-Based Outlier Removal

The notebook compares properties with different bedroom counts within the same location.

For a given BHK category, lower price-per-square-foot records are removed when the preceding BHK category has more than five records and provides the comparison statistics.

This step is intended to remove cases where larger-bedroom properties have unusually low prices relative to smaller-bedroom properties in the same location.

#### D. Bathroom Outliers

The notebook examines bathroom counts and removes records where the number of bathrooms is greater than or equal to `BHK + 2`.

These filtering steps help reduce extreme observations before training the regression models.

### 6. Exploratory Data Analysis (EDA)

The notebook uses Matplotlib to visualize housing data and inspect relationships between property characteristics.

The visualizations include:

- **Scatter plots:** Compare property prices and total area for 2-BHK and 3-BHK properties in Rajaji Nagar.
- **Price-per-square-foot histogram:** Examine the distribution of property prices per square foot.
- **Bathroom-count histogram:** Understand the distribution of bathroom counts.
- **Statistical summaries:** Inspect the distribution of price-per-square-foot values.

These visualizations help identify unusual observations and understand patterns in the dataset.

### 7. Encoding Categorical Variables

Machine Learning regression models require numerical input features.

The notebook uses one-hot encoding to convert the `location` column into numerical indicator columns.

```python
dummies = pd.get_dummies(
    df["location"]
).astype(int)
```

The dummy columns are added to the dataset, the `other` dummy column is dropped as the reference category, and the original `location` column is removed.

The `size` and `price_per_sqft` columns are also removed before model training because `BHK` already represents bedroom count and `price_per_sqft` is derived from the target price.

The final preprocessed dataset contains 6,733 records and 235 columns.

### 8. Feature Selection

The target variable is `price`.

All remaining columns, including `total_sqft`, `bath`, `BHK`, and the encoded location features, are used as input features.

```python
X = df.drop("price", axis="columns")
y = df["price"]
```

- **Input features (`X`):** Property size, bathrooms, BHK, and location indicators.
- **Target (`y`):** Property price.

### 9. Train-Test Split

The dataset is split into training and testing subsets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
```

- Training set: 80% of the data.
- Testing set: 20% of the data.

The training data is used to fit the model, while the testing data is used to evaluate its performance on unseen records.

### 10. Machine Learning Models

The notebook explores three regression algorithms.

#### Linear Regression

Linear Regression models the relationship between property features and prices. It is used as the primary prediction model in the notebook.

#### Lasso Regression

Lasso is a regularized linear regression algorithm that applies an L1 penalty. It can help reduce the influence of less useful features.

#### Decision Tree Regressor

Decision Tree Regression learns feature-based decision rules to estimate property prices. It can model nonlinear relationships between property characteristics and prices.

### 11. Model Evaluation and Cross-Validation

The notebook evaluates Linear Regression using its test-set `score()` method, which returns the coefficient of determination, R².

It also uses `ShuffleSplit` cross-validation with five splits and a test size of 20%.

This evaluates model performance across multiple randomized train-test partitions rather than relying on a single split.

### 12. Hyperparameter Tuning

The project uses `GridSearchCV` to evaluate different hyperparameter combinations for each regression algorithm.

The tested parameters are:

| Model | Parameters explored |
|---|---|
| Linear Regression | `fit_intercept`, `positive` |
| Lasso Regression | `alpha`, `selection` |
| Decision Tree Regressor | `criterion`, `splitter` |

The notebook uses five `ShuffleSplit` cross-validation splits with a test size of 20%.

The results are collected in a DataFrame containing each model's best cross-validation score and best parameter combination.

## Results

### Linear Regression Performance

The saved notebook output reports the following results:

| Evaluation | R² score |
|---|---:|
| Linear Regression test-set score | 0.8563 |
| Cross-validation split 1 | 0.8640 |
| Cross-validation split 2 | 0.7487 |
| Cross-validation split 3 | 0.8644 |
| Cross-validation split 4 | 0.7901 |
| Cross-validation split 5 | 0.8937 |
| Linear Regression best GridSearchCV score | 0.8324 |
| Lasso best GridSearchCV score | 0.6818 |
| Decision Tree best GridSearchCV score | 0.6885 |

The test-set R² score of approximately 0.856 indicates that the fitted Linear Regression model explains about 85.6% of the variance in the test-set target values for that particular split.

The cross-validation results vary between splits, showing that performance depends on the data partition.

Among the three models evaluated through GridSearchCV, Linear Regression achieved the highest reported best cross-validation score.

**Note:** These scores are the outputs saved in the notebook, not a guarantee of identical results on every run. The notebook does not set a `random_state` for its initial train-test split, so the test score can change when the experiment is rerun.

## Project Structure

A recommended GitHub repository structure is:

```text
Bengaluru-House-Price-Prediction/
│
├── 14_Bengaluru_House_Data.ipynb
├── Bengaluru_House_Data.csv
├── README.md
└── requirements.txt
```

The CSV file is required to execute the notebook. If the dataset is not distributed with the repository, document its original source and instructions for obtaining it.

## Installation and Setup

### Prerequisites

- Python 3.9 or a compatible Python version.
- Jupyter Notebook or JupyterLab.
- The required Python libraries.
- The housing CSV dataset.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd Bengaluru-House-Price-Prediction
```

Replace `<your-repository-url>` with the actual GitHub repository URL.

### Step 2: Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Step 3: Add the Dataset

Place `Bengaluru_House_Data.csv` in the project directory, alongside the notebook.

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `14_Bengaluru_House_Data.ipynb` and run the cells in sequence.

## Future Improvements

- Build a simple web interface using Flask or Streamlit.
- Add a user-friendly property price prediction form.
- Improve model selection using additional regression algorithms.
- Evaluate models using MAE, MSE, RMSE, and R².
- Set reproducible random seeds and improve validation methodology.
- Save the trained model and preprocessing pipeline for reuse.
- Add robust handling for unseen locations and invalid user inputs.
- Explore feature scaling, feature selection, and more systematic outlier detection.
- Document the dataset source and add automated tests.

## Key Learnings

Through this project, I explored:

- Data cleaning and preprocessing using Pandas.
- Feature engineering for real-estate data.
- Handling missing values and duplicate records.
- Converting string-based area ranges into numerical values.
- Location-based grouping and one-hot encoding.
- Outlier detection using statistical methods.
- Exploratory data analysis with Matplotlib.
- Training regression models using Scikit-learn.
- Cross-validation and hyperparameter tuning using `GridSearchCV`.

## Conclusion

This project demonstrates an end-to-end Machine Learning workflow for Bengaluru house price prediction, from raw-data preprocessing and exploratory analysis to regression modeling and model evaluation.

The experiments show that Linear Regression achieved the strongest reported cross-validation performance among the three evaluated algorithms. The project provides a foundation for further experimentation and for developing a property price estimation application.

## Author

**Anandhu P S**

AI/ML Trainer | Data Science | Machine Learning | Python

- GitHub: [Anandhu114](https://github.com/Anandhu114)
- LinkedIn: [anandhups36](https://www.linkedin.com/in/anandhups36)
