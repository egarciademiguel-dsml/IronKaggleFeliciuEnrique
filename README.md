King County Houses Price Prediction - Notebook Walkthrough
Project Overview
This notebook predicts house prices in King County using machine learning, focusing on a dataset of properties with a minimum price of $650,000. The analysis combines exploratory data analysis, feature engineering, geospatial analysis, and advanced modeling techniques to achieve optimal predictions.

1. Data Import and Exploration
Libraries Setup
Core libraries: NumPy, Pandas
Visualization: Matplotlib, Seaborn
Geospatial: GeoPandas, Contextily
ML: Scikit-learn, LightGBM, XGBoost components
Explainability: SHAP
Initial Data Assessment
Loaded King County houses dataset
Dataset contains 21,613 records with features including: price, bedrooms, bathrooms, sqft_living, grade, condition, waterfront, view, lat, long, zipcode, date, and derived temporal features
Identified data types, missing values, and distribution patterns
2. Exploratory Data Analysis (EDA)
Feature Categorization
Categorical Features: waterfront, view, zipcode, condition, grade, floors, bathrooms, bedrooms
Numerical Features: sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15, yr_built, yr_renovated
Price Distribution Analysis
Price distribution is right-skewed (non-normal)
Applied log transformation to normalize: price_log = np.log(price)
This improves model performance for regression algorithms
Outlier Detection
Identified extreme outlier: one house with 30+ bedrooms
Removed houses with bedrooms ≥ 10 to clean dataset
3. Seasonality Study
Seasonal Price Patterns
Created season feature based on sale month (Winter, Spring, Summer, Autumn)
Finding: Only 6.44% price difference between winter and spring sales
Conclusion: Seasonality has minimal impact on price
4. Correlation Matrix Analysis
Key Correlations with Price
Strong predictors (0.40+):

sqft_living (0.70) - strongest physical predictor
grade (0.67) - construction quality
bathrooms (0.53) - house size indicator
view (0.40) - value indicator
Weak raw correlations (required transformation):

yr_built (0.05) - solved by creating house_age feature
yr_renovated (0.13) - solved by creating is_renovated binary feature
Grade Impact: Price increases exponentially (not linearly) with house grade

Waterfront Impact: Significantly boosts price; even non-waterfront homes can reach extreme prices through other factors

5. Geospatial Analysis
Geographic Clustering
Used KMeans clustering (k=70) to create price-based neighborhoods from lat/long coordinates
Generated 70 synthetic geographical clusters representing distinct price zones
Zipcode vs. Geographic Clusters
Compared zipcode-based vs. cluster-based models
Result: Geographic clusters slightly outperformed zipcode in R² score
Purity average (~0.72): Zipcode partially aligns with geography but misses intra-zipcode variation
Feature Engineering
Created dist_to_center: Euclidean distance from Seattle city center (47.62°N, 122.33°W)
Created geospatial interaction features:
lat_long = lat × long
lat_sq = lat²
long_sq = long²
6. Data Processing and Feature Engineering
Feature Transformations
Date processing: Converted to UTC datetime, extracted temporal features
Renovation encoding: is_renovated = 1 if yr_renovated > 0, else 0
Age features:
house_age = sale_year - yr_built
house_age_renovated = sale_year - yr_renovated
Temporal Feature Extraction
year, month, day, dayofweek, quarter from sale date
Target Encoding for Zipcode
Computed mean price per zipcode in training data only
Mapped to test zipcodes; unseen zipcodes filled with global mean
This preserves information while avoiding data leakage
Final Dataset Features
27 numerical/engineered features
Dropped: yr_built, yr_renovated, id, date_only, season, neighborhood_group, dist_to_center, sqft_price
Kept: zipcode_te (target-encoded) instead of raw zipcode
7. Train/Test Split with Time Series Strategy
Time-Based Split (80/20)
Sorted data by sale date (oldest → newest)
Training set: First 80% (historical data)
Test set: Last 20% (most recent data)
This prevents data leakage in temporal patterns
Cross-Validation: TimeSeriesSplit
Used TimeSeriesSplit(n_splits=5) to respect temporal order
Ensures models don't train on future data
8. Model Selection and Evaluation
Model Dictionary (11 models total)
Ensemble Models:

Random Forest (600 estimators)
Extra Trees (600 estimators)
Gradient Boosting (800 estimators)
AdaBoost (300 estimators)
LightGBM (2000 estimators)
Linear Models:

Linear Regression
Ridge, Lasso, ElasticNet (with scaling)
Distance/Instance-Based:

KNeighborsRegressor (with scaling)
Support Vector Regressor (RBF kernel, with scaling)
Single Tree:

Decision Tree
Evaluation Function
Used TimeSeriesSplit cross-validation with 5 splits
Metrics computed on real price scale (de-logged predictions):
R² score
RMSE
MAE
MAPE (Mean Absolute Percentage Error)
Results
Top 3 models identified based on MAPE
LightGBM emerged as best performer
9. LightGBM Hyperparameter Tuning
Hyperparameter Search Space
Search Strategy
RandomizedSearchCV with 25 iterations
TimeSeriesSplit (5 splits)
Scoring metric: R²
Best Model Performance
Generated optimal hyperparameters
Evaluated on test set:
R²: High predictive power
RMSE: Measures average prediction error in dollars
MAE: Mean absolute error in real price scale
MAPE: Percentage-based error metric
10. Ensemble Method
Weighted Ensemble
Combined three top models with optimized weights:

60% LightGBM (best individual model)
25% Gradient Boosting (strong runner-up)
15% Random Forest (diversification)
Result
Ensemble often outperforms individual models by reducing variance
11. Feature Importance and Explainability
Top Feature Importances
Identified 15 most important features for price prediction
Ranked by LightGBM's built-in feature importance
Feature Group Analysis
Features grouped into 6 categories:

Location Features: latitude, longitude, spatial interactions, zipcode target-encoding
Size Features: sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15
Structure Features: bedrooms, bathrooms, floors
Quality Features: condition, grade, view, waterfront
Age Features: house_age, house_age_renovated, is_renovated
Temporal Features: year, month, day, dayofweek, quarter
SHAP Explainability
Generated SHAP summary plots to visualize feature impact
Explained model predictions at both global and individual levels
Key Findings & Insights
Price Prediction Success: LightGBM achieved high R² on test set through careful feature engineering
Size is King: Square footage metrics are the strongest predictors
Quality Matters: Grade and condition exponentially impact price
Location Nuance: While latitude/longitude directly matter, cluster-based spatial analysis adds predictive value
Seasonality Negligible: Less than 7% price variation by season
Feature Engineering Critical: Transforming yr_built → house_age increased model performance
Time-Series Approach Essential: Respecting temporal order prevents model overfitting and ensures real-world applicability
Model Output
Final predictions available in log scale and real dollar scale
Model ready for production deployment with trained hyperparameters
Feature importance provides interpretability for stakeholders