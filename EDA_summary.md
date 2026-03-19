EDA Summary (from the notebook)
1) Data Loading & Inspection
Loaded king_ country_ houses_aa.csv into df.
Inspected:
df.head(), df.describe(), df.dtypes, df.info().
Missing values via df.isna().sum().
Unique counts via df.nunique().
2) Initial Cleaning / Feature Setup
Defined categorical vs numeric feature lists (cat_features, num_features).

cat_features = [
    "waterfront",
    "view",
    "zipcode",
    "condition",
    "grade",
    "floors",
    "bathrooms",
    "bedrooms",
]

num_features = [
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_basement",
    "sqft_living15",
    "sqft_lot15",
    "yr_built",
    "yr_renovated",
]

Converted date column to datetime (UTC).
Removed extreme outliers:
Filtered out houses with bedrooms >= 10 into df_cleaned.
3) Target Distribution & Outliers
Price distribution (sns.histplot):
Right-skewed; recommended log transform for modeling.
Bedroom/Bathroom outliers:
One house with 30+ bedrooms (removed). Might be an typing error bedrooms=3?
Bathrooms included fractional values (e.g., 0.25), likely representing half-baths or partial baths.
Basement size:
Some extreme values (>3000 sqft) flagged as potential outliers.
4) Seasonality Analysis
Created season from sale date (Winter/Spring/Summer/Autumn).
Computed average price per season and plotted bar chart.
Found only a "small" seasonal effect:
Spring vs Winter price difference ≈ 6.44%. Take into consideration.
5) Correlation & Feature Insights
Correlation heatmap over numeric features.
Key findings (noted in notebook comments):
Strongest predictors: sqft_living (≈0.70), grade (≈0.67), bathrooms (≈0.53), view (≈0.40).
yr_built and yr_renovated have low correlation, suggesting transformation into:
age = 2026 - yr_built
is_renovated = yr_renovated > 0
6) Spatial / Geographical Exploration
Calculated distance to a city center point (dist_to_center) using lat/long.
Used KMeans (70 clusters) on lat/lon to create neighborhood_group clusters (“geo clusters”).
Visualized clusters with scatterplot and with a geographic basemap using GeoPandas + Contextily.
7) Zipcode vs Geo-Cluster Comparison
Compared geographic clustering vs original zipcode feature:
Computed purity of clusters within zipcodes (mean purity ~0.72).
Built two linear models (one using zipcode dummies (1-hot-encoding), one using cluster dummies).
Reported R² and RMSE for each model (printed in a results DataFrame).
✅ Overall Takeaways
The dataset has strong spatial structure (clusters + zipcode), but zipcode alone doesn’t fully capture geographic variation.
Price is skewed, so logs or robust modeling could help.
Seasonal effect appears minimal (~6%).
Top predictors are size/quality-related (sqft_living, grade, bathrooms, view).

1-hot-encoding
zipcode

renovated houses
4,22%
913
21608

Bathrooms are related with the bedrooms (no more info from dataframe). We keep it like this.

DataFrame legend
