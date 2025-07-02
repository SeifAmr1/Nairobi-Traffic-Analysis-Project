

# Nairobi Traffic Bus Ticket Demand Forecasting

This project aims to **predict the number of tickets sold per ride** using a dataset of historical bus rides. The workflow involves **feature engineering**, **time series forecasting (Prophet)**, and **supervised machine learning models** (Random Forest and XGBoost) for accurate ticket count predictions.

---


## Features and Preprocessing

* **Datetime Features**: Extracted weekday, hour, and day-of-year from `travel_date` and `travel_time`.
* **Aggregates**: Total tickets per `ride_id`.
* **Custom Features**:

  * `rush_hour`: Binary feature for high-demand hours.
  * `demand_level_encoded`: Demand level encoding based on hour.
  * PCA-reduced feature from `rush_hour` and `demand_level_encoded`.
* **Categorical Encoding**:

  * `travel_from` encoded using LabelEncoder.
  * `car_type` mapped to 0 (shuttle) and 1 (bus).

---

## Time Series Forecasting with Prophet

Used Facebook Prophet to:

* Forecast daily total ticket sales.
* Plot predictions and forecast components.
* Analyze trends and seasonality.

---

## Machine Learning Models

### Random Forest Regressor

* Used engineered features to predict `number_of_ticket`.
* Evaluated using:

  * Mean Absolute Error (MAE)
  * R² Score
  * Cross-validation MAE

### XGBoost Regressor

* Boosted model for better performance tuning.
* Also evaluated using MAE and R² Score.

---

## Model Evaluation & Visualization

* **Residual Plot**: To assess prediction bias.
* **Histogram**: Distribution of predicted ticket counts.
* **Feature Importance**: From Random Forest.
* **Correlation Analysis**:

  * Bar chart of feature-target correlations.
  * Heatmap of all feature correlations.

---

## Output

* `submission.csv`: Ride IDs with predicted ticket counts (rounded), You can review the file in colab.
* Clear insights into ticket demand patterns across time.

---

## Requirements

* Python (3.8+)
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* xgboost
* prophet


