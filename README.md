# Airbnb Price Prediction Model

A comprehensive machine learning project for predicting Airbnb listing prices in Chicago using ensemble methods and advanced feature engineering.

## Project Overview

This project implements a robust price prediction system for Airbnb listings using multiple machine learning algorithms including Random Forest, XGBoost, and CatBoost. The model achieved strong predictive performance through sophisticated feature engineering, hyperparameter optimization, and ensemble methods.

## 🚀 Key Features

- **Multiple Model Ensemble**: Combines Random Forest, XGBoost, and CatBoost for optimal predictions
- **Advanced Feature Engineering**: Polynomial features, target encoding, and temporal feature extraction
- **Outlier Detection & Handling**: Statistical methods to identify and adjust high-variance predictions
- **Hyperparameter Optimization**: Bayesian optimization using scikit-optimize
- **Cross-Validation**: Robust model evaluation with K-fold cross-validation

## Dataset

- **Training Data**: `train_regression.csv` - Historical Airbnb listings with price targets
- **Test Data**: `test_regression.csv` - Listings for price prediction
- **Features**: 
  - Property characteristics (bedrooms, bathrooms, accommodates)
  - Host information (response rate, superhost status, tenure)
  - Location data (neighborhood, property type)
  - Review metrics (number of reviews, ratings, recency)

## Technical Approach

### Data Preprocessing
- **Missing Value Imputation**: KNN imputation for ensemble models, mean/mode filling for boosting models
- **Feature Engineering**:
  - Temporal features from date columns (host_since, first_review, last_review)
  - Bathroom text parsing into numeric and categorical features
  - Property type and neighborhood categorization
  - Polynomial feature generation (degree 2)
- **Encoding**: Target encoding for categorical variables
- **Scaling**: StandardScaler for numerical features

### Models Used

1. **Random Forest Regressor**
   - Robust to outliers and missing values
   - Handles mixed data types well
   - Provides feature importance insights

2. **XGBoost Regressor**
   - Gradient boosting with regularization


3. **CatBoost Regressor**
   - Handles categorical features natively
   - Robust to overfitting
   - Excellent performance on structured data

### Ensemble Strategy
- **Weighted Average**: `(RF + CatBoost + 2×XGBoost) / 4`
- **Outlier Adjustment**: Manual tuning for high-variance predictions based on neighborhood characteristics

## Model Performance

The final ensemble model demonstrated strong predictive capability with:
- Comprehensive cross-validation evaluation
- RMSE optimization through Bayesian hyperparameter tuning
- Outlier detection and correction for improved accuracy

## Key Insights

### High-Price Property Characteristics
- **Location**: Loop, Near North Side, West Town neighborhoods
- **Capacity**: High accommodates (≥14 people) and beds (≥7-9)
- **Reviews**: Low or no reviews (potentially new listings)
- **Bathroom Type**: Private bathrooms preferred
- **Host Status**: Mix of superhosts and regular hosts

### Model-Specific Performance
- **XGBoost**: Better performance on lower-priced properties
- **Random Forest**: Robust across all price ranges
- **CatBoost**: Excellent handling of categorical features

## Project Structure

```
Airbnb_Price_Prediction/
├── Final_Model_And_Prep.ipynb    # Main analysis notebook
├── train_regression.csv          # Training dataset
├── test_regression.csv           # Test dataset  
├── README.md                     # Project documentation
└── regression_final.csv          # Final predictions (generated)
```

## Methodology

1. **Data Exploration**: Understanding feature distributions and relationships
2. **Feature Engineering**: Creating meaningful predictors from raw data
3. **Model Selection**: Comparing multiple algorithms
4. **Hyperparameter Tuning**: Optimizing model parameters
5. **Ensemble Creation**: Combining models for better performance
6. **Outlier Analysis**: Identifying and adjusting problematic predictions
7. **Final Validation**: Cross-validation and residual analysis

## Results

The final model successfully predicts Airbnb prices with:
- Ensemble approach combining three complementary algorithms
- Sophisticated feature engineering pipeline
- Outlier detection and manual adjustment for edge cases
- Robust cross-validation framework

## Notes

- The model uses different preprocessing pipelines for different algorithms to maximize diversity
- Manual outlier adjustment based on domain knowledge of Chicago neighborhoods
- Weighted ensemble gives more importance to XGBoost for lower-priced properties
- Cross-validation predictions used for residual analysis and model improvement



