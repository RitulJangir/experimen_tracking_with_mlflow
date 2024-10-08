import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import category_encoders as ce

# Load dataset
df = pd.read_csv('./data/hour.csv')
df['day_night'] = df['hr'].apply(lambda x: 'day' if 6 <= x <= 18 else 'night')
df.drop(['instant', 'casual', 'registered'], axis=1, inplace=True)
df['dteday'] = pd.to_datetime(df.dteday)
df['season'] = df.season.astype('category')
df['holiday'] = df.holiday.astype('category')
df['weekday'] = df.weekday.astype('category')
df['weathersit'] = df.weathersit.astype('category')
df['workingday'] = df.workingday.astype('category')
df['mnth'] = df.mnth.astype('category')
df['yr'] = df.yr.astype('category')
df['hr'] = df.hr.astype('category')
df.drop(columns=['dteday'], inplace=True)

# Separating features and target variable
X = df.drop(columns=['cnt'])  # Features
y = df['cnt']  # Target

# Numerical features preprocessing
numerical_features = ['temp', 'hum', 'windspeed']
numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', MinMaxScaler())])
X[numerical_features] = numerical_pipeline.fit_transform(X[numerical_features])

# Categorical features preprocessing
categorical_features = ['season', 'weathersit', 'day_night']
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, drop='first'))
])
X_encoded = categorical_pipeline.fit_transform(X[categorical_features])
X_encoded = pd.DataFrame(X_encoded, columns=categorical_pipeline.named_steps['onehot'].get_feature_names_out(categorical_features))

# Target Encoding for Linear Regression
categorical_pipeline_LR = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_encoder', ce.TargetEncoder())
])
X_encoded_LR = categorical_pipeline_LR.fit_transform(X[categorical_features], y)

# Combine preprocessed features for RandomForest and LinearRegression
X_RF = pd.concat([X.drop(columns=categorical_features), X_encoded], axis=1)
X_LR = pd.concat([X.drop(columns=categorical_features), X_encoded_LR], axis=1)
X_LR.columns = X_LR.columns.astype(str)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_RF, y, test_size=0.2, random_state=42)
LR_X_train, LR_X_test, LR_y_train, LR_y_test = train_test_split(X_LR, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Bike Sharing Prediction")

# Random Forest Model
with mlflow.start_run(run_name="RandomForest_Model"):
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    # Log model, parameters, and metrics
    mlflow.log_param("model_type", "randomforest")

    y_pred_rf = model_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    mlflow.log_metric("mse", mse_rf)
    mlflow.log_metric("r2", r2_rf)
    mlflow.sklearn.log_model(model_rf, "random_forest_model")

    print(f'RandomForest Mean Squared Error: {mse_rf}')
    print(f'RandomForest R-squared: {r2_rf}')

# Linear Regression Model
with mlflow.start_run(run_name="LinearRegression_Model"):
    model_lr = LinearRegression()
    model_lr.fit(LR_X_train, LR_y_train)


    # Log model, parameters, and metrics
    mlflow.log_param("model_type", "LinearRegression")
    y_pred_lr = model_lr.predict(LR_X_test)
    mse_lr = mean_squared_error(LR_y_test, y_pred_lr)
    r2_lr = r2_score(LR_y_test, y_pred_lr)

    mlflow.log_metric("mse", mse_lr)
    mlflow.log_metric("r2", r2_lr)
    mlflow.sklearn.log_model(model_lr, "linear_regression_model")

    print(f'LinearRegression Mean Squared Error: {mse_lr}')
    print(f'LinearRegression R-squared: {r2_lr}')
