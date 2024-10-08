import mlflow

# Set the tracking URI to a folder where you have permissions (e.g., /tmp/mlruns)
mlflow.set_tracking_uri("file:///tmp/mlruns")

mlflow.set_experiment("Bike Sharing Prediction")

