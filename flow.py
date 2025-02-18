import mlflow
import tensorflow as tf


model = tf.keras.models.load_model("F:\machine learning\Rice_leaf_Disease\Disease_dtct (1).h5")  # Make sure the file exists

mlflow.set_experiment("Rice Disease Detection")

model_accuracy = 0.95  

with mlflow.start_run():

    mlflow.log_param("model_type", "DenseNet-121")
    
    mlflow.log_metric("accuracy", model_accuracy)
    
    mlflow.tensorflow.log_model(model, artifact_path="model")
