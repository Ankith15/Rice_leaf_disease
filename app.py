
import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv
load_dotenv()

fun = os.getenv('anki')
print(fun)

@st.cache_resource
def load_trained_model():
    model = load_model(r'F:\\machine learning\\Rice_leaf_Disease\Disease_dtct (1).h5')
    return model


# fun_id = f"runs:/{fun}/model"

model =load_model(r'https://github.com/Ankith15/Rice_leaf_disease/blob/main/Disease_dtct%20(1).h5')


class_labels = ["Rice_Blast","Bacterial_leaf_blight", "Healthy_leaf", "Tungro"]

st.title(" Rice Plant Disease Detection")
st.write("upload rice plant leaf image")


upload_file = st.file_uploader("choose image from Device",type = ['jpg', 'png', 'jpeg', 'bmp'])


if upload_file is not None:

    image_data = Image.open(upload_file)
    st.image(image_data,caption= "uploaded Image",use_column_width = True)

    def preprocess_image(img):
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr,axis = 0)
        img_arr = img_arr/255.0
        return img_arr

    processed_image = preprocess_image(image_data)

    

    predictions = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]

    st.write(f"Predicted class: {predicted_class}")
    st.write(f"confidence : {np.max(predictions)*100:.2f}%")
