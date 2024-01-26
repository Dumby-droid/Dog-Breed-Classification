import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('Downloads/Dog Breed Classification.hdf5', compile=False)

st.title("Dog Breed Classifier")

uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

map_dict = {0: 'Beagle',
            1: 'Bernese Mountain Dog',
            2: 'Chohuahua',
            3: 'Corgi',
            4: 'Dalmatian',
            5: 'Doberman',
            6: 'German Shepherd',
            7: 'Golden Retriver',
            8: 'Pomeranian',
            9: 'Poodle',
            10: 'Pug',
            11: 'Rottweiler',
            12: 'Siberian Husky'}

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    pil_image = Image.open(uploaded_file)

    if st.button("Predict Dog Breed"):
        predictions = model.predict(img_array).argmax()

        st.subheader("Prediction:")
        st.write("The predicted dog breed is: {}".format(map_dict [predictions]))