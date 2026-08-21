
import os
import numpy as np
import streamlit as st
import tensorflow as tf
import pydicom

from PIL import Image


# ============================================================
# Application Configuration
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="centered"
)

MODEL_PATH = "pneumonia_best_model.keras"
IMG_HEIGHT = 224
IMG_WIDTH = 224


# ============================================================
# Load Trained Model
# ============================================================

@st.cache_resource
def load_pneumonia_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_pneumonia_model()


# ============================================================
# Image Preprocessing
# ============================================================

def prepare_array_for_model(image):

    image = image.astype(np.float32)

    # Min-max scaling
    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image)

    # Convert to 0-255
    image = (image * 255).astype(np.uint8)

    # Resize
    image = Image.fromarray(image).resize(
        (IMG_WIDTH, IMG_HEIGHT),
        Image.Resampling.BILINEAR
    )

    image = np.array(image, dtype=np.float32)

    # Replicate grayscale channel to RGB-compatible 3 channels
    image_3channel = np.repeat(
        image[..., np.newaxis],
        3,
        axis=-1
    )

    model_input = np.expand_dims(
        image_3channel,
        axis=0
    )

    return image, model_input


def preprocess_uploaded_file(uploaded_file):

    file_extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    # DICOM image
    if file_extension == ".dcm":

        uploaded_file.seek(0)
        dcm = pydicom.dcmread(uploaded_file)

        image = dcm.pixel_array.astype(np.float32)

        if getattr(
            dcm,
            "PhotometricInterpretation",
            ""
        ) == "MONOCHROME1":
            image = np.max(image) - image

        if image.ndim == 3:
            image = np.mean(image, axis=-1)

    # PNG / JPG / JPEG
    else:

        uploaded_file.seek(0)

        image = Image.open(
            uploaded_file
        ).convert("L")

        image = np.array(
            image,
            dtype=np.float32
        )

    return prepare_array_for_model(image)


# ============================================================
# Streamlit User Interface
# ============================================================

st.title("🩻 Pneumonia Detection System")

st.write(
    """
    Upload a chest X-ray image to obtain the predicted class
    and estimated pneumonia probability using the selected
    Enhanced MobileNetV2 deep learning model.
    """
)

st.info(
    "Supported formats: DICOM (.dcm), PNG, JPG and JPEG."
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["dcm", "png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    try:

        display_image, model_input = preprocess_uploaded_file(
            uploaded_file
        )

        st.image(
            display_image,
            caption="Uploaded Chest X-ray",
            clamp=True
        )

        if st.button("Predict"):

            probability = float(
                model.predict(
                    model_input,
                    verbose=0
                )[0][0]
            )

            predicted_class = (
                "Pneumonia Positive"
                if probability >= 0.5
                else "Pneumonia Negative"
            )

            st.subheader("Prediction Result")

            if probability >= 0.5:
                st.error(
                    f"Predicted Class: {predicted_class}"
                )
            else:
                st.success(
                    f"Predicted Class: {predicted_class}"
                )

            st.metric(
                "Pneumonia Probability",
                f"{probability:.2%}"
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

    except Exception as error:

        st.error(
            f"Unable to process the uploaded image: {error}"
        )


st.divider()

st.caption(
    "Academic decision-support prototype only. "
    "The prediction is not a medical diagnosis and should not "
    "replace assessment by a qualified healthcare professional."
)
