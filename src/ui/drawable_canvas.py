import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

def drawable_canvas(uploaded_image):
    """Creates a Streamlit drawable canvas with the uploaded image as the background."""

    st.write("Draw a line to set the scale:")

    background_image = Image.open(uploaded_image)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#e00",
        background_image=background_image,
        height=550,
        drawing_mode="line",
        key="canvas",
    )

    return canvas_result