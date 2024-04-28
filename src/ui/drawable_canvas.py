# src\ui\drawable_canvas.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

def drawable_canvas(uploaded_image, input_size):
    """Creates a Streamlit drawable canvas with the resized image as the background."""
    # Generate a unique key for the canvas based on the input size
    canvas_key = f"canvas_{input_size}"

    st.write("Draw a line to set the scale:")
    original_image = Image.open(uploaded_image)
    image_width, image_height = original_image.size
    scale = input_size / max(image_width, image_height)
    new_w = int(image_width * scale)
    new_h = int(image_height * scale)
    resized_image = original_image.resize((new_w, new_h))
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#e00",
        background_image=resized_image,
        height=new_h,
        width=new_w,
        drawing_mode="line",
        key=canvas_key,
    )
    return canvas_result

