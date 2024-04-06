import streamlit as st

def streamlit_ui():
    """Creates the Streamlit user interface with input controls."""

    st.sidebar.title("Segmentation Parameters")

    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    input_size = st.sidebar.slider(
        "Input Size", 512, 1024, 1024, 64,
        help="Size of the input image. Higher values may improve detection but will be slower."
    )

    iou_threshold = st.sidebar.slider(
        "IOU Threshold", 0.0, 0.9, 0.7, 0.1, 
        help="Intersection over Union threshold for object detection. Higher values reduce false positives."
    )

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 0.9, 0.5, 0.05, 
        help="Minimum confidence level for detected objects. Lower values may detect more objects but increase false positives."
    )

    better_quality = st.sidebar.checkbox(
        "Better Visual Quality", True, 
        help="Check to improve the visual quality of the segmentation. May be slower."
    )

    contour_thickness = st.sidebar.slider(
        "Contour Thickness", 1, 50, 1,
        help="Thickness of the contour lines around detected objects."
    )

    real_world_length = st.sidebar.number_input(
        "Enter the real-world length of the line in micrometers:", 
        min_value=1, value=100,
        help="Length of the reference line in the real world, used for scaling object parameters."
    )

    return uploaded_image, input_size, iou_threshold, conf_threshold, better_quality, contour_thickness, real_world_length