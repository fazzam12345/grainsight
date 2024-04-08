import streamlit as st
import logging
from PIL import Image
from src.ui import streamlit_ui, drawable_canvas
from src.segmentation import segment_everything
from src.utils import calculate_parameters, plot_distribution, calculate_pixel_length
from ultralytics import YOLO
import torch
import cv2

logging.basicConfig(filename="grainsight.log", level=logging.INFO)

model_path= "src\\model\\FastSAM-x.pt"  
model_path= "src\\model\\FastSAM-x.pt"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(model_path)


def main():
    """Main application logic."""

    # Get user inputs and set defaults from config
    uploaded_image, input_size, iou_threshold, conf_threshold, better_quality, contour_thickness, real_world_length = streamlit_ui()

    if uploaded_image is not None:
        try:
            canvas_result = drawable_canvas(uploaded_image)
            pixel_length = None

            if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                if len(canvas_result.json_data["objects"]) > 0:
                    line_object = canvas_result.json_data["objects"][0]
                    start_point = [line_object['x1'], line_object['y1']]
                    end_point = [line_object['x2'], line_object['y2']]
                    pixel_length = calculate_pixel_length(start_point, end_point)
                    st.write(f"Pixel length of the line: {pixel_length}")
                else:
                    st.write("Please draw a line to set the scale or enter the real-world length.")
            else:
                st.write("Please draw a line to set the scale or enter the real-world length.")

            if pixel_length is not None and real_world_length is not None:
                scale_factor = real_world_length / pixel_length 
            else:
                st.write("Scale factor could not be calculated. Make sure to draw a line and enter the real-world length.")
                return  

            input_image = Image.open(uploaded_image)
            segmented_image, annotations = segment_everything(
                input_image,
                model=model,
                device=device,
                input_size=input_size,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold,
                better_quality=better_quality,
                contour_thickness=contour_thickness
            )

            st.image(segmented_image, caption="Segmented Image", use_column_width=True)

            # Calculate and display object parameters
            df = calculate_parameters(annotations, scale_factor)
            if not df.empty:
                st.write("Summary of Object Parameters:")
                st.dataframe(df)

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='grain_parameters.csv',
                    mime='text/csv',
                )

                # Parameter distribution plot
                filtered_columns = [col for col in df.columns.tolist() if col != 'Object']
                selected_parameter = st.selectbox("Select a parameter to see its distribution:", filtered_columns)
                if selected_parameter:
                    plot_distribution(df, selected_parameter)
                else:
                    st.write("No parameter selected for plotting.")
            else:
                st.write("No objects detected.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            st.error("An error occurred during processing. Please check the logs for details.")
    else:
        st.write("Please upload an image.")

if __name__ == "__main__":
    main()
    