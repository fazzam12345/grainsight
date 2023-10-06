import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from streamlit_drawable_canvas import st_canvas
from tools import format_results, box_prompt, point_prompt, text_prompt




device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
model = YOLO('C:\\Users\\Fares\\Zircon detection and morphology\\FastSAM\\weights\\FastSAM-x.pt')

annotations = []


def streamlit_ui():
    st.title("Segment grains using Fast SAM 🤗")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    input_size = st.slider("Input Size", 512, 1024, 1024, 64, 
                           help="Size of the input image. Higher values may improve detection but will be slower.")
    
    iou_threshold = st.slider("IOU Threshold", 0.0, 0.9, 0.7, 0.1, 
                              help="Intersection over Union threshold for object detection. Higher values reduce false positives.")
    
    conf_threshold = st.slider("Confidence Threshold", 0.0, 0.9, 0.5, 0.05, 
                               help="Minimum confidence level for detected objects. Lower values may detect more objects but increase false positives.")
    
    better_quality = st.checkbox("Better Visual Quality", True, 
                                 help="Check to improve the visual quality of the segmentation. May be slower.")
    
    contour_thickness = st.slider("Contour Thickness", 1, 50, 1, 
                                  help="Thickness of the contour lines around detected objects.")
    
    real_world_length = st.number_input("Enter the real-world length of the line in micrometers:", min_value=1, value=100, 
                                       help="Length of the reference line in the real world, used for scaling object parameters.")
    return uploaded_image, input_size, iou_threshold, conf_threshold, better_quality, contour_thickness, real_world_length

def calculate_pixel_length(start_point, end_point):
    x1, y1 = start_point
    x2, y2 = end_point
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def drawable_canvas(uploaded_image):
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


def fast_process(annotations,
    image,
    device,
    scale,
    better_quality=False,
    mask_random_color=True,
    bbox=None,
    use_retina=True,
    withContours=True,
    contour_thickness=2
):
    if isinstance(annotations[0], dict):
        annotations = [annotation['segmentation'] for annotation in annotations]

    original_h = image.height
    original_w = image.width
    if better_quality:
        if isinstance(annotations[0], torch.Tensor):
            annotations = np.array(annotations.cpu())
        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
    if device == 'cpu':
        annotations = np.array(annotations)
        inner_mask = fast_show_mask(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bbox=bbox,
            retinamask=use_retina,
            target_height=original_h,
            target_width=original_w,
        )
    else:
        if isinstance(annotations[0], np.ndarray):
            annotations = torch.from_numpy(annotations)
        inner_mask = fast_show_mask_gpu(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bbox=bbox,
            retinamask=use_retina,
            target_height=original_h,
            target_width=original_w,
        )
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()

    kernel = np.ones((5, 5), np.uint8)
    
    if withContours:
        contour_all = []
        temp = np.zeros((original_h, original_w, 1))
        for i, mask in enumerate(annotations):
            if type(mask) == dict:
                mask = mask['segmentation']

            # Convert to uint8
            annotation = mask.astype(np.uint8)

            # Perform morphological operations for separating connected objects and smoothing contours
            kernel = np.ones((5,5),np.uint8)
            annotation = cv2.morphologyEx(annotation, cv2.MORPH_OPEN, kernel)

            annotation = cv2.GaussianBlur(annotation, (5, 5), 0) # Gaussian blur

            # Find contours
            contours, _ = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Add contour approximation here
            for contour in contours:
                hull = cv2.convexHull(contour)
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                contour_all.append(approx)  # Append the approximated contour

        # New code to display object indices
        for i, contour in enumerate(contour_all):
            # Calculate the centroid of the object to place the index text
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Put the index at the centroid of the object
            cv2.putText(temp, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 125, 255), 2)

        # Increase contour thickness (you can also make this a variable)
        cv2.drawContours(temp, contour_all, -1, (255, 255, 255), contour_thickness)

        # Change the color here (this example sets it to red)
        color = np.array([255 / 255, 0 / 255, 0 / 255, 1])  # RGBA

        contour_mask = temp / 255 * color.reshape(1, 1, -1)


    image = image.convert('RGBA')
    overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), 'RGBA')
    image.paste(overlay_inner, (0, 0), overlay_inner)

    if withContours:  # Make sure contour_mask is defined when this block is executed
        overlay_contour = Image.fromarray((contour_mask * 255).astype(np.uint8), 'RGBA')
        image.paste(overlay_contour, (0, 0), overlay_contour)

    return image


# CPU post process
def fast_show_mask(
    annotation,
    ax,
    random_color=False,
    bbox=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    # 将annotation 按照面积 排序
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::1]
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    if random_color:
        color = np.random.random((mask_sum, 1, 1, 3))
    else:
        color = np.ones((mask_sum, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 255 / 255])
    transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    mask = np.zeros((height, weight, 4))

    h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

    mask[h_indices, w_indices, :] = mask_image[indices]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))

    if not retinamask:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    return mask


def fast_show_mask_gpu(
    annotation,
    ax,
    random_color=False,
    bbox=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    device = annotation.device
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    # 找每个位置第一个非零值下标 (translate to english: find the first non-zero value index for each position)
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    if random_color:
        color = torch.rand((mask_sum, 1, 1, 3)).to(device)
    else:
        color = torch.ones((mask_sum, 1, 1, 3)).to(device) * torch.tensor(
            [30 / 255, 144 / 255, 255 / 255]
        ).to(device)
    transparency = torch.ones((mask_sum, 1, 1, 1)).to(device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    # 按index取数，index指每个位置选哪个batch的数，把mask_image转成一个batch的形式(translate to english: use vectorization to get the value of the batch)
    mask = torch.zeros((height, weight, 4)).to(device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    # 使用向量化索引更新show的值
    mask[h_indices, w_indices, :] = mask_image[indices]
    mask_cpu = mask.cpu().numpy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="b", linewidth=1
            )
        )
    if not retinamask:
        mask_cpu = cv2.resize(
            mask_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
    return mask_cpu


# segment_everything function
def segment_everything(_input, input_size=1024, iou_threshold=0.7, conf_threshold=0.25, better_quality=False, contour_thickness=1):
    input = _input
    input_size = int(input_size)
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    results = model(input,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size)

    annotations = results[0].masks.data

    fig = fast_process(annotations=annotations,device=device,
                       image=input,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       contour_thickness=contour_thickness)  
    return fig,annotations

def calculate_parameters(annotations, scale_factor):
    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['Object', 'Area', 'Perimeter', 'Roundness', 'Aspect Ratio', 'Longest Length'])
    
    if len(annotations) > 0:  # Check if annotations list is not empty
        for i, mask in enumerate(annotations):
            # Convert mask to binary image
            binary_mask = mask.cpu().numpy().astype(np.uint8)
            
            # Calculate area in pixels
            area_pixel = np.sum(binary_mask)
            
            # Convert area to microns
            area_micron = area_pixel * (scale_factor ** 2)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate Perimeter in pixels
            perimeter_pixel = cv2.arcLength(contours[0], True)
            
            # Convert perimeter to microns
            perimeter_micron = perimeter_pixel * scale_factor
            
            # Fit an ellipse to the object
            if len(contours[0]) >= 5:  # Check if there are enough points to fit an ellipse
                ellipse = cv2.fitEllipse(contours[0])
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
            else:
                major_axis = minor_axis = 0  # Default values if not enough points

            # Convert major and minor axis to microns
            major_axis_micron = major_axis * scale_factor
            minor_axis_micron = minor_axis * scale_factor
            
            # Calculate Roundness
            roundness = 4 * area_micron / (np.pi * (major_axis_micron ** 2))
            
            # Calculate Aspect Ratio
            if minor_axis_micron != 0:  # Check to avoid division by zero
                aspect_ratio = major_axis_micron / minor_axis_micron
            else:
                aspect_ratio = "Undefined due to zero minor axis"

            # Longest Length (Major Axis)
            longest_length_micron = major_axis_micron
            
            # Add to DataFrame
            new_row = pd.DataFrame({
                'Object': [f"Object {i+1}"],
                'Area': [area_micron],
                'Perimeter': [perimeter_micron],
                'Roundness': [roundness],
                'Aspect Ratio': [aspect_ratio],
                'Longest Length': [longest_length_micron]
            })
            
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Display in Streamlit
            #st.write(f"Object {i+1}: Area = {area_micron:.2f} µm², Perimeter = {perimeter_micron:.2f} µm, Roundness = {roundness:.2f}, Aspect Ratio = {aspect_ratio}, Longest Length = {longest_length_micron:.2f} µm")
    
    return df

# Function to plot distribution
def plot_distribution(df, selected_parameter):
    try:
        fig, ax = plt.subplots()
        sns.histplot(df[selected_parameter], kde=True, ax=ax)
        ax.set_title(f'Distribution of {selected_parameter}')
        ax.set_xlabel(selected_parameter)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    except Exception as e:
        st.write(f"An error occurred while plotting: {e}")


def main():
    uploaded_image, input_size, iou_threshold, conf_threshold, better_quality, contour_thickness, real_world_length = streamlit_ui()
    if uploaded_image is not None:
        canvas_result = drawable_canvas(uploaded_image)
        pixel_length = None  # Initialize pixel_length

        # Check if line is drawn on canvas and pixel_length is not None
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
            scale_factor = real_world_length / pixel_length  # Calculate scale_factor
        else:
            st.write("Scale factor could not be calculated. Make sure to draw a line and enter the real-world length.")
            return  # Exit the function if scale_factor can't be calculated
        
        input_image = Image.open(uploaded_image)
        segmented_image, annotations = segment_everything(
        input_image, 
        input_size=input_size, 
        iou_threshold=iou_threshold, 
        conf_threshold=conf_threshold, 
        better_quality=better_quality,
        contour_thickness=contour_thickness)
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)

        # Call your new function
        df = calculate_parameters(annotations, scale_factor)
        
        # Display DataFrame in Streamlit
        if not df.empty:
            st.write("Summary of Object Parameters:")
            st.dataframe(df)
            
            # Let the user select the parameter to plot
            filtered_columns = [col for col in df.columns.tolist() if col != 'Object']
            selected_parameter = st.selectbox("Select a parameter to see its distribution:", filtered_columns)            
            # Check if selected_parameter is defined
            if selected_parameter:
                # Plot the distribution of the selected parameter
                plot_distribution(df, selected_parameter)
            else:
                st.write("No parameter selected for plotting.")
        else:
            st.write("No objects detected.")
    else:
        st.write("Please upload an image.")

if __name__ == "__main__":
    main()



    