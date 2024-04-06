import cv2
import numpy as np
import pandas as pd

def calculate_parameters(annotations, scale_factor):
    """Calculates parameters for each segmented object."""

    df = pd.DataFrame(columns=['Object', 'Area', 'Perimeter', 'Roundness', 'Aspect Ratio', 'Longest Length'])
    if len(annotations) > 0:
        for i, mask in enumerate(annotations):
            binary_mask = mask.cpu().numpy().astype(np.uint8)
            area_pixel = np.sum(binary_mask)
            area_micron = area_pixel * (scale_factor ** 2)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            perimeter_pixel = cv2.arcLength(contours[0], True)
            perimeter_micron = perimeter_pixel * scale_factor

            # Fit ellipse for roundness and aspect ratio
            if len(contours[0]) >= 5:
                ellipse = cv2.fitEllipse(contours[0])
                major_axis, minor_axis = ellipse[1]
            else:
                major_axis = minor_axis = 0
            
            major_axis_micron = major_axis * scale_factor
            minor_axis_micron = minor_axis * scale_factor
            roundness = 4 * area_micron / (np.pi * (major_axis_micron ** 2))
            aspect_ratio = major_axis_micron / minor_axis_micron if minor_axis_micron != 0 else "Undefined"
            longest_length_micron = major_axis_micron

            new_row = pd.DataFrame({                
                'Object': [f"Object {i+1}"],                
                'Area': [area_micron],                
                'Perimeter': [perimeter_micron],                
                'Roundness': [roundness],                
                'Aspect Ratio': [aspect_ratio],                
                'Longest Length': [longest_length_micron]            
            })
            df = pd.concat([df, new_row], ignore_index=True)

    return df