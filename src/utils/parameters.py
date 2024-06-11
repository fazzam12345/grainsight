import cv2
import numpy as np
import pandas as pd
import streamlit as st

def calculate_parameters(annotations, scale_factor):
    """Calculates parameters for each segmented object, including Feret diameter."""
    df = pd.DataFrame(columns=['Object', 'Area', 'Perimeter', 'Roundness', 
                                'Aspect Ratio (Elongation)', 'Longest Feret Diameter'])
    if len(annotations) > 0:
        for i, mask in enumerate(annotations):
            binary_mask = mask.cpu().numpy().astype(np.uint8)
            area_pixel = np.sum(binary_mask)
            area_micron = area_pixel * (scale_factor ** 2)

            # Find contours with all points (no approximation)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if contours:  
                perimeter_pixel = cv2.arcLength(contours[0], True)
                perimeter_micron = perimeter_pixel * scale_factor

                # Fit ellipse for roundness and aspect ratio (check for sufficient points)
                if len(contours[0]) >= 5:
                    ellipse = cv2.fitEllipse(contours[0])
                    major_axis, minor_axis = ellipse[1]
                else:
                    major_axis = minor_axis = 0

                major_axis_micron = major_axis * scale_factor
                minor_axis_micron = minor_axis * scale_factor
                roundness = (4 * np.pi * area_micron) / (perimeter_micron ** 2)
                aspect_ratio = major_axis_micron / minor_axis_micron if minor_axis_micron != 0 else "Undefined"

                # Calculate Feret diameter and Elongation
                hull = cv2.convexHull(contours[0])
                distances = np.linalg.norm(hull - hull[:, 0, :], axis=2)
                max_feret_diameter_micron = np.max(distances) * scale_factor

                new_row = pd.DataFrame({
                    'Object': [f"Object {i}"],
                    'Area': [area_micron],
                    'Perimeter': [perimeter_micron],
                    'Roundness': [roundness],
                    'Aspect Ratio (Elongation)': [aspect_ratio],
                    'Longest Feret Diameter': [max_feret_diameter_micron],
                })
                df = pd.concat([df, new_row], ignore_index=True)

        # Eliminate artifacts with undefined parameters
        df = df[(df['Longest Feret Diameter'] != 0) & (df['Roundness'] >= 0) & (df['Roundness'] <= 1)]

    return df