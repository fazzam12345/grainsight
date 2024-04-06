import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import streamlit as st


def segment_everything(_input, model, device, input_size=1024, iou_threshold=0.7, conf_threshold=0.25, better_quality=False, contour_thickness=1):
    """
    Performs segmentation on the input image and returns the segmented image and annotations.
    """
    
    input_image = _input
    input_size = int(input_size)
    w, h = input_image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input_image = input_image.resize((new_w, new_h))

    results = model(input_image,                    
                   retina_masks=True,                    
                   iou=iou_threshold,                    
                   conf=conf_threshold,                    
                   imgsz=input_size)

    annotations = results[0].masks.data
    segmented_image = fast_process(annotations=annotations,                       
                       device=device,                       
                       image=input_image,                       
                       scale=(1024 // input_size),                       
                       better_quality=better_quality,                       
                       contour_thickness=contour_thickness)

    return segmented_image, annotations


def fast_process(annotations,
                  image,
                  device,
                  scale,
                  better_quality=False,
                  mask_random_color=True,
                  bbox=None,
                  use_retina=True,
                  withContours=True,
                  contour_thickness=2):
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
            kernel = np.ones((5, 5), np.uint8)
            annotation = cv2.morphologyEx(annotation, cv2.MORPH_OPEN, kernel)
            annotation = cv2.GaussianBlur(annotation, (5, 5), 0)  # Gaussian blur
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
    # Sort annotation by area
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
    # Find the first non-zero value index for each position
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
    # Use vectorization to get the value of the batch
    mask = torch.zeros((height, weight, 4)).to(device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    # Use vectorized indexing to update the show values
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


