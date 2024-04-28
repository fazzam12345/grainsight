import math

def calculate_pixel_length(start_point, end_point):
    """Calculates the pixel length of a line."""
    x1, y1 = start_point
    x2, y2 = end_point
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)