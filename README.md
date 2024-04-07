# GrainSight: README

GrainSight is a user-friendly application designed for petrographers and microscopists to perform real-time grain segmentation and analysis on microscopic thin section images. Built on top of the powerful FastSAM segmentation model, GrainSight allows you to extract quantitative data and insights from your images, aiding in various petrographic studies.

## Importance in Petrographic Studies:

- **Automated Grain Segmentation**: GrainSight eliminates the need for manual grain boundary tracing, saving significant time and effort.
- **Quantitative Analysis**: Extract object-specific parameters such as area, perimeter, roundness, aspect ratio, and longest length, enabling quantitative analysis of grain characteristics.
- **Mineral Identification and Classification**: The extracted parameters can assist in mineral identification and classification based on their morphological properties.
- **Textural Analysis**: Grain size distribution and spatial arrangement of grains can be studied to understand the depositional and diagenetic history of rocks.

## Installation and Usage:

### 1. Create a Virtual Environment (Recommended):

It's recommended to use a virtual environment to manage project-specific dependencies. You can create one using venv or conda:

```bash
# Using venv
python3 -m venv grainsight_env

# Using conda
conda create -n grainsight_env python=3.8  # Replace 3.8 with your desired Python version
```

### 2. Activate the Virtual Environment:

```bash
# For venv
source grainsight_env/bin/activate  # On Linux/macOS
grainsight_env\Scripts\activate  # On Windows

# For conda
conda activate grainsight_env
```

### 3. Clone the GrainSight Repository:

```bash
git clone https://github.com/fazzam12345/grainsight.git
```     

### 4. Install Requirements:

Install the required libraries from the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 5. Run the Application:

Start the Streamlit application:

```bash
streamlit run app.py
```

This will open the GrainSight application in your web browser.

### Usage:

- **Upload an Image**: Select a microscopic thin section image in JPG, PNG, or JPEG format.
- **Set Parameters (Optional)**: Adjust segmentation parameters like input size, IOU threshold, and confidence threshold as needed.
- **Draw a Line for Scale**: Draw a line on the image and enter its real-world length (in micrometers) to set the scale for measurements.
- **Run Segmentation**: Click the "Run Segmentation" button to segment the image and extract grain parameters.
- **Analyze Results**: View the segmented image and the table of calculated grain parameters. You can also download the data as a CSV file.
- **Visualize Distributions**: Select a parameter to plot its distribution and gain further insights into grain characteristics.

### Additional Notes:

- **Dependencies**: Make sure you have the required versions of Python, PyTorch, Torchvision, and other libraries installed. Refer to the requirements.txt file for details.
- **GPU Acceleration**: For faster processing, you can use a CUDA-enabled GPU with the appropriate drivers and PyTorch version.
- **Customization**: The code is modular and can be easily extended or customized to suit your specific needs.

