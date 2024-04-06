from setuptools import setup, find_packages

setup(
    name='Grainsight',  
    version='0.1.0',  
    author='Your Name',  
    author_email='your_email@example.com',  
    description='A Streamlit app for segmenting grains using FastSAM',  
    packages=find_packages(where="src"),  
    install_requires=[
        'streamlit',
        'pillow',
        'ultralytics',
        'torch',
        'numpy',
        'opencv-python',
        'matplotlib',
        'pandas',
        'seaborn',
        'streamlit-drawable-canvas',
        'pyyaml'  
    ],
    entry_points={
        'console_scripts': [
            'grainsight = grainsight.app:main'
        ]
    }
)