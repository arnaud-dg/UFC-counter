# SMART Gelose Counter

## Introduction
The SMART Gelose Counter is a machine learning-based application designed to automate the counting of Colony Forming Units (CFUs) in bacterial cultures. This application leverages a custom-trained YOLOv5 model to detect and count bacterial colonies from images, providing a significant improvement in speed and accuracy over manual counting methods. The project is particularly useful in fields like microbiology, food safety, and vaccine development where CFU counting is a common task.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features
- **Automatic CFU Counting**: Utilizes a custom YOLOv5 model to detect and count CFUs in images.
- **Web Interface**: Provides a user-friendly interface built with Streamlit for uploading images and viewing results.
- **Customizable Predictions**: Options to display prediction probabilities and use shutter view for detailed analysis.
- **Support for Multiple Image Formats**: Compatible with JPG, PNG, and JPEG image formats.
- **Platform Independence**: Designed to work seamlessly on different operating systems.

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/smart-gelose-counter.git
    cd smart-gelose-counter
    ```

2. **Set up a virtual environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install -r api/requirements.txt
    ```

4. **To start the web application**:
    ```bash
    streamlit run app.py
    ```

## API Usage

The application also provides an API for programmatic access:
- GET /: Health check endpoint.
- POST /predict/: Upload an image to receive CFU predictions.

## Dependencies
- Python 3.8+
- FastAPI: For building the API.
- Streamlit: For the web interface.
- PIL (Pillow): For image processing.
- Torch: For loading and running the YOLOv5 model.
- Pandas: For handling data structures and processing results.

## Configuration
- Model Path: Ensure the YOLOv5 model is located in the models directory.
- API Endpoint: Configure the API endpoint in the Streamlit app if running the API on a different server (local VS online).

## Documentation
- The code is well-commented to help understand the flow and purpose of each function.
- Several publicly accessible research papers are available in the assets/docs/ repository
- Refer to the original research paper: "Annotated dataset for deep-learning-based bacterial colony detection" for more insights into the data and model used.

## Troubleshooting
- Model Not Found Error: Ensure the model file is in the correct path.
- API Request Failures: Check if the API server is running and accessible.

## License
This project is licensed under the MIT License - see the LICENSE file for details.