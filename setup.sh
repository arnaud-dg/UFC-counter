#!/bin/bash

# Remove streamlit from requirements.txt to decrease the size of the bundle
# sed -i '/streamlit/d' requirements.txt

# Install the remaining dependencies
# pip install -r requirements_api.txt

# Install necessary system packages
# apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0