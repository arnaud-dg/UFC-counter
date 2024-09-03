import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pathlib import Path
import os
import io
from io import StringIO, BytesIO
import requests
from helper import load_image, add_transparent_mask, combine_images, draw_rectangles

# Set up the Streamlit page configuration
st.set_page_config(page_title="SMART Gelose Counter", page_icon="👀", layout="wide")

# Hide the slider numbers with custom CSS
st.markdown("""
    <style>
    div[data-testid="stTickBar"] {
        display: none;
    }
    div[data-testid="stThumbValue"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Define a color palette for the app's theme
COLOR_PALETTE = {
    'grey': '#94979C',
    'bordeaux': '#6A0910',
    'cream': '#E3DCD4',
    'red': '#9A0E1B'
}

# Get the path of the current script directory and the associated paths
APP_DIRECTORY = Path(__file__).resolve().parent
LOGO_PATH = APP_DIRECTORY.parent / "assets" / "SMART_Gelose.png"
DATA_PATH = APP_DIRECTORY.parent / "assets" / "sample" / "Test_countings.csv"

# Add the logo to the sidebar
st.sidebar.image(str(LOGO_PATH), width=250)

# Load the test results dataset into a DataFrame
df = pd.read_csv(str(DATA_PATH), sep=";")

# Create tabs for the app
tab1, tab2, tab3 = st.tabs(["Prototype Demo", "Model Performance", "About the app"])

# Configure the sidebar with information and options
st.sidebar.write("")
st.sidebar.write("The purpose of this web app is to streamline the process of *counting Colony-Forming Units* in *gelose plates* using advanced computer vision techniques.")
st.sidebar.divider()
st.sidebar.write("Please load a gelose plate picture to trigger off the analysis. You can choose a picture among a sample library or use your own picture witht the toggle button below.")
st.sidebar.divider()
st.sidebar.write("Options")

# Sidebar toggle options
sample_library = st.sidebar.toggle("Sample library loading", value=False)
activate_shutter_view = st.sidebar.toggle("Activate the shutter view", value=False)
show_probabilities = st.sidebar.toggle("Show prediction probabilities", value=False)

st.title("Prototype Demonstration 🦠🧫🔎")

def load_sample_image(selected_image):
    """
    Load an image from a remote GitHub repository.
    
    Args:
        selected_image (str): Name of the image file to load.
        
    Returns:
        BytesIO: Image file in a BytesIO format.
    """
    github_repo_url = "https://github.com/arnaud-dg/CFU-counter/assets/sample/"
    image_url = github_repo_url + selected_image
    
    response = requests.get(image_url)
    if response.status_code == 200:
        img_file = BytesIO(response.content)
        img_file.name = selected_image
        return img_file
    else:
        st.error("Failed to load the image from GitHub.")
        return None

# Load the image based on user input or from sample library
if sample_library:
    image_files = ["test.jpg", "test_2.jpg", "test_3.jpg", "test_4.jpg", "test_5.jpg", "test_6.jpg", "test_7.jp", "test_8.jpg"]
    selected_image = st.selectbox("Please select a sample picture.", image_files)
    uploaded_file = load_sample_image(selected_image)
else:
    uploaded_file = st.file_uploader("Please select your own new file through the browser of drag and drop it.", type=["jpg", "png", "jpeg"])

def analyze_image(uploaded_file):
    """
    Analyze the uploaded image using the FastAPI server and return the prediction results.
    
    Args:
        uploaded_file (BytesIO): The uploaded image file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the prediction results.
    """
    st.session_state['image'] = load_image(uploaded_file)
    st.session_state['image_name'] = uploaded_file.name
    st.session_state['image_with_mask'] = add_transparent_mask(st.session_state['image'], 0.15)
    
    with st.spinner("Processing..."):
        img_byte_arr = io.BytesIO()
        st.session_state['image'].save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        response = requests.post("https://ufc-counter-api-e72d4934bdd3.herokuapp.com/predict/", files={"file": img_byte_arr})
        
        if response.status_code == 200:
            try:
                response_json = response.json()
                return pd.read_json(StringIO(response_json["predictions"]))
            except ValueError:
                st.error("Error parsing JSON response from API")
                st.write(response.text)
        else:
            st.error(f"API request failed with status code {response.status_code}")
            st.write(response.text)

if uploaded_file is not None:
    if 'results' not in st.session_state or st.session_state['image_name'] != uploaded_file.name:
        st.session_state['results'] = analyze_image(uploaded_file)

    # Display the number of detected objects
    st.session_state['predicted_ufc_count'] = len(st.session_state['results'])  
    known_image = st.session_state['image_name'] in df['image_name'].unique()
    if known_image:
        st.session_state['real_ufc_count'] = df[df['image_name'] == st.session_state['image_name']]['result'].values[0]

    # Configure layout columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Draw rectangles for each prediction
        st.session_state['image_with_rectangles'] = draw_rectangles(st.session_state['image'].copy(), st.session_state['results'], show_probabilities)
        
        # Shutter view logic
        if activate_shutter_view:
            split_percentage = st.slider("Shutter parameter", 0, 100, 50, key="slider1", label_visibility='collapsed')
            combined_image = combine_images(st.session_state['image_with_mask'], st.session_state['image_with_rectangles'], split_percentage)
            st.image(combined_image, caption='Prediction results', use_column_width=True)
        else:
            st.slider("Shutter parameter", 0, 100, 50, key="slider1", disabled=True, label_visibility='collapsed')
            st.image(st.session_state['image_with_rectangles'], caption='Prediction results', use_column_width=True)

    with col2:
        if uploaded_file is not None:
            # Display metrics
            st.markdown('#')
            st.markdown('#')
            st.metric("**Predicted number of UFC:**", st.session_state['predicted_ufc_count'])
            st.markdown('##')
            if known_image:
                st.metric("**Real number of UFC:**", st.session_state['real_ufc_count'])
                st.markdown('##')
                if int(st.session_state['predicted_ufc_count']) == int(st.session_state['real_ufc_count']):
                    st.success("It's a perfect match!")
                elif abs(int(st.session_state['predicted_ufc_count']) - int(st.session_state['real_ufc_count'])) <= int(st.session_state['real_ufc_count']) * 0.05:
                    st.warning("Model is pretty close! (+/-5%)")
                else:
                    st.error("It's not a match! Model must be optimized.")
