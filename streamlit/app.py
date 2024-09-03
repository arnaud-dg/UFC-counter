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

# CSS to hide slider numbers
hide_slider_numbers = """
    <style>
    /* Hide the min and max labels */
    div[data-testid="stTickBar"] {
        display: none;
    }

    /* Hide the current value displayed above the slider */
    div[data-testid="stThumbValue"] {
        display: none;
    }
    </style>
"""

st.markdown(hide_slider_numbers, unsafe_allow_html=True)

# Define a color palette for the app's theme
color_palette = {
    'grey': '#94979C',
    'bordeaux': '#6A0910',
    'cream': '#E3DCD4',
    'red': '#9A0E1B'
}

# Get the path of the current script directory and the associated paths
app_directory = Path(__file__).resolve().parent
logo_path = app_directory.parent / "assets" / "SMART_Gelose.png"
data_path = app_directory.parent / "assets" / "sample" / "Test_countings.csv"

# Add the logo to the sidebar
st.sidebar.image(str(logo_path), width=250)

# Load the test results dataset into a DataFrame
df = pd.read_csv(str(data_path), sep=";")

# Create tabs for the app
tab1, tab2, tab3 = st.tabs(["Prototype Demo", "Model Performance", "About the app"])

# Configure the sidebar with information and options
st.sidebar.write("")
st.sidebar.write("The purpose of this web app is to streamline the process of *counting UFC* in *gelose plates* using advanced computer vision techniques.")
st.sidebar.write("3 navigation tabs are available:")
st.sidebar.write("- *Prototype Demonstration*: Please load a Petri dish picture to trigger off the analysis. You can choose a predefined set or use your own picture.")
st.sidebar.write("- *Model Performance*: Provide the model learning and efficiency metrics. This is the data scientist geek part.")
st.sidebar.write("- *About the app*: Describe information about this project.")
st.sidebar.divider()
st.sidebar.write("Options")
sample_library = st.sidebar.toggle("Sample library loading", value=False)
activate_shutter_view = st.sidebar.toggle("Activate the shutter view", value=False)
show_probabilities = st.sidebar.toggle("Show prediction probabilities", value=False)

with tab1:
    st.title("Prototype Demonstration 🦠🧫🔎")

    if sample_library==True:
        # List of image files available in the GitHub repository
        github_repo_url = "https://github.com/arnaud-dg/CFU-counter/assets/sample/"
        image_files = ["test.jpg", "test_2.jpg", "test_3.jpg", "test_4.jpg", "test_5.jpg", "test_6.jpg", "test_7.jp", "test_8.jpg"]
        # Dropdown to select image from GitHub
        selected_image = st.selectbox("Please select a sample picture.", image_files) 
        image_url = github_repo_url + selected_image
        # Simulate the uploaded_file by downloading and converting it to a BytesIO object
        response = requests.get(image_url)
        if response.status_code == 200:
            uploaded_file = BytesIO(response.content)
            uploaded_file.name = selected_image  # Simulate the file name attribute
        else:
            st.error("Failed to load the image from GitHub.")
            uploaded_file = None
    else:
        # Upload image
        uploaded_file = st.file_uploader("Please select your own new file through the browser of drag and drop it.", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        if 'results' not in st.session_state or st.session_state['image_name'] != uploaded_file.name:
            # Load and analyse image only if the image is new
            st.session_state['image'] = load_image(uploaded_file)
            st.session_state['image_name'] = uploaded_file.name
            st.session_state['image_with_mask'] = add_transparent_mask(st.session_state['image'], 0.15)

            # Make a prediction request to the FastAPI server
            with st.spinner("Processing..."):
                img_byte_arr = io.BytesIO()
                st.session_state['image'].save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                response = requests.post("https://cfucounter-6baf3091d9c8.herokuapp.com/predict/", files={"file": img_byte_arr})

                if response.status_code == 200:
                    try:
                        response_json = response.json()
                        results = pd.read_json(StringIO(response_json["predictions"]))
                        st.session_state['results'] = results
                    except ValueError:
                        st.error("Error parsing JSON response from API")
                        st.write(response.text)  # Display raw response content in case of error
                else:
                    st.error(f"API request failed with status code {response.status_code}")
                    st.write(response.text)  # Display raw response content in case of error

        # Display the number of detected objects
        st.session_state['predicted_ufc_count'] = len(st.session_state['results'])  
        if st.session_state['image_name'] in df['image_name'].unique():
            known_image = True
            st.session_state['real_ufc_count'] = df[df['image_name'] == st.session_state['image_name']]['result'].values[0]
        else:
            known_image = False

        # Configure layout columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Draw rectangles for each prediction
            st.session_state['image_with_rectangles'] = draw_rectangles(st.session_state['image'].copy(), st.session_state['results'], show_probabilities)

            divider1, divider2, divider3 = st.columns([1, 5, 1])
            with divider1:
                st.write('')
            with divider2:
                st.write('')
                if activate_shutter_view:
                    split_percentage = st.slider("Shutter parameter", 0, 100, 50, key="slider1", label_visibility='collapsed')  # Shutter slider
                    combined_image = combine_images(st.session_state['image_with_mask'], st.session_state['image_with_rectangles'], split_percentage)
                    st.image(combined_image, caption='Prediction results', use_column_width=True)
                else:
                    st.slider("Shutter parameter", 0, 100, 50, key="slider1", disabled=True, label_visibility='collapsed')  # Disabled shutter slider
                    st.image(st.session_state['image_with_rectangles'], caption='Prediction results', use_column_width=True)
            with divider3:
                st.write('')

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
                    elif (int(st.session_state['predicted_ufc_count']) >= int(st.session_state['real_ufc_count']) * 0.95) and (int(st.session_state['predicted_ufc_count']) <= int(st.session_state['real_ufc_count']) * 1.05):
                        st.warning("Model is pretty close! (+/-5%)")
                    else:
                        st.error("It's not a match! Model must be optimized.")

with tab2:
    st.title("Model Performance")
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Confusion matrix")
            confusion_matrix_path = app_directory.parent / "plots" / "confusion_matrix.png"
            st.image(str(confusion_matrix_path), use_column_width=True)
        with col2:
            st.subheader("Learning results")
            results_path = app_directory.parent / "plots" / "results.png"
            st.image(str(results_path), use_column_width=True)

    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("PR Curve")
            pr_curve_path = app_directory.parent / "plots" / "PR_curve.png"
            st.image(str(pr_curve_path), use_column_width=True)
        with col2:
            st.subheader("Error analysis")
            error_analysis_path = app_directory.parent / "plots" / "confusion_matrix.png"
            st.image(str(error_analysis_path), use_column_width=True)

with tab3:
    st.title("About the app")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=600)
