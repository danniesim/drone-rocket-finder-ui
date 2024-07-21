import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
from streamlit_image_zoom import image_zoom  # Import the zoom functionality

# Function to load and filter data


def load_data():
    csv_path = 'drone-find-rocket/gpt4o-results.csv'  # Path to your CSV file
    df = pd.read_csv(csv_path)
    # Filter rows where 'detected' == 'Yes'
    df_filtered = df[df['detected'] == 'Yes']
    return df_filtered


def apply_high_pass_filter(image_path):
    # Load image with OpenCV
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # High-pass filter
    high_pass = cv2.subtract(gray, blurred)

    # Convert to RGB to maintain consistency
    high_pass_rgb = cv2.cvtColor(high_pass, cv2.COLOR_GRAY2RGB)

    # Convert back to PIL Image
    img_pil = Image.fromarray(high_pass_rgb)

    return img_pil


def main():
    st.title("Drone Rocket Finder")

    # Load and filter data
    df = load_data()

    if df.empty:
        st.write("No images detected.")
        return

    # Initialize session state to track the current image index and toggle
    # state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    if 'high_pass_filter_toggle' not in st.session_state:
        st.session_state.high_pass_filter_toggle = False

    # Get the current image based on the index
    selected_image = df.iloc[st.session_state.current_index]['image']

    # Layout with two columns: one for image details and control
    # buttons/checkbox
    details_col, controls_col = st.columns([3, 1])

    with details_col:
        # Display image details
        image_details = df[df['image'] == selected_image].iloc[0]
        st.write(f"### Image {selected_image}")
        st.write(f"**Description:** {image_details['description']}")

    with controls_col:
        # Toggle button for applying high-pass filter
        st.checkbox(
            'Apply High-Pass Filter',
            key="high_pass_filter_toggle"
        )

        # Navigation buttons
        prev_button, next_button = st.columns([1, 1])
        with prev_button:
            if st.button('Previous'):
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.rerun()

        with next_button:
            if st.button('Next'):
                if st.session_state.current_index < len(df) - 1:
                    st.session_state.current_index += 1
                    st.rerun()

    # Display image
    image_path = f'drone-find-rocket/tiles-1024/{selected_image}'
    if os.path.exists(image_path):
        if st.session_state.high_pass_filter_toggle:
            img = apply_high_pass_filter(image_path)
        else:
            img = Image.open(image_path)

        # Ensure image is in the correct mode for display
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Use the streamlit_image_zoom component to display the zoomable image
        image_zoom(img, size=(700, 700))

    else:
        st.error(f"Image {selected_image} not found.")

    # Display the filtered dataframe at the bottom
    st.write("### Overview")

    col_i, col1, col2 = st.columns([0.3, 4, 2])

    with col1:
        st.write("#### Description")

    with col2:
        st.write("#### Select")

    for i, (idx, row) in enumerate(df.iterrows()):
        col_i, col1, col2 = st.columns([0.3, 4, 2])
        with col_i:
            st.write(f"{i + 1}")
        with col1:
            st.write(f"{row['description']}")
        with col2:
            if st.button(
                    f"{row['image']}", key=f"select_{i}"):
                st.session_state.current_index = i
                st.rerun()


if __name__ == "__main__":
    main()
