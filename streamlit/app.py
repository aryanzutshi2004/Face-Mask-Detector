import streamlit as st
import requests
import json # Good practice for handling JSON data

# Define the server URL (ensure your FastAPI server is running here)
FAST_API_URL = "http://127.0.0.1:8000" 

st.title("Face Mask Detector 😷")
st.subheader("Camera Input Test")

# 1. Open webcam and capture image
img_file = st.camera_input("Take a picture")

# Check if an image has been captured before proceeding
if img_file is not None:
    st.info("Sending image to server for prediction...")
    
    # Prepare the payload for the POST request
    # The server expects the file under the key "image"
    # `img_file.getvalue()` gets the raw bytes of the image file
    files = {'image': img_file.getvalue()}
    
    try:
        # 2. Corrected requests.post call
        response = requests.post(FAST_API_URL + "/prediction", files=files)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # 3. Correctly parse the JSON response
            prediction_result = response.json()
            predicted_label = prediction_result.get("Prediction", "Error: Prediction key not found.")
            
            # Display the result
            st.success(f"The person is **{predicted_label}**")
            
        else:
            # Handle server errors (e.g., status 400, 500)
            st.error(f"Server Error: Status Code {response.status_code}")
            try:
                error_details = response.json().get("error", "No detailed error message.")
                st.exception(f"Details: {error_details}") # type: ignore
            except json.JSONDecodeError:
                st.exception("Server returned a non-JSON error response.") # type: ignore

    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the FastAPI server at {FAST_API_URL}. Ensure it is running.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")