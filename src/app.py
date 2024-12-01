import streamlit as st
import os
from prometheus_client import start_http_server, Gauge
from UNET_script import generate_output
import boto3


### GLOBAL VARIABLES
ROOT_PATH = os.getcwd()
 

st.title("FOCUS: Fully Optimized Convolutional Unet Segmentation for Tumor Detection")
st.write("To use FOCUS please upload 2 CT scans using the input button. Please note that FOCUS only takes in .nii files. After upload is complete, please wait for FOCUS to process the image for segmentation. The output will then display the newly segmented images.")
input = st.file_uploader(label="Upload", type='nii', key='input', accept_multiple_files=True)
if len(input) > 1:
    # saves images to s3 bucket
    s3 = boto3.client(
        service_name='s3',
        region_name='us-east-2',
        aws_access_key_id='AKIAUQ4L3FBTESMORT75',
        aws_secret_access_key='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',
    )
    bucket_name = "ct-scans--use2-az1--x-s3"
    for img in input:
        file_name = img.name
        #s3.upload_fileobj(img, bucket_name, file_name)
        st.success(f"Successfully Uploaded {file_name} into S3 Bucket")

    # initiating script to generate gifs from CT scans in data
    mean_dice_score, recall, precision = generate_output()
    f1_score = (2 * recall * precision) / (precision + recall)
    st.write(f"Results of the CT Scans- \nMean Dice: {mean_dice_score} | Recall: {recall} | Precision: {precision} | F1-Score: {f1_score}")

    if st.button("Record Metrics with Prometheus"):
        # Define Prometheus metrics
        mean_dice_metric = Gauge("mean_dice_score", "Mean Dice Score")
        recall_metric = Gauge("recall", "Recall Metric")
        precision_metric = Gauge("precision", "Precision Metric")
        f1_score_metric = Gauge("f1_score", "F1 Score Metric")

        # Start Prometheus HTTP server
        # to access prometheus stats type "localhost:8000" into your browswer
        start_http_server(8000)

        # Update Prometheus metrics
        mean_dice_metric.set(mean_dice_score)
        recall_metric.set(recall)
        precision_metric.set(precision)
        f1_score_metric.set(f1_score)
    
    # creating a 2x2 grid for the gifs
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    gif_1 = os.path.join(ROOT_PATH, "saved_gifs", "mask0.gif")
    gif_2 = os.path.join(ROOT_PATH, "saved_gifs", "mask1.gif")
    gif_3 = os.path.join(ROOT_PATH, "saved_gifs", "mask2.gif")
    gif_4 = os.path.join(ROOT_PATH, "saved_gifs", "mask3.gif")

    # Display the GIFs in the grid
    with col1:
        st.image(gif_1, caption="Predicted Mask 1", use_container_width=True)
    with col2:
        st.image(gif_2, caption="Ground Truth 1", use_container_width=True)
    with col3:
        st.image(gif_3, caption="Predicted Mask 2", use_container_width=True)
    with col4:
        st.image(gif_4, caption="Ground Truth 2", use_container_width=True)

st.link_button("Feedback", "https://forms.gle/iW9ZBiDqkPBYNmpT8")