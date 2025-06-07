#!/usr/bin/python
import google.generativeai as genai
import PIL.Image
from PIL import Image
from io import BytesIO
import os
import cv2
import streamlit as st
import numpy as np
import zipfile
import json
import shutil
import pandas as pd
import streamlit.components.v1 as components
from datetime import datetime
from dotenv import load_dotenv
import crop as cp

load_dotenv()

# Configurer l'API Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Initialiser l'état de session
if 'customer' not in st.session_state:
    st.session_state.customer = []
if 'cheque_data' not in st.session_state:
    st.session_state.cheque_data = {}

# Charger le prompt
input_prompt = st.secrets['prompt1']['input_prompt']

# Créer les dossiers
upload_directory = "input_images"
if not os.path.exists(upload_directory):
    os.makedirs(upload_directory)

signature_directory = "sign_images"
if not os.path.exists(signature_directory):
    os.makedirs(signature_directory)

cheque_directory = "cheque_images"
if not os.path.exists(cheque_directory):
    os.makedirs(cheque_directory)

# Interface Streamlit
st.markdown("""
<h1 style='text-align: center; color: black; font-size: 40px; width: 100%; background-color: lightgray; padding: 10px; margin-bottom: 10px'>
Cheque Book Extraction using AI
</h1>
""", unsafe_allow_html=True)

# Téléchargement des images recto et verso
st.subheader("Upload Cheque Images")
col1, col2 = st.columns(2)
with col1:
    front_image_file = st.file_uploader("Choose front image (recto)...", type=["jpg", "jpeg", "png"])
with col2:
    back_image_file = st.file_uploader("Choose back image (verso)...", type=["jpg", "jpeg", "png"])

# Téléchargement d'un fichier ZIP
zip_file = st.file_uploader("Choose a zip file containing front and back images", type=["zip"])

if front_image_file is not None and back_image_file is not None:
    front_image = Image.open(front_image_file)
    back_image = Image.open(back_image_file)
    
    st.subheader("Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(front_image, caption='Front image (recto)', use_column_width=True)
    with col2:
        st.image(back_image, caption='Back image (verso)', use_column_width=True)
    
    if st.button("Extract Cheque Data"):
        try:
            # Extraire les données des deux images
            response = model.generate_content([input_prompt, front_image, back_image])
            st.session_state.cheque_data = response.text.strip("```json")
            st.session_state.cheque_data = json.loads(st.session_state.cheque_data)
            st.success("Data extracted successfully!")
            st.json(st.session_state.cheque_data)
            
            # Sauvegarder les images
            cheque_name = f'{st.session_state.cheque_data["num_cheque"]}_cheque.jpg'
            front_path = os.path.join(cheque_directory, f"front_{cheque_name}")
            back_path = os.path.join(cheque_directory, f"back_{cheque_name}")
            front_image.convert('RGB').save(front_path)
            back_image.convert('RGB').save(back_path)
            
            # Prétraiter et sauvegarder la signature (depuis le recto)
            cp_image = cp.preprocess_image(front_image)
            cp_image = cp.sharpen_image(cp_image)
            sign_name = f'{st.session_state.cheque_data["num_cheque"]}_signature.jpg'
            sign_path = os.path.join(signature_directory, sign_name)
            cp_image = Image.fromarray(cp_image)
            cp_image.save(sign_path)
            
            # Ajouter les chemins aux données
            st.session_state.cheque_data["front_img"] = front_path
            st.session_state.cheque_data["back_img"] = back_path
            st.session_state.cheque_data["sign_img"] = sign_path
            
            st.session_state.customer.append(st.session_state.cheque_data)
            
        except json.JSONDecodeError:
            st.error("Failed to parse extracted data as JSON.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if zip_file is not None:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(upload_directory)
    st.success("Images have been uploaded successfully!")
    
    if st.button("Process ZIP"):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]
            # Associer les images par paires (front et back)
            front_files = sorted([f for f in image_files if "front" in f.lower()])
            back_files = sorted([f for f in image_files if "back" in f.lower()])
            
            if len(front_files) != len(back_files):
                st.error("The ZIP must contain an equal number of front and back images.")
                st.stop()
            
            for front_file, back_file in zip(front_files, back_files):
                try:
                    with zip_ref.open(front_file) as img_file:
                        front_image = Image.open(BytesIO(img_file.read()))
                    with zip_ref.open(back_file) as img_file:
                        back_image = Image.open(BytesIO(img_file.read()))
                    
                    # Extraire les données
                    response = model.generate_content([input_prompt, front_image, back_image])
                    cheque_data = response.text.strip("```json")
                    cheque_data = json.loads(cheque_data)
                    
                    # Sauvegarder les images
                    cheque_name = f'{cheque_data["num_cheque"]}_cheque.jpg'
                    front_path = os.path.join(cheque_directory, f"front_{cheque_name}")
                    back_path = os.path.join(cheque_directory, f"back_{cheque_name}")
                    front_image.convert('RGB').save(front_path)
                    back_image.convert('RGB').save(back_path)
                    
                    # Prétraiter et sauvegarder la signature
                    cp_image = cp.preprocess_image(front_image)
                    cp_image = cp.sharpen_image(cp_image)
                    sign_name = f'{cheque_data["num_cheque"]}_signature.jpg'
                    sign_path = os.path.join(signature_directory, sign_name)
                    cp_image = Image.fromarray(cp_image)
                    cp_image.save(sign_path)
                    
                    # Ajouter les chemins aux données
                    cheque_data["front_img"] = front_path
                    cheque_data["back_img"] = back_path
                    cheque_data["sign_img"] = sign_path
                    
                    st.session_state.customer.append(cheque_data)
                except json.JSONDecodeError:
                    st.error(f"Failed to parse data for {front_file} or {back_file}")
                except Exception as e:
                    st.error(f"An error occurred for {front_file} or {back_file}: {str(e)}")
            
            st.success("Batch processing completed!")

if st.session_state.customer:
    if st.button("Ready for Download"):
        def create_zip_with_folder_and_file(zip_name, folder1, folder2, filename):
            with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(folder1):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(os.path.basename(folder1), file)
                        zipf.write(file_path, arcname)
                
                for root, _, files in os.walk(folder2):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(os.path.basename(folder2), file)
                        zipf.write(file_path, arcname)
                
                zipf.write(filename, os.path.basename(filename))
        
        def cleanup():
            for directory in [upload_directory, cheque_directory, signature_directory]:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
            if os.path.exists(excel_file_name):
                os.remove(excel_file_name)
            if os.path.exists(zip_output):
                os.remove(zip_output)
            st.session_state.customer = []
            st.session_state.cheque_data = {}
        
        df = pd.DataFrame(st.session_state.customer)
        excel_file_name = 'cheque_table.xlsx'
        df.to_excel(excel_file_name, index=False)
        
        zip_output = 'Result_data.zip'
        create_zip_with_folder_and_file(zip_output, cheque_directory, signature_directory, excel_file_name)
        
        with open(zip_output, 'rb') as f:
            st.download_button(
                label="Download Output in zip (Table, cheque & sign folders)",
                data=f.read(),
                file_name=zip_output,
                mime='application/zip'
            )
            cleanup()

# Pied de page
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: lightgray;
    color: black;
    text-align: center;
    padding: 10px;
}
a {
    color: blue;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
<div class="footer">
</div>
"""
components.html(footer, height=100)