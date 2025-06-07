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
import sqlite3
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
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []

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

# Dictionnaire des codes bancaires
BANK_CODES = {
    "01": "Arab Tunisian Bank (ATB)",
    "03": "Banque de Tunisie (BT)",
    "04": "Attijari Bank",
    "05": "Banque Tuniso-Koweitienne (BTK)",
    "06": "Banque Tuniso-Libyenne (BTL)",
    "07": "Amen Bank",
    "08": "Banque Internationale Arabe de Tunisie (BIAT)",
    "09": "Banque Zitouna",
    "10": "Société Tunisienne de Banque (STB)",
    "11": "Union Bancaire pour le Commerce et l’Industrie (UBCI)",
    "12": "Union Internationale de Banques (UIB)",
    "13": "Wifak Bank",
    "14": "Banque de l'Habitat (BH Bank)",
    "15": "Al Baraka Bank Tunisia",
    "16": "Qatar National Bank Tunisia (QNB Tunisia)",
    "17": "Citibank Tunisia",
    "18": "Banque de Financement des PME (BFPME)",
    "19": "Banque Tunisienne de Solidarité (BTS)",
    "20": "Banque de Tunisie et des Émirats (BTE)"
}

def get_bank_from_rib(rib):
    if not rib or len(rib) < 2:
        return "Unknown"
    prefix = rib[:2]
    return BANK_CODES.get(prefix, "Unknown")

def safe_float(val):
    try:
        return float(val.replace(",", "."))
    except:
        return 0.0

def save_transaction_to_db(data):
    try:
        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                rib1 TEXT PRIMARY KEY,
                rib2 TEXT,
                nom TEXT,
                nomreciver TEXT,
                plafond REAL,
                num_cheque TEXT,
                montant REAL,
                date TEXT,
                bank TEXT,
                created_at TEXT
            )
        """)

        bank = get_bank_from_rib(data.get("rib1", "Not Detected"))
        cursor.execute("""
            INSERT OR REPLACE INTO transactions (
                rib1, rib2, nom, nomreciver, plafond, num_cheque, montant, date, bank, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("rib1", "Not Detected"),
            data.get("rib2", "Not Detected"),
            data.get("nom", "Not Detected"),
            data.get("nomreciver", "Not Detected"),
            safe_float(data.get("plafond", "0")),
            data.get("num_cheque", "Not Detected"),
            safe_float(data.get("montant", "0")),
            data.get("date", "Not Detected"),
            bank,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()
        print("✅ Transaction insérée dans la base de données.")
    except Exception as e:
        print("❌ Erreur insertion dans SQLite :", str(e))

def analyze_transaction(data):
    result = {"potential_customer": False, "fraud_risk": False, "analysis": {}}

    rib1 = data.get("rib1", "Not Detected")
    plafond = safe_float(data.get("plafond", "0"))
    montant = safe_float(data.get("montant", "0"))
    num_cheque = data.get("num_cheque", "Not Detected")
    nom = data.get("nom", "Not Detected")
    date = data.get("date", "Not Detected")

    bank = get_bank_from_rib(rib1)
    result["analysis"]["bank"] = bank

    if plafond > 10000 and bank != "Attijari Bank":
        result["analysis"]["high_plafond"] = True
        if montant > 0.5 * plafond:
            result["analysis"]["high_amount_ratio"] = True
            result["potential_customer"] = True

    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT num_cheque, montant, nom FROM transactions WHERE rib1 = ?", (rib1,))
    history = cursor.fetchall()
    conn.close()

    if history:
        periodic = len(history) > 1 and all(abs(float(h[1]) - montant) < 1000 for h in history if h[1])
        successive = all(int(h[0]) == int(num_cheque) + i for i, h in enumerate(history)) if num_cheque != "Not Detected" and all(h[0] != "Not Detected" for h in history) else False
        if periodic and montant > 5000:
            result["analysis"]["periodic_high_transactions"] = True
            result["potential_customer"] = True
        if successive and all(h[2] == nom for h in history):
            result["analysis"]["successive_cheques_same_sender"] = True
            result["potential_customer"] = False

    if montant > 0.9 * plafond:
        result["analysis"]["near_plafond"] = True
        result["fraud_risk"] = True
    if plafond > 100000 and montant < 1000 and len(history) > 5:
        result["analysis"]["small_amounts_high_plafond"] = True
        result["fraud_risk"] = True

    return result

# Interface Streamlit
st.markdown("""
<h1 style='text-align: center; color: black; font-size: 40px; width: 100%; background-color: lightgray; padding: 10px; margin-bottom: 10px'>
Cheque Book Extraction using AI
</h1>
""", unsafe_allow_html=True)

st.subheader("Upload Cheque Images")
col1, col2 = st.columns(2)
with col1:
    front_image_file = st.file_uploader("Choose front image (recto)...", type=["jpg", "jpeg", "png"])
with col2:
    back_image_file = st.file_uploader("Choose back image (verso)...", type=["jpg", "jpeg", "png"])

zip_file = st.file_uploader("Choose a zip file containing front and back images", type=["zip"])

if front_image_file is not None and back_image_file is not None:
    front_image = Image.open(front_image_file)
    back_image = Image.open(back_image_file)
    
    st.subheader("Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(front_image, caption='Front image (recto)', use_container_width=True)
    with col2:
        st.image(back_image, caption='Back image (verso)', use_container_width=True)
    
    if st.button("Extract Cheque Data"):
        try:
            response = model.generate_content([input_prompt, front_image, back_image])
            st.session_state.cheque_data = response.text.strip("```json")
            st.session_state.cheque_data = json.loads(st.session_state.cheque_data)
            st.success("Data extracted successfully!")
            st.json(st.session_state.cheque_data)
            
            cheque_name = f'{st.session_state.cheque_data["num_cheque"]}_cheque.jpg'
            front_path = os.path.join(cheque_directory, f"front_{cheque_name}")
            back_path = os.path.join(cheque_directory, f"back_{cheque_name}")
            front_image.convert('RGB').save(front_path)
            back_image.convert('RGB').save(back_path)
            
            cp_image = cp.preprocess_image(front_image)
            cp_image = cp.sharpen_image(cp_image)
            sign_name = f'{st.session_state.cheque_data["num_cheque"]}_signature.jpg'
            sign_path = os.path.join(signature_directory, sign_name)
            cp_image = Image.fromarray(cp_image)
            cp_image.save(sign_path)
            
            st.session_state.cheque_data["front_img"] = front_path
            st.session_state.cheque_data["back_img"] = back_path
            st.session_state.cheque_data["sign_img"] = sign_path
            
            save_transaction_to_db(st.session_state.cheque_data)
            analysis = analyze_transaction(st.session_state.cheque_data)
            st.session_state.customer.append({**st.session_state.cheque_data, **{"analysis": analysis}})
            st.json(analysis)
            
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
            # Grouper les images par nom de base (ex. cheque_001_0 et cheque_001_1)
            image_pairs = {}
            for img_file in image_files:
                base_name = img_file.rsplit('_', 1)[0]  # Extrait "cheque_001" de "cheque_001_0.jpg"
                if base_name not in image_pairs:
                    image_pairs[base_name] = {"front": None, "back": None}
                if img_file.endswith('_0.jpg') or img_file.endswith('_0.png') or img_file.endswith('_0.jpeg'):
                    image_pairs[base_name]["front"] = img_file
                elif img_file.endswith('_1.jpg') or img_file.endswith('_1.png') or img_file.endswith('_1.jpeg'):
                    image_pairs[base_name]["back"] = img_file

            for base_name, pair in image_pairs.items():
                if pair["front"] and pair["back"]:
                    try:
                        with zip_ref.open(pair["front"]) as front_img_file:
                            front_image = Image.open(BytesIO(front_img_file.read()))
                        with zip_ref.open(pair["back"]) as back_img_file:
                            back_image = Image.open(BytesIO(back_img_file.read()))
                        
                        response = model.generate_content([input_prompt, front_image, back_image])
                        cheque_data = response.text.strip("```json")
                        cheque_data = json.loads(cheque_data)
                        
                        cheque_name = f'{cheque_data["num_cheque"]}_cheque.jpg'
                        front_path = os.path.join(cheque_directory, f"front_{cheque_name}")
                        back_path = os.path.join(cheque_directory, f"back_{cheque_name}")
                        front_image.convert('RGB').save(front_path)
                        back_image.convert('RGB').save(back_path)
                        
                        cp_image = cp.preprocess_image(front_image)
                        cp_image = cp.sharpen_image(cp_image)
                        sign_name = f'{cheque_data["num_cheque"]}_signature.jpg'
                        sign_path = os.path.join(signature_directory, sign_name)
                        cp_image = Image.fromarray(cp_image)
                        cp_image.save(sign_path)
                        
                        cheque_data["front_img"] = front_path
                        cheque_data["back_img"] = back_path
                        cheque_data["sign_img"] = sign_path
                        
                        save_transaction_to_db(cheque_data)
                        analysis = analyze_transaction(cheque_data)
                        st.session_state.customer.append({**cheque_data, **{"analysis": analysis}})
                    except json.JSONDecodeError:
                        st.error(f"Failed to parse data for {pair['front']} or {pair['back']}")
                    except Exception as e:
                        st.error(f"An error occurred for {pair['front']} or {pair['back']}: {str(e)}")
                else:
                    st.warning(f"Missing front or back image for {base_name}")

            st.success("Batch processing completed!")

if st.session_state.customer:
    st.subheader("Transaction History and Analysis")
    for transaction in st.session_state.customer:
        st.json(transaction)
    
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