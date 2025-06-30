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
import ollama
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='chequebot.log'
)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# System prompt for the chatbot
system_prompt = """
You are ChequeBot, a friendly and helpful assistant designed to assist with the Cheque Book AI platform and analyze cheque transactions. I'm here to make your experience smooth and informative—it's 01:08 PM CET on Tuesday, June 24, 2025, and I'm ready to assist!

Here is what the platform does:
- 📤 Upload Cheque Images (front and back or ZIP batch)
- 🧠 Extract important fields using AI (e.g., sender name, receiver, RIBs, cheque number, date, amount, plafond)
- ✍️ Isolate and save signature images from cheques
- 📊 Store multiple cheque records and export them
- 🕵️‍♂️ Analyze transactions for fraud risks and potential customers
- 📁 Download everything as a structured ZIP with images and Excel
- 🤖 The AI model used is Gemini for extraction, and the interface is built with Streamlit

### Transaction Analysis Rules:
- **Potential Customer**: Marked as "Yes" if:
  - Plafond > 10,000 and Amount > 50% of Plafond (high_plafond and high_amount_ratio).
  - Periodic transactions (>1) with Amount > 5,000 (periodic_high_transactions).
- **Fraud Risk**: Marked as "Yes" if:
  - Amount > Plafond (amount_exceeds_plafond). This triggers a red alert with the message: "The amount exceeds the plafond."
  - Amount > 90% of Plafond (near_plafond).
  - Plafond > 100,000, Amount < 1,000, and >5 transactions in history (small_amounts_high_plafond).
  - Successive cheques (>1) from the same sender with the same RIBs and consecutive cheque numbers (difference of +1 or -1) with increasing amounts (successive_cheques_same_sender_increasing).
- Other factors (e.g., successive_cheques_same_sender) may influence the analysis.

### Capabilities:
- Respond warmly to greetings (e.g., "Hi!" → "Hi! How can I help you today at 01:08 PM CET on June 24, 2025?", "Hello!" → "Hello! What can I do for you this afternoon?").
- Answer questions about how the platform works, its benefits, or technical capabilities.
- Analyze transaction history stored in the SQLite database ("transactions.db") and the current session data.
- Provide details on specific cheques (e.g., RIBs, amounts, dates, plafond) based on the data available.
- Explain why a transaction is flagged as a potential customer or fraud risk using the above rules.
- Answer financial questions related to banks (e.g., bank codes, transaction patterns) based on the provided BANK_CODES.

❌ Do NOT answer general questions, casual conversation beyond greetings, jokes, news, or anything not directly related to this platform, cheque transactions, or bank-related financial queries.

If the question is unrelated, respond strictly with:
**"I'm only able to assist with questions about the Cheque Book AI platform, cheque transactions, or bank-related financial queries."**
"""

# Function to query LLaMA with transaction history
def query_llama(user_message):
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions")
    history = cursor.fetchall()
    conn.close()
    
    history_context = "Transaction History:\n" + "\n".join([f"- Cheque {row[1]}: RIB1={row[0]}, RIB2={row[2]}, Sender={row[3]}, Receiver={row[4]}, Amount={row[6]}, Plafond={row[5]}, Date={row[7]}, Bank={row[8]}"])
    history_context += "\nCurrent Session Data:\n" + "\n".join([f"- Cheque {c.get('num_cheque', '')}: RIB1={c.get('rib1', '')}, RIB2={c.get('rib2', '')}, Sender={c.get('nom', '')}, Receiver={c.get('nomreciver', '')}, Amount={c.get('montant', '')}, Plafond={c.get('plafond', '')}, Date={c.get('date', '')}, Analysis={c.get('analysis', {})}" for c in st.session_state.contact])
    
    # Combine system prompt with history context and user message
    full_prompt = f"{system_prompt}\n{history_context}\nUser Message: {user_message}"
    
    # Query Gemini API
    response = model.generate_content(full_prompt)
    
    return response.text

# Bank codes dictionary
BANK_CODES = {
    "01": "Arab Tunisian Bank (ATB)",
    "03": "Banque de Tunisie (BT)",
    "04": "Attijari Bank",
    "05": "Banque Tuniso-Koweitienne (BTK)",
    "06": "Banque Tuniso-Libyeene (BTL)",
    "07": "Amen Bank",
    "08": "Banque Internationale Arabe de Tunisie (BIAT)",
    "09": "Banque Zitouna",
    "10": "Société Tunisienne de Banque (STB)",
    "11": "Union Bancaire pour le Commerce et l'Industrie (UBCI)",
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
    try:
        if not rib or rib == "Not Detected":
            logging.warning(f"Empty or invalid RIB: {rib}")
            return "Unknown"
        
        rib_str = str(rib).strip()
        cleaned_rib = ''.join(filter(str.isdigit, rib_str))
        
        if len(cleaned_rib) < 2:
            logging.warning(f"Cleaned RIB too short: {cleaned_rib}")
            return "Unknown"
        
        prefix = cleaned_rib[:2]
        bank_name = BANK_CODES.get(prefix, "Unknown")
        logging.info(f"RIB: {rib} → Cleaned: {cleaned_rib} → Bank: {bank_name}")
        return bank_name
    except Exception as e:
        logging.error(f"Error in get_bank_from_rib: {str(e)}")
        return "Unknown"

def safe_float(val):
    try:
        if val is None or str(val).strip() == "":
            logging.warning("Empty value in safe_float")
            return 0.0
            
        if isinstance(val, str):
            # Supprimer les espaces et remplacer les virgules par des points
            cleaned = val.replace(" ", "").replace(",", ".")
            logging.info(f"Converting string to float: '{val}' -> '{cleaned}'")
            return float(cleaned)
            
        return float(val)
    except (ValueError, AttributeError) as e:
        logging.error(f"Failed to convert value '{val}' to float: {str(e)}")
        return 0.0

def clear_database():
    try:
        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions")
        conn.commit()
        conn.close()
        logging.info("Database cleared successfully")
        return True
    except Exception as e:
        logging.error(f"Error clearing database: {str(e)}")
        return False

def save_transaction_to_db(data):
    try:
        rib1 = data.get("rib1", data.get("Rib1", data.get("RIB1", "Not Detected")))
        num_cheque = data.get("num_cheque", "Not Detected")
        
        if rib1 == "Not Detected" or num_cheque == "Not Detected":
            logging.warning(f"Skipping transaction - Invalid RIB: {rib1} or Cheque Number: {num_cheque}")
            return False
        
        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                rib1 TEXT,
                num_cheque TEXT,
                rib2 TEXT,
                nom TEXT,
                nomreciver TEXT,
                plafond REAL,
                montant REAL,
                date TEXT,
                bank TEXT,
                created_at TEXT,
                PRIMARY KEY (rib1, num_cheque)
            )
        """)
        
        bank = get_bank_from_rib(rib1)
        plafond_value = data.get("plafond", None)
        logging.info(f"Saving transaction - RIB1: {rib1}, Cheque: {num_cheque}, Plafond: {plafond_value}")
        
        cursor.execute("""
            INSERT INTO transactions (
                rib1, num_cheque, rib2, nom, nomreciver, plafond, montant, date, bank, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rib1,
            num_cheque,
            data.get("rib2", data.get("Rib2", data.get("RIB2", "Not Detected"))),
            data.get("nom", "Not Detected"),
            data.get("nomreciver", "Not Detected"),
            safe_float(plafond_value),
            safe_float(data.get("montant", "0")),
            data.get("date", "Not Detected"),
            bank,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        conn.commit()
        conn.close()
        logging.info(f"Transaction saved successfully: RIB1={rib1}, Cheque={num_cheque}, Plafond={plafond_value}")
        return True
    except sqlite3.IntegrityError as e:
        logging.warning(f"Duplicate transaction detected for RIB1={rib1}, Cheque={num_cheque}: {str(e)}")
        conn.close()
        return False
    except sqlite3.Error as e:
        logging.error(f"SQLite error: {str(e)}")
        conn.close()
        return False
    except Exception as e:
        logging.error(f"Error saving transaction: {str(e)}")
        conn.close()
        return False

def migrate_database():
    try:
        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions_new (
                rib1 TEXT,
                num_cheque TEXT ,
                rib2 TEXT,
                nom TEXT,
                nomreciver TEXT,
                plafond REAL,
                montant REAL,
                date TEXT,
                bank TEXT,
                created_at TEXT,
                PRIMARY KEY (num_cheque)
            )
        """)
        
        cursor.execute("""
            INSERT OR IGNORE INTO transactions_new
            SELECT rib1, num_cheque, rib2, nom, nomreciver, plafond, montant, date, bank, created_at
            FROM transactions
        """)
        
        cursor.execute("DROP TABLE IF EXISTS transactions")
        cursor.execute("ALTER TABLE transactions_new RENAME TO transactions")
        
        conn.commit()
        conn.close()
        logging.info("Database migration completed successfully")
    except sqlite3.Error as e:
        logging.error(f"Database migration error: {str(e)}")
    except Exception as e:
        logging.error(f"Error during migration: {str(e)}")

def analyze_transaction(data):
    result = {
        "potential_customer": False,
        "fraud_risk": False,
        "analysis": {
            "bank": "Unknown",
            "high_plafond": False,
            "high_amount_ratio": False,
            "periodic_high_transactions": False,
            "near_plafond": False,
            "amount_exceeds_plafond": False,  # New flag for amount > plafond
            "small_amounts_high_plafond": False,
            "successive_cheques_same_sender": False,
            "successive_cheques_same_sender_increasing": False,
            "consecutive_cheque_numbers": False,
            "consecutive_cheque_details": []  # Stocke les détails des chèques consécutifs
        }
    }
    
    try:
        rib1 = data.get("rib1", data.get("Rib1", data.get("RIB1", "Not Detected")))
        rib2 = data.get("rib2", data.get("Rib2", data.get("RIB2", "Not Detected")))
        plafond = safe_float(data.get("plafond", "0"))
        montant = safe_float(data.get("montant", "0"))
        num_cheque = data.get("num_cheque", "Not Detected")
        nom = data.get("nom", "Not Detected")
        
        bank = get_bank_from_rib(rib1)
        result["analysis"]["bank"] = bank
        
        # Vérification du client potentiel
        if plafond > 10000 and bank != "Attijari Bank":
            result["analysis"]["high_plafond"] = True
            if montant > 0.5 * plafond:
                result["analysis"]["high_amount_ratio"] = True
                result["potential_customer"] = True
        
        # Connexion à la base de données pour récupérer l'historique
        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()
        cursor.execute("SELECT num_cheque, montant, nom, rib2 FROM transactions WHERE rib1 = ? ORDER BY num_cheque", (rib1,))
        history = cursor.fetchall()
        conn.close()
        
        # Vérification des chèques consécutifs
        if history and len(history) > 0 and rib2 != "Not Detected":
            # Convertir et nettoyer les numéros de chèque
            history_cheques = []
            for h in history:
                try:
                    cleaned = ''.join(filter(str.isdigit, str(h[0])))
                    num = int(cleaned.lstrip('0')) if cleaned else 0
                    history_cheques.append((num, h[0], safe_float(h[1]), h[3]))  # (num_clean, num_original, montant, rib2)
                except:
                    history_cheques.append((0, h[0], safe_float(h[1]), h[3]))
            
            # Nettoyer le numéro de chèque actuel
            try:
                cleaned_current = ''.join(filter(str.isdigit, str(num_cheque)))
                current_num = int(cleaned_current.lstrip('0')) if cleaned_current else 0
            except:
                current_num = 0
            
            history_amounts = [h[2] for h in history_cheques]
            history_rib2s = [h[3] for h in history_cheques]
            
            # Vérification des numéros consécutifs et même RIB
            if all(r != "Not Detected" for r in history_rib2s) and all(r == rib2 for r in history_rib2s):
                for i, (hist_num, hist_original, hist_amount, _) in enumerate(history_cheques):
                    diff = abs(current_num - hist_num)
                    if diff == 1:  # Numéros consécutifs (±1)
                        result["analysis"]["consecutive_cheque_numbers"] = True
                        result["analysis"]["successive_cheques_same_sender"] = True
                        result["analysis"]["consecutive_cheque_details"].append({
                            "previous_cheque": hist_original,
                            "current_cheque": num_cheque,
                            "difference": diff
                        })
                        # Activer fraud_risk dès qu'un numéro consécutif est détecté
                        result["fraud_risk"] = True
                        if montant > hist_amount:
                            result["analysis"]["successive_cheques_same_sender_increasing"] = True

        # Vérification des transactions périodiques élevées
        if len(history) > 1 and all(abs(safe_float(h[1]) - montant) < 1000 for h in history if h[1]):
            if montant > 5000:
                result["analysis"]["periodic_high_transactions"] = True
                result["potential_customer"] = True
        
        # Vérification si le montant dépasse le plafond
        if montant > plafond and plafond > 0:  # Ensure plafond is positive to avoid division issues
            result["analysis"]["amount_exceeds_plafond"] = True
            result["fraud_risk"] = True
        
        # Autres règles de fraude
        if montant > 0.9 * plafond and not result["analysis"]["amount_exceeds_plafond"]:
            result["analysis"]["near_plafond"] = True
            result["fraud_risk"] = True
        
        if plafond > 100000 and montant < 1000 and len(history) > 5:
            result["analysis"]["small_amounts_high_plafond"] = True
            result["fraud_risk"] = True
            
    except Exception as e:
        logging.error(f"Error in analyze_transaction: {str(e)}")
    
    return result

def preprocess_image(image):
    try:
        img = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    except Exception as e:
        logging.error(f"Error in preprocess_image: {str(e)}")
        return None

def sharpen_image(image):
    try:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    except Exception as e:
        logging.error(f"Error in sharpen_image: {str(e)}")
        return image

# Streamlit page configuration
st.set_page_config(page_title="Cheque Book Extraction", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.sidebar-container { padding: 1.5rem 1rem; }
.sidebar-title { font-size: 1.8rem; font-weight: 700; color: #1565c0; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.6rem; }
.nav-button { display: block; width: 100%; text-align: left; padding: 12px 16px; margin: 0.4rem 0; font-size: 1.15rem; font-weight: 600; color: #1f2937; background: transparent; border: none; border-radius: 8px; transition: all 0.3s ease; cursor: pointer; }
.nav-button:hover { background-color: #e3f2fd; color: #0d47a1; }
.nav-button.active { background-color: #bbdefb; color: #0d47a1; }
.custom-header { background-color: #1565c0; color: #ffffff; text-align: center; font-size: 2.3em; font-weight: 700; padding: 0.8em 1.5em; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); letter-spacing: 0.4px; display: inline-block; margin: 0 auto 2rem auto; }
.cheque-box { background-color: #ffffff; max-width: 700px; margin: 30px auto; padding: 25px; border-radius: 14px; box-shadow: 0 6px 12px rgba(0,0,0,0.08); font-size: 1.05rem; color: #1f2937; }
.cheque-box h3 { font-size: 1.6rem; margin-bottom: 20px; }
.cheque-box ul { list-style: none; padding-left: 0; }
.cheque-box li { margin-bottom: 12px; }
.cheque-box strong { font-weight: 700; color: #0ea5e9; }
.custom-success { background-color: #d0f0e0; color: #155724; border: 1px solid #b2dfdb; padding: 14px 20px; border-radius: 10px; font-size: 1.05rem; margin-top: 20px; box-shadow: 0 3px 8px rgba(0,0,0,0.05); }
div[data-testid="stFileUploader"] { background-color: #ffffff; border-radius: 12px; padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); transition: all 0.3s ease-in-out; width: 100% !important; }
div[data-testid="stFileUploader"]:hover { background-color: #f0f8ff; transform: scale(1.01); cursor: pointer; }
div[data-testid="stFileUploader"] label { font-family: 'Segoe Script', cursive; font-size: 1.1rem; font-weight: 600; padding-bottom: 8px; display: block; }
.hero-section { background: linear-gradient(135deg, #1565c0, #1e88e5); color: white; padding: 2.5rem; border-radius: 16px; text-align: center; margin-bottom: 2rem; box-shadow: 0 8px 20px rgba(0,0,0,0.15); }
.hero-section h1 { font-size: 2.8rem; font-weight: 800; margin-bottom: 0.5rem; }
.hero-section p { font-size: 1.2rem; opacity: 0.9; }
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 2rem; margin: 2rem 0; }
.feature-box { background: #ffffff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.08); text-align: center; transition: transform 0.3s ease; }
.feature-box:hover { transform: translateY(-5px); }
.feature-box h4 { margin-top: 1rem; font-size: 1.2rem; color: #1565c0; }
.get-started-box { background-color: #e3f2fd; border: 1px solid #90caf9; border-radius: 14px; padding: 2rem; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-top: 2rem; }
.get-started-box h3 { font-size: 1.6rem; color: #0d47a1; margin-bottom: 0.5rem; }
.get-started-box .stButton button { background-color: #1565c0 !important; color: white !important; font-weight: 600; padding: 0.75rem 1.5rem; font-size: 1rem; border: none; border-radius: 8px; cursor: pointer; margin-top: 1rem; }
.get-started-box .stButton button:hover { background-color: #0d47a1 !important; }
.footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f8f9fa; color: #1f2937; text-align: center; padding: 10px; font-size: 0.85em; }
a { color: #0ea5e9; text-decoration: none; }
a:hover { text-decoration: underline; }
.debug-info { background-color: #fff8e1; padding: 15px; border-radius: 8px; margin-top: 15px; font-family: monospace; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# Initialize directories
upload_directory = "input_images"
signature_directory = "sign_images"
cheque_directory = "cheque_images"
for d in [upload_directory, signature_directory, cheque_directory]:
    if not os.path.exists(d): 
        os.makedirs(d)
        logging.info(f"Created directory: {d}")

# Initialize session state
if 'contact' not in st.session_state:
    st.session_state.contact = []
if 'cheque_data' not in st.session_state:
    st.session_state.cheque_data = {}
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'nav_page' not in st.session_state:
    st.session_state.nav_page = "Homepage"

# Run database migration once at startup
migrate_database()

# Navigation Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-container"><div class="sidebar-title">📘 Navigation</div>', unsafe_allow_html=True)
    def nav_button(label, key_name):
        is_active = st.session_state.nav_page == key_name
        if st.button(label, key=f"nav_{key_name}", use_container_width=True):
            st.session_state.nav_page = key_name
        class_name = "nav-button active" if is_active else "nav-button"
        st.markdown(f"""
            <script>
            const btn = window.parent.document.querySelector('[data-testid="stButton"][data-streamlit-key="nav_{key_name}"] button');
            if (btn) btn.className = "{class_name}";
            </script>
        """, unsafe_allow_html=True)
    nav_button("🏠 Homepage", "Homepage")
    nav_button("📞 Contact", "Contact")
    nav_button("📤 Upload", "Upload")
    nav_button("📊 Transactions", "Transactions")
    st.markdown("</div>", unsafe_allow_html=True)

# Page: Upload
if st.session_state.nav_page == "Upload":
    input_prompt = st.secrets['prompt1']['input_prompt'] or """
    Extract the following fields from the cheque images (front and back): rib1, rib2, num_cheque, nom, nomreciver, montant, plafond, date. 
    Ensure all numeric fields (e.g., montant, plafond) are returned as numbers. If a field is not detectable, return "Not Detected".
    """
    st.markdown("""
    <div style="text-align: center;">
    <div class="custom-header">📘 𝐂𝐡𝐞𝐪𝐮𝐞 𝐁𝐨𝐨𝐤 𝐄𝐱𝐭𝐫𝐚c𝐭𝐢𝐨𝐧 𝐮𝐬𝐢𝐧𝐠 𝐀𝐈</div>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("📤 Upload Cheque Images")
    col1, col2 = st.columns(2)
    with col1:
        front_image_file = st.file_uploader("🖼️ 𝑪𝒉𝒐𝒐𝒔𝒆 𝒇𝒓𝒐𝒏𝒕 𝒊𝒎𝒂𝒈𝒆 (𝒓𝒆𝒄𝒕𝒐)...", type=["jpg", "jpeg", "png"], key="front")
    with col2:
        back_image_file = st.file_uploader("🖼️ 𝑪𝒉𝒐𝒐𝒔𝒆 𝒃𝒂𝒄𝒌 𝒊𝒎𝒂𝒈𝒆 (𝒗𝒆𝒓𝒔𝒐)...", type=["jpg", "jpeg", "png"], key="back")
    zip_file = st.file_uploader("📦 𝑼𝒑𝒍𝒐𝒂𝒅 𝒂 𝒁𝑰𝑷 𝒇𝒊𝒍𝒆 𝒄𝒐𝒏𝒕𝒂𝒊𝒏𝒊𝒏𝒈 𝒇𝒓𝒐𝒏𝒕 𝒂𝒏𝒅 𝒃𝒂𝒄𝒌 𝒊𝒎𝒂𝒈𝒆𝒔", type=["zip"], key="zip")

    if front_image_file and back_image_file:
        front_image = Image.open(front_image_file)
        back_image = Image.open(back_image_file)
        st.subheader("🖼️ Uploaded Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(front_image, caption='Front image (recto)', use_container_width=True)
        with col2:
            st.image(back_image, caption='Back image (verso)', use_container_width=True)
        center_col = st.columns([1, 2, 1])[1]
        with center_col:
            if st.button("Extract Cheque Data", use_container_width=True):
                try:
                    with st.spinner("Extracting cheque data..."):
                        response = model.generate_content([input_prompt, front_image, back_image])
                        logging.info(f"Raw Gemini response: {response.text}")
                    
                    response_text = response.text.strip()
                    if response_text.startswith("```json") and response_text.endswith("```"):
                        response_text = response_text[7:-3].strip()
                    
                    data = json.loads(response_text)
                    st.session_state.cheque_data = data
                    
                    st.markdown("""
                    <div class='custom-success'>✅ <strong>Data extracted successfully!</strong></div>
                    """, unsafe_allow_html=True)
                    
                    rib1 = data.get("rib1", data.get("Rib1", data.get("RIB1", "Not Detected")))
                    rib2 = data.get("rib2", data.get("Rib2", data.get("RIB2", "Not Detected")))
                    plafond = data.get("plafond", "Not Detected")
                    bank = get_bank_from_rib(rib1)
                    
                    st.markdown(f"""
                    <div class="cheque-box">
                        <h3>🧾 Cheque Details</h3>
                        <ul>
                            <li><strong>🔢 Cheque No:</strong> {data.get('num_cheque', 'Not Detected')}</li>
                            <li><strong>🏢 Sender:</strong> {data.get('nom', 'Not Detected')}</li>
                            <li><strong>🏨 Receiver:</strong> {data.get('nomreciver', 'Not Detected')}</li>
                            <li><strong>💳 RIB 1:</strong> {rib1}</li>
                            <li><strong>🏦 RIB 2:</strong> {rib2}</li>
                            <li><strong>💰 Amount:</strong> {data.get('montant', 'Not Detected')}</li>
                            <li><strong>📅 Date:</strong> {data.get('date', 'Not Detected')}</li>
                            <li><strong>🏦 Plafond:</strong> {plafond}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    cheque_name = f"{data['num_cheque']}_cheque.jpg"
                    front_path = os.path.join(cheque_directory, f"front_{cheque_name}")
                    back_path = os.path.join(cheque_directory, f"back_{cheque_name}")
                    front_image.convert('RGB').save(front_path)
                    back_image.convert('RGB').save(back_path)
                    
                    cp_image = sharpen_image(preprocess_image(front_image))
                    sign_path = os.path.join(signature_directory, f"{data['num_cheque']}_signature.jpg")
                    if cp_image is not None:
                        Image.fromarray(cv2.cvtColor(cp_image, cv2.COLOR_GRAY2RGB)).save(sign_path)
                    
                    data.update({
                        "front_img": front_path,
                        "back_img": back_path,
                        "sign_img": sign_path if cp_image is not None else "",
                        "bank": bank
                    })
                    
                    if save_transaction_to_db(data):
                        logging.info("Transaction saved to database successfully")
                    else:
                        logging.warning("Failed to save transaction to database")
                    
                    analysis = analyze_transaction(data)
                    st.session_state.contact.append({**data, **{"analysis": analysis}})
                    
                    customer_explanation = ""
                    fraud_explanation = ""
                    consecutive_explanation = ""
                    
                    if analysis["potential_customer"]:
                        customer_explanation += "<li><strong>Why Potential Customer?</strong> This transaction meets the following criteria:</li>"
                        if analysis["analysis"].get("high_plafond", False) and analysis["analysis"].get("high_amount_ratio", False):
                            customer_explanation += "<ul><li>Plafond > 10,000 and Amount > 50% of Plafond.</li></ul>"
                        if analysis["analysis"].get("periodic_high_transactions", False):
                            customer_explanation += "<ul><li>Periodic transactions (>1) with Amount > 5,000.</li></ul>"
                    
                    if analysis["fraud_risk"]:
                        fraud_explanation += "<li><strong>Why Fraud Risk?</strong> This transaction meets the following criteria:</li>"
                        if analysis["analysis"].get("amount_exceeds_plafond", False):
                            fraud_explanation += "<ul><li style='color: red; font-weight: bold;'>🚨 The amount exceeds the plafond.</li></ul>"
                        if analysis["analysis"].get("near_plafond", False):
                            fraud_explanation += "<ul><li>Amount > 90% of Plafond.</li></ul>"
                        if analysis["analysis"].get("small_amounts_high_plafond", False):
                            fraud_explanation += "<ul><li>Plafond > 100,000, Amount < 1,000, and >5 transactions in history.</li></ul>"
                        if analysis["analysis"].get("successive_cheques_same_sender_increasing", False):
                            fraud_explanation += "<ul><li>Successive cheques (>1) from the same sender with consecutive numbers and increasing amounts.</li></ul>"
                    
                    if analysis["analysis"].get("consecutive_cheque_numbers", False):
                        consecutive_explanation += "<li><strong>⚠️ Transaction précédente avec numéro consécutif détectée :</strong></li>"
                        for detail in analysis["analysis"].get("consecutive_cheque_details", []):
                            consecutive_explanation += f"<ul><li>Chèque actuel : {detail['current_cheque']}, Chèque précédent : {detail['previous_cheque']} (différence : ±{detail['difference']}).</li></ul>"
                    else:
                        consecutive_explanation += "<li><strong>✅ Aucun chèque consécutif détecté.</strong></li>"
                    
                    st.markdown(f"""
                    <div class="cheque-box">
                        <h3>🔍 Transaction Analysis</h3>
                        <ul>
                            <li><strong>🏦 Bank:</strong> {bank}</li>
                            <li><strong>🎯 Potential Customer:</strong> {'Yes' if analysis['potential_customer'] else 'No'}</li>
                            {customer_explanation if customer_explanation else '<li>No specific criteria met.</li>'}
                            <li><strong>⚠️ Fraud Risk:</strong> <span style="color: {'red' if analysis['fraud_risk'] else 'green'}; font-weight: bold;">{'Yes' if analysis['fraud_risk'] else 'No'}</span></li>
                            {fraud_explanation if fraud_explanation else '<li>No specific criteria met.</li>'}
                            {consecutive_explanation}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse extracted data as JSON: {str(e)}")
                    logging.error(f"JSON decode error: {str(e)} - Response text: {response.text}")
                except Exception as e:
                    st.error(f"An error occurred during extraction: {str(e)}")
                    logging.error(f"Extraction error: {str(e)}")

    if zip_file is not None:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(upload_directory)
        st.markdown("""
        <div class='custom-success'>✅ <strong>Images have been uploaded successfully!</strong></div>
        """, unsafe_allow_html=True)
        if st.button("Process ZIP", use_container_width=True):
            with st.spinner("Processing ZIP file..."):
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    image_pairs = {}
                    
                    for img_file in image_files:
                        base_name = os.path.splitext(img_file)[0]
                        base_name = base_name.rsplit('_', 1)[0]
                        
                        if base_name not in image_pairs:
                            image_pairs[base_name] = {"front": None, "back": None}
                        
                        if img_file.lower().endswith(('_0.jpg', '_0.png', '_0.jpeg')):
                            image_pairs[base_name]["front"] = img_file
                        elif img_file.lower().endswith(('_1.jpg', '_1.png', '_1.jpeg')):
                            image_pairs[base_name]["back"] = img_file
                    
                    for base_name, pair in image_pairs.items():
                        if pair["front"] and pair["back"]:
                            try:
                                with zip_ref.open(pair["front"]) as front_img_file:
                                    front_image = Image.open(BytesIO(front_img_file.read()))
                                with zip_ref.open(pair["back"]) as back_img_file:
                                    back_image = Image.open(BytesIO(back_img_file.read()))
                                
                                with st.spinner(f"Processing {base_name}..."):
                                    response = model.generate_content([input_prompt, front_image, back_image])
                                    logging.info(f"Raw Gemini response for {base_name}: {response.text}")
                                    st.markdown(f"<div class='debug-info'>Raw Gemini Response for {base_name}:<br>{response.text}</div>", unsafe_allow_html=True)
                                
                                response_text = response.text.strip()
                                if response_text.startswith("```json") and response_text.endswith("```"):
                                    response_text = response_text[7:-3].strip()
                                
                                cheque_data = json.loads(response_text)
                                
                                rib1 = cheque_data.get("rib1", cheque_data.get("Rib1", cheque_data.get("RIB1", "Not Detected")))
                                
                                if rib1 != "Not Detected":
                                    cheque_name = f'{cheque_data["num_cheque"]}_cheque.jpg'
                                    front_path = os.path.join(cheque_directory, f"front_{cheque_name}")
                                    back_path = os.path.join(cheque_directory, f"back_{cheque_name}")
                                    front_image.convert('RGB').save(front_path)
                                    back_image.convert('RGB').save(back_path)
                                    
                                    cp_image = sharpen_image(preprocess_image(front_image))
                                    sign_path = os.path.join(signature_directory, f"{cheque_data['num_cheque']}_signature.jpg")
                                    if cp_image is not None:
                                        Image.fromarray(cv2.cvtColor(cp_image, cv2.COLOR_GRAY2RGB)).save(sign_path)
                                    
                                    bank = get_bank_from_rib(rib1)
                                    
                                    cheque_data.update({
                                        "front_img": front_path,
                                        "back_img": back_path,
                                        "sign_img": sign_path if cp_image is not None else "",
                                        "bank": bank
                                    })
                                    
                                    if save_transaction_to_db(cheque_data):
                                        logging.info(f"Saved transaction for cheque {cheque_data['num_cheque']}")
                                    else:
                                        logging.warning(f"Failed to save transaction for cheque {cheque_data['num_cheque']}")
                                    
                                    analysis = analyze_transaction(cheque_data)
                                    st.session_state.contact.append({**cheque_data, **{"analysis": analysis}})
                                    st.success(f"Processed cheque {cheque_data['num_cheque']}")
                                    
                                    customer_explanation = ""
                                    fraud_explanation = ""
                                    consecutive_explanation = ""
                                    
                                    if analysis["potential_customer"]:
                                        customer_explanation += "<li><strong>Why Potential Customer?</strong> This transaction meets the following criteria:</li>"
                                        if analysis["analysis"].get("high_plafond", False) and analysis["analysis"].get("high_amount_ratio", False):
                                            customer_explanation += "<ul><li>Plafond > 10,000 and Amount > 50% of Plafond.</li></ul>"
                                        if analysis["analysis"].get("periodic_high_transactions", False):
                                            customer_explanation += "<ul><li>Periodic transactions (>1) with Amount > 5,000.</li></ul>"
                                    
                                    if analysis["fraud_risk"]:
                                        fraud_explanation += "<li><strong>Why Fraud Risk?</strong> This transaction meets the following criteria:</li>"
                                        if analysis["analysis"].get("amount_exceeds_plafond", False):
                                            fraud_explanation += "<ul><li style='color: red; font-weight: bold;'>🚨 The amount exceeds the plafond.</li></ul>"
                                        if analysis["analysis"].get("near_plafond", False):
                                            fraud_explanation += "<ul><li>Amount > 90% of Plafond.</li></ul>"
                                        if analysis["analysis"].get("small_amounts_high_plafond", False):
                                            fraud_explanation += "<ul><li>Plafond > 100,000, Amount < 1,000, and >5 transactions in history.</li></ul>"
                                        if analysis["analysis"].get("successive_cheques_same_sender_increasing", False):
                                            fraud_explanation += "<ul><li>Successive cheques (>1) from the same sender with consecutive numbers and increasing amounts.</li></ul>"
                                    
                                    if analysis["analysis"].get("consecutive_cheque_numbers", False):
                                        consecutive_explanation += "<li><strong>⚠️ Transaction précédente avec numéro consécutif détectée :</strong></li>"
                                        for detail in analysis["analysis"].get("consecutive_cheque_details", []):
                                            consecutive_explanation += f"<ul><li>Chèque actuel : {detail['current_cheque']}, Chèque précédent : {detail['previous_cheque']} (différence : ±{detail['difference']}).</li></ul>"
                                    else:
                                        consecutive_explanation += "<li><strong>✅ Aucun chèque consécutif détecté.</strong></li>"
                                    
                                    st.markdown(f"""
                                    <div class="cheque-box">
                                        <h3>🔍 Transaction Analysis for Cheque {cheque_data['num_cheque']}</h3>
                                        <ul>
                                            <li><strong>🏦 Bank:</strong> {bank}</li>
                                            <li><strong>🎯 Potential Customer:</strong> {'Yes' if analysis['potential_customer'] else 'No'}</li>
                                            {customer_explanation if customer_explanation else '<li>No specific criteria met.</li>'}
                                            <li><strong>⚠️ Fraud Risk:</strong> <span style="color: {'red' if analysis['fraud_risk'] else 'green'}; font-weight: bold;">{'Yes' if analysis['fraud_risk'] else 'No'}</span></li>
                                            {fraud_explanation if fraud_explanation else '<li>No specific criteria met.</li>'}
                                            {consecutive_explanation}
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.warning(f"Skipped {base_name} - No RIB detected")
                            except json.JSONDecodeError:
                                st.error(f"Failed to parse data for {pair['front']} or {pair['back']}")
                                logging.error(f"JSON decode error for {base_name}")
                            except Exception as e:
                                st.error(f"Error processing {base_name}: {str(e)}")
                                logging.error(f"Error processing {base_name}: {str(e)}")
                        else:
                            st.warning(f"Missing front or back image for {base_name}")
                            logging.warning(f"Incomplete pair for {base_name}")
            
            st.markdown("""
            <div class='custom-success'>✅ <strong>Batch processing completed!</strong></div>
            """, unsafe_allow_html=True)

    if st.session_state.contact:
        center_col = st.columns([1, 2, 1])[1]
        with center_col:
            if st.button("Ready for Download", use_container_width=True):
                def create_zip_with_folder_and_file(zip_name, folder1, folder2, filename):
                    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        if os.path.exists(folder1):
                            for root, _, files in os.walk(folder1):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.join(os.path.basename(folder1), file)
                                    zipf.write(file_path, arcname)
                        if os.path.exists(folder2):
                            for root, _, files in os.walk(folder2):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.join(os.path.basename(folder2), file)
                                    zipf.write(file_path, arcname)
                        if os.path.exists(filename):
                            zipf.write(filename, os.path.basename(filename))

                def cleanup():
                    import time
                    max_attempts = 5
                    for directory in [upload_directory, cheque_directory, signature_directory]:
                        if os.path.exists(directory):
                            shutil.rmtree(directory, ignore_errors=True)
                            os.makedirs(directory)
                    if os.path.exists(excel_file_name):
                        os.remove(excel_file_name)
                    if os.path.exists(zip_output):
                        for attempt in range(max_attempts):
                            try:
                                os.remove(zip_output)
                                break
                            except PermissionError:
                                if attempt < max_attempts - 1:
                                    time.sleep(1)
                                else:
                                    st.warning(f"Could not delete {zip_output}. Please ensure it is not in use and try again.")
                    st.session_state.contact = []
                    st.session_state.cheque_data = {}

                df = pd.DataFrame(st.session_state.contact)
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

                if st.button("Cleanup Files", use_container_width=True):
                    cleanup()
                    st.success("Files cleaned up successfully!")
                    st.rerun()

# Page: Homepage
elif st.session_state.nav_page == "Homepage":
    st.markdown("""
    <div class="hero-section">
        <h1>Welcome to Cheque Book AI</h1>
        <p>Automate your cheque processing effortlessly with AI-driven precision.</p>
    </div>
    <div class="feature-grid">
        <div class="feature-box">
            <img src="https://img.icons8.com/color/64/000000/document--v1.png"/>
            <h4>Scan Cheques</h4>
            <p>Upload images and let AI extract key details from your cheques instantly.</p>
        </div>
        <div class="feature-box">
            <img src="https://img.icons8.com/ios-filled/50/1565c0/document--v1.png"/>
            <h4>Data Extraction</h4>
            <p>Parse RIBs, sender/receiver names, amounts, dates, and plafond with high accuracy.</p>
        </div>
        <div class="feature-box">
            <img src="https://img.icons8.com/50/1565c0/save--v1.png"/>
            <h4>Download Results</h4>
            <p>Get structured output as Excel files with cheque and signature images.</p>
        </div>
        <div class="feature-box">
            <img src="https://img.icons8.com/color/64/000000/security-checked.png"/>
            <h4>Fraud Detection</h4>
            <p>Analyze transactions for potential fraud risks and identify valuable customers.</p>
        </div>
    </div>
    <div style="display: flex; justify-content: center; margin-top: 3rem;">
        <div style="background: linear-gradient(135deg, #e3f2fd, #ffffff); padding: 2.5rem 2rem; border-radius: 20px; max-width: 850px; text-align: center; box-shadow: 0 10px 24px rgba(21, 101, 192, 0.15); border: 1px solid #bbdefb;">
            <img src="https://img.icons8.com/80/money.png" width="80" style="margin-bottom: 1rem;">
            <h3 style="color: #1565c0; font-size: 1.7rem; font-weight: 700;">Intelligent Cheque Processing</h3>
            <p style="font-size: 1.1rem; color: #1f2937; line-height: 1.6;">Our system is powered by state-of-the-art AI models designed to handle cheque formats, extract fields, and assist in digitizing banking workflows.<br>No manual input, no errors — just pure automation.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown("""
        <div class="get-started-box">
            <h3>Get Started Now!</h3>
            <p>Head to the Upload tab to begin processing your cheques.</p>
        """, unsafe_allow_html=True)
        button_col = st.columns([8, 2, 8])[1]
        with button_col:
            if st.button("Upload Cheques"):
                st.session_state.nav_page = "Upload"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("## 🤖 Ask About Cheque Book AI")
    user_input = st.text_input("Ask me about this platform or cheque transactions...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        bot_response = query_llama(user_input)
        st.session_state.chat_history.append(("bot", bot_response))
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.write("User: " + msg)
        else:
            st.write("Assistant: " + msg)

# Page: Contact
elif st.session_state.nav_page == "Contact":
    st.markdown("## 📬 Contact Us")
    st.write("Fill in the form below to leave us a message.")
    
    def reset_form():
        st.session_state.contact_name = ""
        st.session_state.contact_email = ""
        st.session_state.contact_message = ""
        st.session_state.contact_submitted = False
    
    # Initialize session state variables
    if "contact_name" not in st.session_state:
        st.session_state.contact_name = ""
    if "contact_email" not in st.session_state:
        st.session_state.contact_email = ""
    if "contact_message" not in st.session_state:
        st.session_state.contact_message = ""
    if "contact_submitted" not in st.session_state:
        st.session_state.contact_submitted = False
    
    # Display success message if form was submitted
    if st.session_state.contact_submitted:
        st.success("✅ Message sent successfully!")
        reset_form()
    
    # Create the form
    with st.form(key="contact_form"):
        st.markdown('<label style="font-weight:600;font-size:1.05rem;color:#0d47a1;">👤 Your Name</label>', unsafe_allow_html=True)
        name = st.text_input("", value=st.session_state.contact_name, key="contact_name")
        st.markdown('<label style="font-weight:600;font-size:1.05rem;color:#0d47a1;">📧 Your Email</label>', unsafe_allow_html=True)
        email = st.text_input("", value=st.session_state.contact_email, key="contact_email")
        st.markdown('<label style="font-weight:600;font-size:1.05rem;color:#0d47a1;">💬 Your Message</label>', unsafe_allow_html=True)
        message = st.text_area("", value=st.session_state.contact_message, key="contact_message")
        submitted = st.form_submit_button("📨 Send Message")
        
        if submitted:
            if not name or not email or not message:
                st.warning("⚠️ Please fill out all fields.")
            else:
                with open("contact_messages.csv", "a", newline="", encoding="utf-8") as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow([datetime.now().isoformat(), name, email, message])
                st.session_state.contact_submitted = True
                st.rerun()

# Page: Transactions
elif st.session_state.nav_page == "Transactions":
    st.markdown("## 📊 Transactions History")
    
    if st.button("🗑️ Vider la Base de Données", type="primary", 
                help="Cette action supprimera définitivement toutes les transactions"):
        try:
            conn = sqlite3.connect("transactions.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions")
            conn.commit()
            conn.close()
            st.success("Base de données vidée avec succès!")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de la suppression des données: {str(e)}")
            logging.error(f"Error clearing database: {str(e)}")
    
    try:
        conn = sqlite3.connect("transactions.db")
        df = pd.read_sql("SELECT * FROM transactions", conn)
        conn.close()
        
        if not df.empty:
            st.dataframe(df)
            
            for _, row in df.iterrows():
                analysis = analyze_transaction(row.to_dict())
                
                customer_explanation = ""
                fraud_explanation = ""
                consecutive_explanation = ""
                
                if analysis["potential_customer"]:
                    customer_explanation += "<li><strong>Why Potential Customer?</strong> This transaction meets the following criteria:</li>"
                    if analysis["analysis"].get("high_plafond", False) and analysis["analysis"].get("high_amount_ratio", False):
                        customer_explanation += "<ul><li>Plafond > 10,000 and Amount > 50% of Plafond.</li></ul>"
                    if analysis["analysis"].get("periodic_high_transactions", False):
                        customer_explanation += "<ul><li>Periodic transactions (>1) with Amount > 5,000.</li></ul>"
                
                if analysis["fraud_risk"]:
                    fraud_explanation += "<li><strong>Why Fraud Risk?</strong> This transaction meets the following criteria:</li>"
                    if analysis["analysis"].get("amount_exceeds_plafond", False):
                        fraud_explanation += "<ul><li style='color: red; font-weight: bold;'>🚨 The amount exceeds the plafond.</li></ul>"
                    if analysis["analysis"].get("near_plafond", False):
                        fraud_explanation += "<ul><li>Amount > 90% of Plafond.</li></ul>"
                    if analysis["analysis"].get("small_amounts_high_plafond", False):
                        fraud_explanation += "<ul><li>Plafond > 100,000, Amount < 1,000, and >5 transactions in history.</li></ul>"
                    if analysis["analysis"].get("successive_cheques_same_sender_increasing", False):
                        fraud_explanation += "<ul><li>Successive cheques (>1) from the same sender cheque with consecutive cheque numbers and increasing amounts.</li></ul>"
                
                if analysis["analysis"].get("consecutive_cheque_numbers", False):
                    consecutive_explanation += "<li><strong>⚠️ Transaction précédente avec numéro consécutif détectée :</strong></li>"
                    for detail in analysis["analysis"].get("consecutive_cheque_details", []):
                        consecutive_explanation += f"<ul><li>Chèque actuel : {detail['current_cheque']}, Chèque précédent : {detail['previous_cheque']} (différence : ±{detail['difference']}).</li></ul>"
                else:
                    consecutive_explanation += "<li><strong>✅ Aucun chèque consécutif détecté.</strong></li>"
                
                st.markdown(f"""
                <div class="cheque-box">
                    <h3>🧾 Cheque {row['num_cheque']}</h3>
                    <ul>
                        <li><strong>🔢 Cheque No:</strong> {row['num_cheque']}</li>
                        <li><strong>🏢 Sender:</strong> {row['nom']}</li>
                        <li><strong>🏨 Receiver:</strong> {row['nomreciver']}</li>
                        <li><strong>💳 RIB:</strong> {row['rib1']} <em>(Bank: {row['bank']})</em></li>
                        <li><strong>🏦:</strong> {row['rib2']}</li>
                        <li><strong>💰 Amount:</strong> {row['montant']}</li>
                        <li><strong>📅 Date:</strong> {row['date']}</li>
                        <li><strong>🏦 Bank:</strong> {row['bank']}</li>
                        <li><strong>🏦 Plafond:</strong> {row['plafond']}</li>
                        <li><strong>🎯 Potential Customer:</strong> {'Yes' if analysis['potential_customer'] else 'No'}</li>
                        {customer_explanation if customer_explanation else '<li>No specific criteria met.</li>'}
                        <li><strong>⚠️ Fraud Risk:</strong> <span style="color: {'red' if analysis['fraud_risk'] else 'green'}; font-weight: bold;">{'Yes' if analysis['fraud_risk'] else 'No'}</span></li>
                        {fraud_explanation if fraud_explanation else '<li>No specific criteria met.</li>'}
                        {consecutive_explanation}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucune transaction trouvée dans la base de données.")
    except sqlite3.Error as e:
        st.error(f"Erreur de base de données: {str(e)}")
        logging.error(f"Database error: {str(e)}")
    except Exception as e:
        st.error(f"Erreur lors du chargement des transactions: {str(e)}")
        logging.error(f"Error loading transactions: {str(e)}")

st.markdown("""
<div class="footer">
    Made with 💡 by Ons
</div>
""", unsafe_allow_html=True)