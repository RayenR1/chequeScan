# 📋 Cheque Book AI - Système Intelligent de Traitement de Chèques

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![Google AI](https://img.shields.io/badge/Google%20AI-Gemini-green.svg)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Vue d'ensemble

**Cheque Book AI** est une solution avancée de traitement automatisé de chèques utilisant l'intelligence artificielle. Le système permet d'extraire automatiquement les informations cruciales des chèques, de détecter les risques de fraude et d'identifier les clients potentiels grâce à des algorithmes d'analyse sophistiqués.

### ✨ Fonctionnalités Principales

- **🤖 Extraction Intelligente** : Utilisation de Google Gemini AI pour extraire automatiquement les champs des chèques
- **📸 Traitement d'Images** : Support des formats JPG, PNG avec préprocessing OpenCV
- **✍️ Isolation de Signatures** : Extraction et sauvegarde automatique des signatures
- **🔍 Détection de Fraude** : Analyse des transactions avec alertes en temps réel
- **👥 Identification de Clients** : Détection automatique des clients potentiels
- **📊 Analyse Transactionnelle** : Historique et analyse des patterns de transactions
- **💾 Base de Données** : Stockage SQLite des transactions avec migration automatique
- **📁 Export Complet** : Génération de fichiers ZIP avec images et rapports Excel
- **🤖 Assistant Virtuel** : ChequeBot pour assistance et analyse des données

## 🏗️ Architecture Technique

### Technologies Utilisées

- **Backend** : Python 3.8+, Streamlit
- **Intelligence Artificielle** : Google Gemini 2.0 Flash, Ollama
- **Traitement d'Images** : OpenCV, PIL/Pillow
- **Base de Données** : SQLite3
- **Interface Utilisateur** : Streamlit avec design moderne
- **Export de Données** : pandas, openpyxl

### Structure du Projet

```
chequeScan/
├── app.py                      # Application principale Streamlit
├── app2.py                     # Version alternative de l'application
├── app3.py                     # Version de développement
├── crop.py                     # Utilitaires de traitement d'images
├── requirements.txt            # Dépendances Python
├── .env                        # Variables d'environnement (API keys)
├── transactions.db             # Base de données SQLite
├── documents.db                # Base de données des documents
├── cheque_table.xlsx           # Export Excel des données
├── contact_messages.csv        # Messages de contact
├── cheque_images/              # Images de chèques traitées
├── document_images/            # Images de documents
├── sign_images/                # Signatures extraites
├── input_images/               # Images d'entrée
└── cheque tests/               # Images de test
```

## 🚀 Installation et Configuration

### Prérequis

- Python 3.8 ou supérieur
- Clé API Google Gemini
- Environnement Windows/Linux/macOS

### Installation

1. **Cloner le projet**
```bash
git clone <repository-url>
cd chequeScan
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Configuration des variables d'environnement**
```bash
# Créer un fichier .env dans le répertoire racine
echo "GEMINI_API_KEY=votre_cle_api_gemini" > .env
```

4. **Lancer l'application**
```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

## 📖 Guide d'Utilisation

### 1. Upload de Chèques

- **Format supporté** : JPG, PNG
- **Types d'upload** : 
  - Images individuelles (recto/verso)
  - Fichiers ZIP pour traitement par lots
- **Préprocessing automatique** : Amélioration de la qualité d'image

### 2. Extraction de Données

Le système extrait automatiquement :
- Nom de l'expéditeur et du bénéficiaire
- RIB expéditeur et bénéficiaire
- Numéro de chèque
- Date d'émission
- Montant en chiffres et lettres
- Plafond du compte
- Code banque avec identification automatique

### 3. Analyse de Fraude

#### Critères de Détection :
- **Dépassement de plafond** : Montant > Plafond
- **Proximité du plafond** : Montant > 90% du plafond
- **Chèques consécutifs suspects** : Numéros consécutifs avec montants croissants
- **Petits montants/gros plafonds** : Montant < 1000 avec plafond > 100,000

#### Alertes Automatiques :
- 🔴 **Alerte Rouge** : Dépassement de plafond
- 🟡 **Alerte Jaune** : Risque modéré détecté

### 4. Identification de Clients Potentiels

#### Critères d'Identification :
- **Gros plafonds** : Plafond > 10,000 avec montant > 50% du plafond
- **Transactions périodiques élevées** : Transactions récurrentes > 5,000€

### 5. Assistant ChequeBot

L'assistant IA peut :
- Répondre aux questions sur la plateforme
- Analyser l'historique des transactions
- Expliquer les alertes de fraude
- Fournir des détails sur les banques tunisiennes

## 🏦 Codes Bancaires Supportés

Le système reconnaît automatiquement les banques tunisiennes :

- **01** : Arab Tunisian Bank (ATB)
- **03** : Banque de Tunisie (BT)
- **04** : Attijari Bank
- **05** : Banque Tuniso-Koweitienne (BTK)
- **08** : Banque Internationale Arabe de Tunisie (BIAT)
- **10** : Société Tunisienne de Banque (STB)
- Et 10+ autres banques...

## 📊 Fonctionnalités d'Export

### Formats d'Export :
- **Excel** : Tableau complet des transactions avec analyses
- **ZIP** : Archive contenant :
  - Images des chèques (recto/verso)
  - Signatures extraites
  - Fichier Excel des données
  - Logs d'analyse

### Données Exportées :
- Informations complètes des chèques
- Résultats d'analyse de fraude
- Scores de clients potentiels
- Métadonnées des images

## 🔧 Configuration Avancée

### Variables d'Environnement

```env
GEMINI_API_KEY=your_gemini_api_key_here
LOG_LEVEL=INFO
DATABASE_PATH=transactions.db
```

### Personnalisation

Le système permet la personnalisation de :
- Seuils de détection de fraude
- Critères d'identification de clients
- Templates d'export
- Messages de l'assistant

## 📝 Logging et Monitoring

- **Fichiers de logs** : `chequebot.log`, `cheque_app.log`
- **Niveau de logging** : INFO par défaut
- **Monitoring** : Suivi des performances d'extraction
- **Audit Trail** : Historique complet des opérations

## 🛠️ Maintenance et Dépannage

### Problèmes Courants :

1. **Erreur d'API Gemini** : Vérifier la clé API dans `.env`
2. **Images non reconnues** : S'assurer de la qualité des images
3. **Base de données corrompue** : Utiliser la fonction de réinitialisation
4. **Performance lente** : Réduire la taille des images d'entrée

### Mise à Jour de la Base de Données :

Le système inclut une migration automatique pour maintenir la compatibilité.

## 🔐 Sécurité et Confidentialité

- **Stockage local** : Toutes les données restent sur votre machine
- **Chiffrement** : Communications sécurisées avec les APIs
- **Audit** : Logs complets des accès et modifications
- **Nettoyage** : Suppression automatique des fichiers temporaires

## 🤝 Contribution

Pour contribuer au projet :

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les modifications (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.



## 🎖️ Auteurs

Développé avec ❤️ par l'équipe Cheque Book AI

---

**Cheque Book AI** - Révolutionnant le traitement bancaire avec l'intelligence artificielle
