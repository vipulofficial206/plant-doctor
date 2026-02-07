# 🌿 Plant Doctor – AI Based Plant Disease Detection & Advisory System

## 📌 Project Overview

Plant Doctor is an Artificial Intelligence powered application that detects plant diseases from leaf images and provides treatment recommendations. The system uses a Convolutional Neural Network (CNN) trained on plant leaf images to classify diseases and integrates a chatbot with a knowledge base to suggest remedies and preventive measures.

The project aims to help farmers, gardeners, and agriculture students quickly identify plant health problems and take appropriate action.

---

## 🎯 Objectives

* Detect plant diseases automatically using image classification
* Provide treatment suggestions
* Offer preventive care tips
* Assist farmers and gardeners in plant health monitoring

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* Convolutional Neural Networks (CNN)
* FAISS (Knowledge retrieval system)
* Jupyter Notebook
* Chatbot interface

---

## 📂 System Components

1. **CNN Model** – Identifies plant disease from leaf image
2. **Knowledge Base** – Stores disease information and remedies
3. **FAISS Index** – Retrieves relevant plant care information
4. **Chatbot** – Provides user-friendly advice and explanations

---

## ⚙️ System Workflow

1. User uploads a plant leaf image
2. CNN model processes the image
3. Disease is predicted
4. Knowledge base is queried
5. Chatbot provides:

   * Disease name
   * Causes
   * Treatment
   * Prevention tips

---

## 📁 Project Structure

```
plant-doctor/
│── ModelTraining.ipynb       # CNN training notebook
│── train.ipynb               # Image classification training
│── modeltrain2.ipynb
│── plantchatbot.py           # Chatbot interface
│── build_faiss_index.py      # Knowledge base indexing
│── KnowledgeBase/            # Plant disease information
│── 1.keras                   # Trained model
│── training_history.csv      # Training metrics
```

---

## 🚀 How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/vipulofficial206/plant-doctor.git
cd plant-doctor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Chatbot

```bash
python plantchatbot.py
```

### 4. Predict Disease

Upload a plant leaf image and the system will classify the disease and provide treatment advice.

---## 🌐 Web Interface (Frontend Application)

A web-based user interface is also available for this project.
The frontend allows users to upload plant leaf images and interact with the AI model through a browser.

👉 Frontend Repository:
[https://github.com/vipulofficial206/plant-doctor-frontend](https://github.com/vipulofficial206/plant-doctor-frontend)

The frontend:

* Sends the uploaded image to the backend API
* Receives the disease prediction
* Displays diagnosis and treatment recommendations in a user-friendly interface

This makes the project a complete AI application (Machine Learning backend + Web frontend).

## 📊 Output

The system provides:

* Predicted disease name
* Description
* Causes
* Treatment suggestions
* Preventive care tips

---

## 💡 Applications

* Smart farming
* Home gardening
* Agricultural education
* Early disease detection

---

## 🔮 Future Improvements

* Mobile app integration
* Real-time camera detection
* Support for more crop types
* Multilingual chatbot (local languages)
* Cloud deployment

---

## 👨‍💻 Author

Vipul

---

## ⭐ Conclusion

This project demonstrates the application of Deep Learning and Computer Vision in agriculture. It can be extended into a smart farming assistant that helps improve crop health and productivity.
