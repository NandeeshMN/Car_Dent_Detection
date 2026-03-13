# 🚗 Vehicle Dent Detection using Deep Learning

This project detects **dents on vehicle surfaces** using **Deep Learning and Computer Vision**.
A **Convolutional Neural Network (CNN)** built with **TensorFlow and MobileNetV2 Transfer Learning** is trained to classify vehicle images as **dented** or **non-dented**.

---

## 🚀 Features

- Automated dent detection from vehicle images  
- Transfer learning using **MobileNetV2**  
- Image preprocessing using **OpenCV**  
- Model evaluation and prediction tools  
- Training performance visualization (**accuracy & loss graphs**)  
- Simple **GUI application** for testing dent detection  

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**

---

## 📂 Project Structure

```text
ML_Project
│
├── dataset/
│   ├── train/                 
│   └── test/                  
│
├── models/
│   ├── class_mapping.json     
│   ├── dent_model_*.keras     
│   └── training_history.pkl   
│
├── tf_env/                    
│
├── train_model.py             
├── evaluate_model.py          
├── show_training_graphs.py    
├── dent_gui.py                
└── .gitignore                 
```


---

## ⚙️ Installation

### Clone the Repository
git clone https://github.com/yourusername/Car_Dent_Detection.git
cd vehicle-dent-detection
### Install Dependencies
pip install tensorflow opencv-python numpy matplotlib

---
### 🧠 Model Training

### Run the training script:

python train_model.py

### This process will:

- Train the CNN model
- Save the trained model inside the models/ folder
- Store the training history for visualization
---
### 📊 Model Evaluation

### To evaluate the trained model:

python evaluate_model.py

This script evaluates the model performance on the test dataset.

---

### 📈 Visualize Training Graphs

### To display the training performance graphs:

python show_training_graphs.py

### This will generate:

- Accuracy vs Epochs
- Loss vs Epochs

---

### 🖥️ GUI Dent Detection

### Run the GUI application:

python dent_gui.py

### Steps:

- Launch the GUI

- Upload a vehicle image

- The model predicts whether the vehicle surface contains a dent or not

---

### 📌 Applications

- Vehicle damage inspection systems

- Insurance claim automation

- Smart vehicle maintenance systems

- Automated quality inspection in manufacturing

---

### 🔮 Future Improvements

- Multi-class damage detection (scratches, cracks, dents)

- Real-time detection using webcam

- Deploy as a web application

- Improve model accuracy using larger datasets
---
### 📜 License

This project is created for educational and research purposes.
