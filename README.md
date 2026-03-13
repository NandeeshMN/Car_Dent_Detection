Vehicle Dent Detection using Deep Learning
This project detects dents on vehicle surfaces using Deep Learning and Computer Vision.
A Convolutional Neural Network built with TensorFlow and MobileNetV2 Transfer Learning is trained to classify vehicle images as dented or non-dented.

🚀 Features
Automated dent detection from vehicle images
Transfer learning using MobileNetV2
Image preprocessing using OpenCV
Model evaluation and prediction tools
Training performance visualization (accuracy & loss graphs)
Simple GUI for testing dent detection
🛠️ Tech Stack
Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
📂 Project Structure
ML_PROJECT
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
├── tf_env/ (virtual environment)
│
├── train_model.py
├── evaluate_model.py
├── show_training_graphs.py
├── dent_gui.py
└── .gitignore
⚙️ Installation
Clone the repository:

git clone https://github.com/yourusername/vehicle-dent-detection.git
cd vehicle-dent-detection
Install required dependencies:

pip install tensorflow opencv-python numpy matplotlib
🧠 Model Training
Run the training script:

python train_model.py
This will:

Train the CNN model
Save the trained model in the models/ folder
Save training history for visualization
📊 Model Evaluation
To evaluate the trained model:

python evaluate_model.py
📈 Visualize Training Graphs
To display training performance graphs:

python show_training_graphs.py
This will show:

Accuracy vs Epochs
Loss vs Epochs
🖥️ GUI Dent Detection
Run the GUI application:

python dent_gui.py
Upload a vehicle image and the model will predict whether the vehicle surface contains a dent.

📌 Applications
Vehicle damage inspection
Insurance claim automation
Smart vehicle maintenance systems
Automated quality inspection
🔮 Future Improvements
Multi-class damage detection (scratches, cracks, dents)
Real-time detection using webcam
Deploy as a web application
Improve model accuracy with larger datasets
📜 License
This project is for educational and research purposes.
