# Plant Leaf Disease Detection 

A deep learning-based plant leaf disease detection system built using **TensorFlow**, **Keras**, and **ResNet50 Transfer Learning**. The model is trained on the PlantVillage dataset and can classify leaf images into **38 different disease and healthy plant categories**.


## 📌 Features

* Transfer Learning using **ResNet50**
* Data Augmentation for improved generalization
* Fine-tuning of pretrained layers
* Early Stopping to prevent overfitting
* Learning Rate Scheduling
* Model Checkpointing
* Supports classification of **38 plant disease classes**


## 📂 Dataset

Dataset used:

* **PlantVillage Dataset**
* Kaggle: [PlantVillage Dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)

The dataset contains images of healthy and diseased plant leaves belonging to multiple crop species.


## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Google Colab
* Kaggle API
  

## 📁 Project Structure

```text
Leaf-Disease-Detection/
│
├── leaf_disease_detection.ipynb
├── requirements.txt
└── README.md
```
  


# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Leaf-Disease-Detection.git
cd Leaf-Disease-Detection
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the beginning of your terminal prompt.


## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## 4. Configure Kaggle API (For Training)

Download your `kaggle.json` API key from your Kaggle account and place it in the project root directory.

### Linux / macOS

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Windows

Create the folder:

```text
C:\Users\<YourUsername>\.kaggle
```

and place `kaggle.json` inside it.


## 5. Download the Dataset

```bash
kaggle datasets download -d mohitsingh1804/plantvillage
unzip plantvillage.zip
```

After extraction, ensure the dataset structure is:

```text
PlantVillage/
├── train/
└── val/
```


## 6. Train the Model

Run the training notebook or Python script to train and save the model:

```bash
python train.py
```

The trained model will be saved as:

```text
final_leaf_disease_model.keras
```


## 7. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.


## 8. Deactivate the Virtual Environment

When you're done:

```bash
deactivate
```


If your repository already contains the trained model (`final_leaf_disease_model.keras`), you can skip **Steps 4–6** and directly run:

```bash
streamlit run app.py
```

after installing the dependencies.


## 🧠 Model Architecture

### Base Model

* ResNet50 (ImageNet Weights)
* `include_top=False`
* Input Size: `224 × 224 × 3`

### Custom Classification Head

```text
ResNet50
    ↓
GlobalAveragePooling2D
    ↓
Dense (512, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (38, Softmax)
```


## 🔄 Training Strategy

### Phase 1: Feature Extraction

* Freeze all ResNet50 layers
* Train custom classification head
* Learning Rate: `0.001`

### Phase 2: Fine Tuning

* Unfreeze last 50 layers of ResNet50
* Freeze earlier layers
* Learning Rate: `0.00001`


## 📈 Data Augmentation

The following augmentations are applied during training:

```python
rotation_range=20
zoom_range=0.2
shear_range=0.2
horizontal_flip=True
```


## ⚙️ Callbacks Used

### EarlyStopping

```python
monitor='val_loss'
patience=5
restore_best_weights=True
```

### ReduceLROnPlateau

```python
factor=0.2
patience=2
min_lr=1e-7
```

### ModelCheckpoint

```python
save_best_only=True
monitor='val_accuracy'
```


## 🏋️ Training

### Initial Training

```python
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=7,
    callbacks=callbacks
)
```

### Fine-Tuning

```python
history_fine = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    callbacks=callbacks
)
```


## 💾 Saved Models

### Best Validation Model

```text
best_resnet50.keras
```

### Final Fine-Tuned Model

```text
final_leaf_disease_model.keras
```


## 🔮 Future Improvements

* Deploy as a web application 
* Mobile deployment using TensorFlow Lite
* Disease severity estimation
* Treatment and pesticide recommendation system
* Real-time detection from camera feed


## 📊 Results

The model leverages transfer learning from ResNet50 and fine-tuning to achieve high classification accuracy on the PlantVillage dataset.


## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request


## 📜 License

This project is intended for educational and research purposes.