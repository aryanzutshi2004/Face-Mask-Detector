# 😷 Face Mask Detection  

A **deep learning project** that detects whether a person is wearing a **mask** or **not** using a **ResNet-50** model fine-tuned with **transfer learning**.  
This project integrates **FastAPI** for serving predictions as a REST API and **Streamlit** for an interactive web interface, making it an end-to-end deployable solution.  

---

## 🚀 Features  

✅ Detects if a person is **wearing a mask** or **not**  
✅ **FastAPI** backend for REST-based inference  
✅ **Streamlit** frontend for intuitive image upload and visualization  
✅ **Transfer Learning** using pretrained **ResNet-50**  
✅ **Optuna** for hyperparameter optimization  
✅ **Data Augmentation** and **WeightedRandomSampler** for handling class imbalance  
✅ GPU-compatible training (CUDA support)  
✅ Modular and clean project structure  

---

## 🧠 Model Overview  

- **Architecture:** ResNet-50 pretrained on ImageNet  
- **Framework:** PyTorch  
- **Dataset:** [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)  
- **Techniques Used:**  
  - Data Augmentation  
  - Transfer Learning  
  - WeightedRandomSampler  
  - Optuna Hyperparameter Tuning  
- **Goal:** Classify input images into two classes:  
  - `with_mask` 😷  
  - `without_mask` 🧑  

---

## 📊 Model Performance  

| Metric | With Mask | Without Mask | Overall |
|:------:|:----------:|:-------------:|:---------:|
| **Precision** | 0.96 | 0.99 | **0.97** |
| **Recall** | 0.99 | 0.96 | **0.97** |
| **F1-Score** | 0.97 | 0.97 | **0.97** |
| **Accuracy** | — | — | **97%** |

✅ Balanced and consistent performance across both classes, confirming the effectiveness of **WeightedRandomSampler** and **Optuna tuning**.  

---

## 📂 Project Structure  

```
FaceMaskDetection/
│
├── artifacts/                        # Trained models and checkpoints
│   ├── face_mask_model.pth           # Final trained model
│   └── model_2.pth                   # Alternative model version
│
├── fast api server/                  # Backend (FastAPI)
│   ├── __pycache__/                  
│   ├── model_helper.py               # Utilities for model loading & prediction
│   └── server.py                     # FastAPI main server file (API endpoints)
│
├── streamlit/                        # Frontend (Streamlit UI)
│   └── app.py                        # Streamlit app for image upload and prediction
│
└── README.md                         # Project documentation
```

---

## ⚙️ How It Works  

1. **Streamlit App** – User uploads an image (face photo).  
2. **FastAPI Server** – Receives the image and passes it to the PyTorch model.  
3. **Model Inference** – Model predicts `with_mask` or `without_mask`.  
4. **Result Display** – Streamlit displays the prediction and confidence score beautifully.  

---

## 🧩 Example Flow  

| Step | Description |  
|------|--------------|  
| 🖼️ Upload Image | User uploads an image via Streamlit UI. |  
| 🔁 API Request | Image is sent to FastAPI backend for inference. |  
| 🧠 Model Prediction | The model predicts whether the person is wearing a mask. |  
| 📊 Display Result | Streamlit shows the predicted class & confidence score. |  

---

## 🧪 Setup Instructions  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<yourusername>/FaceMaskDetection.git
cd FaceMaskDetection
```

### 2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run FastAPI backend  
```bash
cd "fast api server"
uvicorn server:app --reload
```

### 4️⃣ Run Streamlit frontend  
```bash
cd ../streamlit
streamlit run app.py
```

✅ Open your browser → [http://localhost:8501](http://localhost:8501) to access the app.

---

## 📈 Results Visualization  

You can visualize:
- Confusion Matrix  
- Classification Report  
- Accuracy and Loss Curves during training  

(Optional: Add your plots here once available 📊)

---

## 🏁 Future Improvements  

🚀 Add real-time webcam detection  
📈 Integrate Grad-CAM for model interpretability  
☁️ Deploy on Render / Hugging Face / AWS Lambda  
💬 Add multilingual interface support  

---

## 💡 Acknowledgments  

- Dataset: [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)  
- Pretrained Backbone: **ResNet-50** (Torchvision)  
- Tools Used: **PyTorch**, **Optuna**, **Streamlit**, **FastAPI**  
