# MRI Brain Tumor Detection — DL Demo

## Table of Contents
1.  Company Profile
2.  Introduction to the Project
3.  Scope of Work
4.  Existing System and Need for Proposed System
5.  Facilities Required for the Project
     i.  Hardware Requirements
     ii. Software Requirements
6.  Objectives
7.  User Requirements
8.  Methodology
9.  Prototyping
10. System Features
     i.  Module Specifications
     ii. DFD (Level 0 & Level 1)
     iii. ER Diagram
     iv. System Flow Chart
     v.  Database Layout
     vi. Data Dictionary
     vii.User Interfaces
11. Project Structure & Code Sheet
12. Challenges and Risks
13. Future Scope
14. Conclusion
15. References

---

1. Company Profile

    - **Name:** Academic Demo / Research Project
    - **Domain:** Medical Image Analysis, Deep Learning
    - **Purpose:** Provide an educational demonstration of MRI brain tumor classification using transfer learning.

2. Introduction to the Project

    This project implements a deep learning based pipeline to classify MRI brain images into tumor categories (glioma, meningioma, pituitary, and no tumor) using transfer learning (VGG16 backbone) and standard image preprocessing techniques.

3. Scope of Work

    - Build an image preprocessing pipeline (augmentation, resizing, normalization).
    - Train and evaluate a transfer-learning model for multi-class classification.
    - Provide a simple web UI to upload MRI images and display predictions.
    - Package a trained model (`models/model.h5`) for inference.

4. Existing System and Need for Proposed System

    - Existing manual inspection by radiologists is time-consuming and subjective.
    - Proposed system offers fast, repeatable preliminary triage to assist clinicians and to support educational experiments.

5. Facilities Required for the Project

    i. Hardware Requirements

        - CPU: Quad-core recommended for development.
        - GPU: NVIDIA GPU with at least 4GB VRAM (optional but recommended for training).
        - RAM: 8GB+ recommended.
        - Disk: ~5GB free for datasets, models, and environment.

    ii. Software Requirements

        - OS: Windows / Linux / macOS
        - Python 3.8+ (see `requirements.txt`)
        - Libraries: TensorFlow / Keras, OpenCV / Pillow, Flask (for the web app)

6. Objectives

    - Accurately classify MRI images into predefined tumor categories.
    - Demonstrate transfer learning with VGG16.
    - Provide an easy-to-use web interface for inference.

7. User Requirements

    - Users should be able to upload an MRI image and receive a predicted class with confidence score.
    - Admin users should be able to view prediction logs and manage uploads (basic web UI provided in `templates/admin/`).

8. Methodology

    - Data collection: images grouped by class under `mri_sample_images/`.
    - Preprocessing: resize, normalize, augment (rotation, flips, shifts).
    - Model: transfer learning using VGG16 as feature extractor, custom classification head, trained with categorical crossentropy.
    - Inference: load `models/model.h5` and run prediction on uploaded images.

9. Prototyping

    - Rapid prototyping performed in `MRI_Brian_Tumor_Detection_DL.ipynb` and `test_prediction.py`.
    - Iteratively improved preprocessing and model head based on validation performance.

10. System Features

    i. Module Specifications

        - `main.py` / `app.py`: Flask app for web UI and inference endpoints.
        - `models/`: stores trained model files (e.g., `model.h5`).
        - `uploads/`: stores uploaded images for prediction.
        - `templates/` and `assets/`: web UI templates and styling.

    ii. DFD (Level 0 & Level 1)

        - Placeholders: Add DFD images under `assets/` or `docs/` and link them here.

    iii. ER Diagram

        - Not required for this demo (minimal data persistence). If implemented, add diagram under `assets/`.

    iv. System Flow Chart

        - Flow: Upload Image -> Preprocess -> Model Inference -> Display Result -> (Optional) Store Log

    v. Database Layout

        - The demo uses file-based storage. For production, use a small relational DB with tables: `users`, `predictions`.

    vi. Data Dictionary

        - `image_path` (string): path to uploaded image.
        - `predicted_class` (string): model output label.
        - `confidence` (float): probability of predicted label.

    vii. User Interfaces

        - Web upload page: `templates/index.html` — upload image and show prediction.
        - Admin pages: `templates/admin/` for basic user and prediction views.

11. Project Structure & Code Sheet

    - `app.py` or `main.py` : Flask application entry point.
    - `models/model.h5` : trained Keras model used for inference.
    - `MRI_Brian_Tumor_Detection_DL.ipynb` : exploratory notebook used for prototyping/training.
    - `test_prediction.py` : small script to test model predictions.
    - `templates/`, `assets/`, `uploads/` : web UI and static files.

    Project structure (tree):

    ```
    MRI_Brian_Tumor_Detection_DL_Demo/
    ├─ app.py
    ├─ main.py
    ├─ MRI_Brian_Tumor_Detection_DL.ipynb
    ├─ Procfile
    ├─ README.md
    ├─ requirements.txt
    ├─ test_prediction.py
    ├─ __pycache__/
    ├─ assets/
    │  ├─ style.css
    │  └─ icons/
    │     └─ industries/
    ├─ instance/
    ├─ models/
    │  └─ model.h5
    ├─ mri_sample_images/
    ├─ templates/
    │  ├─ index.html
    │  ├─ login.html
    │  ├─ profile.html
    │  ├─ register.html
    │  ├─ admin/
    │  │  ├─ dashboard.html
    │  │  ├─ predictions.html
    │  │  └─ users.html
    │  └─ pages/
    │     ├─ about.html
    │     ├─ blogs.html
    │     ├─ case-studies.html
    │     ├─ faqs.html
    │     ├─ industries.html
    │     ├─ know-more.html
    │     ├─ resources.html
    │     ├─ subscribe.html
    │     └─ tools.html
    └─ uploads/
    ```

12. Challenges and Risks

    - Dataset bias and limited sample size may reduce generalization.
    - Model predictions should not be used as a definitive diagnosis.
    - Regulatory and privacy concerns for medical data in real deployments.

13. Future Scope

    - Extend dataset with more labeled scans and multi-institutional data.
    - Add segmentation models to localize tumor regions.
    - Convert to a clinical-grade pipeline with validation, explainability, and CI/CD.

14. Conclusion

    This demo demonstrates how transfer learning (VGG16) can be applied to MRI classification tasks and provides a simple web interface for inference. It is intended for educational use and research prototyping only.

15. References

    - Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG).
    - TensorFlow / Keras documentation: https://www.tensorflow.org/
    - Project code: see repository root files and `MRI_Brian_Tumor_Detection_DL.ipynb`.

---

Run (development / Windows):

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Model file is at `models/model.h5`. Example quick test script: `python test_prediction.py`.

