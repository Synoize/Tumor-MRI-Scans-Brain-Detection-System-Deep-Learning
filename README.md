# MRI Brain Tumor Detection — DL Demo

## Abstract

Brain tumors are a serious health problem and need to be found quickly and correctly. Doctors usually look at brain MRI scans by hand, which takes a lot of time and can have mistakes. This project uses artificial intelligence to automatically detect and classify brain tumors in MRI images.

The system uses a pre-trained AI model called VGG16 to look for patterns in the MRI images. It can identify four types of brain conditions: glioma tumor, meningioma tumor, pituitary tumor, and no tumor. The images are prepared using special techniques like resizing, adjustment, and rotation to make the AI model work better.

We built a simple website using Flask where doctors or users can upload an MRI image and get an instant result showing what type of tumor (if any) is detected. The website has a login system and an admin section where admins can see all predictions and manage user accounts.

This project shows how AI can help doctors detect brain tumors faster and more accurately. It also teaches people how to use pre-trained models, prepare images for AI, and build web applications for medical purposes. The system is flexible and can be improved by training with more data and making the AI model better.

**Keywords:** Brain Tumor Detection, MRI Images, Artificial Intelligence, VGG16 Model, Web Application

---

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

    Magnetic Resonance Imaging (MRI) is a common tool doctors use to see inside the brain. Finding and identifying tumors in MRI scans usually takes a trained specialist and a lot of time. In this project, we build a computer program that can look at MRI images and decide if there is a tumor and what type it might be. The system uses modern deep learning methods, borrowing a pre‑trained neural network (VGG16) and adding a small custom classifier on top.

    The goal is not to replace doctors, but to give them a fast, second opinion and to show students how machine learning can help in healthcare. The program also comes with a simple web interface so anyone can upload a scan and get a prediction right away.


3. Scope of Work

    - Build an image preprocessing pipeline (augmentation, resizing, normalization).
    - Train and evaluate a transfer-learning model for multi-class classification.
    - Provide a simple web UI to upload MRI images and display predictions.
    - Package a trained model (`models/model.h5`) for inference.

4. Existing System and Need for Proposed System

    - Existing manual inspection by radiologists is time-consuming and subjective.
    - Proposed system offers fast, repeatable preliminary triage to assist clinicians and to support educational experiments.

5. Facilities Required for the Project

    To run or develop this MRI brain-tumor detection system, the following hardware
    and software are recommended. The project will work on lower-spec machines, but
    performance (especially model training) will be slower.

    i. Hardware Requirements

        - **Processor (CPU):** A quad-core or better CPU will make development and
          inference faster. Any modern Intel/AMD processor should suffice.
        - **Graphics Card (GPU):** An NVIDIA GPU with at least 4 GB of VRAM is
          highly recommended for training the deep learning model. Training on the
          CPU is possible but much slower. Inference (prediction) can run entirely
          on the CPU if a GPU isn't available.
        - **Memory (RAM):** 8 GB of system RAM is a good baseline. More memory helps
          when loading large datasets or running multiple services at once.
        - **Storage:** Allocate around 5 GB of free disk space for dataset images,
          trained model files (`models/model.h5`), and the Python environment.

    ii. Software Requirements

        - **Operating System:** Windows, Linux, or macOS are all supported.
        - **Python:** Version 3.8 or newer. The `requirements.txt` file lists
          all Python dependencies; install them using `pip install -r
          requirements.txt`.
        - **Python Libraries:** Key packages include TensorFlow/Keras for the
          neural network, Pillow or OpenCV for image handling, Flask for the web
          application, and SQLAlchemy/Flask-Login for user management.
        - **Database:** SQLite is used by default (bundled with Python) to store
          user accounts and prediction logs. No separate database server is
          required.
        - **Web Browser:** Any modern browser (Chrome, Firefox, Edge) to access the
          web interface at `http://localhost:5000` during development.

6. Objectives

    The main goals of this project are:

    - **Improve diagnostic speed and consistency.** Use a deep learning model to quickly label MRI scans, giving doctors a reliable second opinion.
    - **Leverage transfer learning.** Show how a pre-trained convolutional network (VGG16) can be adapted to a new medical imaging task without training from scratch.
    - **Build a user-friendly tool.** Create a simple website where anyone can upload an MRI image and see the predicted class along with a confidence score.
    - **Educational value.** Provide a clear example of data preprocessing, model training, and deployment that students and researchers can follow and extend.
    - **Modular design.** Structure code and data so that future improvements (more classes, better models, larger datasets) can be added with minimal changes.


7. User Requirements

    - Users should be able to upload an MRI image and receive a predicted class with confidence score.
    - Admin users should be able to view prediction logs and manage uploads (basic web UI provided in `templates/admin/`).

8. Methodology

    The development of the system followed a straightforward machine‑learning pipeline. First,
    a labeled dataset of brain MRI scans was assembled, with separate folders for each
    category (glioma, meningioma, pituitary and healthy). These images were read from the
    `mri_sample_images/` directory.

    Before training, each image was resized to 128×128 pixels, normalized to the [0,1]
    range and augmented with random transformations (rotations, horizontal/vertical flips,
    small shifts) to increase variability and reduce overfitting. The augmented data was
    fed into a convolutional neural network.

    Instead of training a network from scratch, the project uses transfer learning: the
    VGG16 model pre-trained on ImageNet provides the convolutional feature extractor,
    while a small custom head (dense layers with dropout) performs the four-class
    classification. The model is compiled with the categorical crossentropy loss and
    optimized using Adam. Training occurs in the Jupyter notebook, and the final weights
    are saved as `models/model.h5`.

    For inference, the Flask application loads the saved model once at startup. When a user
    uploads an MRI image via the web interface, the same preprocessing steps are applied,
    the image is passed to the model, and the predicted class along with a confidence score
    is returned to the user. Predictions and metadata may be logged in the SQLite database.

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

