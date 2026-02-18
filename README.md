********Project Development Flow********
__________________________________________________________

--> Load MRI Images Dataset
    Testing 
        glioma - 300 files
        meningioma - 306 files
        notumor - 405 files
        pituitary - 300 files

    Training 
        glioma - 300 files
        meningioma - 306 files
        notumor - 405 files
        pituitary - 300 files

--> Data Preprocessing
        Reading Images
        Data Augmentation
        Label Encoding 
        Data Generation

--> Machine Vision & Human Vision
    input -> sensing device -> machine -> output

--> Model Training
    mostly used -> CNN Model
    we used -> Transformer Learning (VGG16) Model

--> Transfer Learning DL: 
    It is a techniquein ML where a model 'trained on one task' is reused or adapted to 'solve a different, but related task' instead of training a model from scratch.

    task A ---> Pre-trained Model -->
                                     |
                                Knowledge (of previous Model)
                                     |
    Task B ---> New Model(create) <--
                                    
--> VGG16 for Transfer Learning
    The model is built on top of VGG16, which is pre-trained convolutional neural network (CNN) for image classification

# run project for windows

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
