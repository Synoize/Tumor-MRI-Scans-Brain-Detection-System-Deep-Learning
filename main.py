import os
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance
import random
import io
import base64
import h5py
import json

app = Flask(__name__, template_folder='templates', static_folder='assets', static_url_path='/assets')
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mri_tumor_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/model.h5'
IMAGE_SIZE = 128  # Model expects 128x128 images

# Class labels (based on typical brain tumor MRI dataset structure)
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'predictions_count': len(self.predictions)
        }

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    has_tumor = db.Column(db.Boolean, nullable=False)
    glioma_prob = db.Column(db.Float)
    meningioma_prob = db.Column(db.Float)
    notumor_prob = db.Column(db.Float)
    pituitary_prob = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'filename': self.filename,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'has_tumor': self.has_tumor,
            'all_probabilities': {
                'glioma': self.glioma_prob,
                'meningioma': self.meningioma_prob,
                'notumor': self.notumor_prob,
                'pituitary': self.pituitary_prob
            },
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def rebuild_and_load_model():
    """Rebuild the model architecture and load weights"""
    try:
        print("Rebuilding model architecture...")
        
        # Build the model architecture (same as training)
        base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
        
        # Freeze all layers initially
        for layer in base_model.layers:
            layer.trainable = False
        
        # Set the last few layers to be trainable (as per original training)
        base_model.layers[-2].trainable = True
        base_model.layers[-3].trainable = True
        base_model.layers[-4].trainable = True
        
        # Build the final model
        model = Sequential([
            Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(4, activation='softmax')  # 4 classes
        ])
        
        # Load weights from the saved model
        model.load_weights(MODEL_PATH)
        print("Model rebuilt and weights loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
        
    except Exception as e:
        print(f"Error rebuilding model: {e}")
        return None

# Try to load the model
model = None
try:
    # First try standard loading
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully (standard)!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"Standard loading failed: {e}")
    print("Attempting to rebuild model architecture...")
    model = rebuild_and_load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def augment_image(img):
    """Apply the same augmentation as training"""
    img = Image.fromarray(np.uint8(img))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    img = np.array(img) / 255.0
    return img

def preprocess_image(img_path):
    """Preprocess image for model prediction - matches training preprocessing"""
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = augment_image(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/style.css')
def serve_css():
    return app.send_static_file('../templates/style.css')

@app.route('/script.js')
def serve_js():
    return app.send_static_file('../templates/script.js')

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    # Store next page for redirect after login
    next_page = request.args.get('next') or request.form.get('next')
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('register', next=next_page))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register', next=next_page))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return redirect(url_for('register', next=next_page))
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register', next=next_page))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register', next=next_page))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        # First user is admin
        if User.query.count() == 0:
            user.is_admin = True
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        # Redirect to login with next parameter
        return redirect(url_for('login', next=next_page)) if next_page else redirect(url_for('login'))
    
    return render_template('register.html', next=next_page)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    # Get next page from query string or form data
    next_page = request.args.get('next') or request.form.get('next')
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f'Welcome back, {user.username}!', 'success')
            # Redirect to next page if valid, otherwise to index
            if next_page and next_page.startswith('/'):
                # Remove .html extension for cleaner URLs
                if next_page.endswith('.html'):
                    next_page = next_page[:-5]
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            # Preserve next parameter on failed login
            return redirect(url_for('login', next=next_page)) if next_page else redirect(url_for('login'))
    
    return render_template('login.html', next=next_page)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('profile.html', user=current_user, predictions=user_predictions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pages/<page>')
def pages(page):
    try:
        # Handle pages with or without .html extension
        if not page.endswith('.html'):
            page = page + '.html'
        return render_template(f'pages/{page}')
    except:
        return "Page not found", 404

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Preprocess and predict
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array, verbose=0)
            
            # Get prediction result
            # Multi-class classification with 4 classes
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx]) * 100
            predicted_label = CLASS_LABELS[predicted_class_idx]
            
            # Determine if tumor is detected
            has_tumor = predicted_label != 'notumor'
            
            result = {
                'prediction': predicted_label.capitalize(),
                'confidence': round(confidence, 2),
                'has_tumor': has_tumor,
                'all_probabilities': {
                    label: round(float(prob) * 100, 2) 
                    for label, prob in zip(CLASS_LABELS, prediction[0])
                },
                'filename': file.filename,
                'saved': False
            }
            
            # Save prediction to database if user is logged in
            if current_user.is_authenticated:
                pred_record = Prediction(
                    user_id=current_user.id,
                    filename=file.filename,
                    prediction=predicted_label.capitalize(),
                    confidence=round(confidence, 2),
                    has_tumor=has_tumor,
                    glioma_prob=round(float(prediction[0][0]) * 100, 2),
                    meningioma_prob=round(float(prediction[0][1]) * 100, 2),
                    notumor_prob=round(float(prediction[0][2]) * 100, 2),
                    pituitary_prob=round(float(prediction[0][3]) * 100, 2)
                )
                db.session.add(pred_record)
                db.session.commit()
                result['prediction_id'] = pred_record.id
                result['saved'] = True
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            import traceback
            return jsonify({'error': f'Prediction error: {str(e)}', 'traceback': traceback.format_exc()}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

# Admin Routes
@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    tumor_detections = Prediction.query.filter_by(has_tumor=True).count()
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    
    return render_template('admin/dashboard.html', 
                         total_users=total_users,
                         total_predictions=total_predictions,
                         tumor_detections=tumor_detections,
                         recent_predictions=recent_predictions)

@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    users = User.query.all()
    return render_template('admin/users.html', users=users)

@app.route('/admin/predictions')
@login_required
def admin_predictions():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template('admin/predictions.html', predictions=predictions)

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot delete yourself'}), 400
    
    # Delete user's predictions first
    Prediction.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({'message': 'User deleted successfully'})

@app.route('/admin/prediction/<int:pred_id>/delete', methods=['POST'])
@login_required
def admin_delete_prediction(pred_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    pred = Prediction.query.get_or_404(pred_id)
    db.session.delete(pred)
    db.session.commit()
    
    return jsonify({'message': 'Prediction deleted successfully'})

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")
    
    print("Starting MRI Brain Tumor Detection Server...")
    print(f"Model status: {'Loaded' if model else 'Not Loaded'}")
    app.run(debug=True, host='0.0.0.0', port=5000)