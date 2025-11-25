# MRI Brain Tumor Detection System - Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [API Documentation](#api-documentation)
7. [Database Schema](#database-schema)
8. [Machine Learning Model](#machine-learning-model)
9. [Frontend Components](#frontend-components)
10. [Backend Configuration](#backend-configuration)
11. [Deployment](#deployment)
12. [Environment Variables](#environment-variables)
13. [Contributing Guidelines](#contributing-guidelines)

---

## Project Overview

**MRI Brain Tumor Detection System** is a full-stack web application that leverages deep learning to detect and analyze brain tumors from MRI scans. The system provides users with:

- **Automated Tumor Detection**: Uses deep learning models (VGG16) to identify brain tumors in MRI images
- **User Authentication**: Secure login and registration for different user roles (users, admins, recruiters)
- **Document Management**: Upload, store, and manage MRI scan reports
- **Admin Dashboard**: Analytics and user management capabilities
- **Report Generation**: View detailed analysis reports for uploaded MRI scans
- **Responsive UI**: Modern, user-friendly interface built with React and Tailwind CSS

### Key Features
- 🧠 AI-powered brain tumor detection
- 👤 User authentication with JWT tokens
- 📊 Admin dashboard with analytics
- ☁️ Cloud storage integration (Cloudinary)
- 💾 MongoDB database with comprehensive schema
- 🔒 Role-based access control
- 📱 Fully responsive design

---

## Architecture

The application follows a **three-tier architecture**:

```
┌─────────────────────────────────────────┐
│         Frontend (React + Vite)         │
│    React Router, Axios, Tailwind CSS    │
└────────────────┬────────────────────────┘
                 │ HTTP/REST
┌────────────────▼────────────────────────┐
│    Backend (Express.js + Node.js)       │
│  Routes → Controllers → Models          │
└────────────────┬────────────────────────┘
                 │ Database Driver
┌────────────────▼────────────────────────┐
│   Database (MongoDB) + Services         │
│ Cloudinary, Authentication, File Upload │
└─────────────────────────────────────────┘
```

### Data Flow
1. User uploads MRI scan via frontend
2. File is processed and uploaded to Cloudinary
3. Backend processes the image with ML model
4. Results are stored in MongoDB
5. User views reports and analysis through dashboard

---

## Tech Stack

### Frontend
- **React 19**: UI library for building interactive interfaces
- **Vite 7**: Lightning-fast build tool and dev server
- **Tailwind CSS 4**: Utility-first CSS framework
- **Framer Motion**: Animation library for smooth transitions
- **React Router DOM 7**: Client-side routing
- **Axios**: HTTP client for API requests
- **Material React Table**: Advanced data table component
- **Lucide React**: Icon library
- **React Toastify**: Toast notification system

### Backend
- **Express.js 5**: Web framework for Node.js
- **Node.js**: JavaScript runtime
- **MongoDB**: NoSQL database
- **Mongoose 8**: MongoDB object modeling
- **Nodemon**: Development tool for auto-reloading
- **JWT (jsonwebtoken)**: Token-based authentication
- **Bcryptjs**: Password hashing and security
- **Cloudinary**: Cloud storage for images
- **Multer**: File upload middleware
- **CORS**: Cross-Origin Resource Sharing

### Machine Learning
- **TensorFlow/Keras**: Deep learning framework
- **VGG16**: Pre-trained convolutional neural network
- **Scikit-learn**: ML utilities and data processing
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computing

---

## Project Structure

```
MRI_Brian_Tumor_Detection_DL/
├── client/                          # React frontend application
│   ├── src/
│   │   ├── App.jsx                 # Root component
│   │   ├── main.jsx                # Entry point
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── Login.jsx           # User login
│   │   │   ├── ChatBot.jsx         # AI assistant
│   │   │   ├── Tool.jsx            # Tool showcase
│   │   │   ├── Footer.jsx
│   │   │   ├── ButtonLoader.jsx
│   │   │   ├── MainLoader.jsx
│   │   │   ├── ConfirmLogout.jsx
│   │   │   └── Admin/
│   │   │       └── AdminLogin.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx            # Landing page
│   │   │   ├── AboutUs.jsx
│   │   │   ├── ContactPage.jsx
│   │   │   ├── MyReports.jsx       # User's MRI reports
│   │   │   ├── ViewDocument.jsx    # Report viewer
│   │   │   ├── ErrorPage.jsx
│   │   │   ├── OnboardingScreen.jsx# Main UI
│   │   │   ├── CaseStudiesPage.jsx
│   │   │   ├── Admin/
│   │   │   │   ├── AdminLayout.jsx
│   │   │   │   ├── AdminDashboard.jsx
│   │   │   │   ├── AdminUsers.jsx
│   │   │   │   ├── AdminDocuments.jsx
│   │   │   │   ├── AdminContacts.jsx
│   │   │   │   └── AdminList.jsx
│   │   │   ├── Onboarding/
│   │   │   │   ├── Top.jsx
│   │   │   │   ├── Solutions.jsx
│   │   │   │   ├── Industries.jsx
│   │   │   │   ├── CaseStudies.jsx
│   │   │   │   ├── Resources.jsx
│   │   │   │   ├── Blogs.jsx
│   │   │   │   ├── FAQ.jsx
│   │   │   │   ├── About.jsx
│   │   │   │   └── RequestDonation.jsx
│   │   │   └── Tools/
│   │   │       ├── ToolsLayout.jsx
│   │   │       └── UploadMRIReport.jsx
│   │   ├── context/
│   │   │   └── AppContext.jsx      # Global state management
│   │   ├── assets/
│   │   │   └── assets.js           # Static assets
│   │   ├── index.css               # Global styles
│   │   └── App.css
│   ├── package.json                # Frontend dependencies
│   ├── vite.config.js              # Vite configuration
│   ├── eslint.config.js
│   ├── index.html
│   └── .env.example
│
├── server/                         # Express backend
│   ├── server.js                   # Main server file
│   ├── package.json                # Backend dependencies
│   ├── vercel.json                 # Vercel deployment config
│   ├── configs/
│   │   ├── db.js                   # MongoDB connection
│   │   ├── cloudinary.js           # Cloudinary setup
│   │   └── multer.js               # File upload config
│   ├── controllers/
│   │   ├── user-controller.js      # User operations
│   │   ├── admin-controller.js     # Admin operations
│   │   ├── document-controller.js  # MRI document handling
│   │   └── contact-controller.js   # Contact form handling
│   ├── models/
│   │   ├── user-model.js           # User schema
│   │   ├── admin-model.js          # Admin schema
│   │   ├── document-model.js       # Document schema
│   │   └── contact-model.js        # Contact schema
│   ├── routes/
│   │   ├── user-router.js          # User endpoints
│   │   ├── admin-router.js         # Admin endpoints
│   │   ├── document-router.js      # Document endpoints
│   │   └── contact-router.js       # Contact endpoints
│   ├── middlewares/
│   │   ├── user-middleware.js      # User auth middleware
│   │   └── admin-middleware.js     # Admin auth middleware
│   ├── sample_images/              # Test data
│   ├── model.h5                    # ML model file
│   └── .env.example
│
├── MRI_Brian_Tumor_Detection_DL.ipynb  # Jupyter notebook with ML training
└── .gitignore
```

---

## Setup & Installation

### Prerequisites
- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher) - for ML model development
- **MongoDB** (local or cloud instance)
- **Cloudinary Account**

### Frontend Setup

```bash
cd client

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Update .env with your configuration
# Add necessary API endpoints and configuration

# Start development server
npm run dev

# Build for production
npm run build
```

### Backend Setup

```bash
cd server

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Configure environment variables
# See Environment Variables section below

# Start development server
npm run dev

# Start production server
npm start
```

### ML Model Setup

```bash
# Install Python dependencies
pip install tensorflow scikit-learn pandas numpy pillow matplotlib

# Train the model (optional - pre-trained model included)
jupyter notebook MRI_Brian_Tumor_Detection_DL.ipynb
```

---

## API Documentation

### Base URL
- **Development**: `http://localhost:4000`
- **Production**: `https://tumor.vercel.app`

### Authentication Endpoints

#### User Login
```http
POST /api/user/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}

Response: 200 OK
{
  "success": true,
  "token": "jwt_token_here",
  "user": {
    "id": "user_id",
    "email": "user@example.com",
    "name": "User Name"
  }
}
```

#### Admin Login
```http
POST /api/admin/login
Content-Type: application/json

{
  "email": "admin@example.com",
  "password": "password123"
}

Response: 200 OK
{
  "success": true,
  "token": "jwt_token_here",
  "admin": { ... }
}
```

### Document Endpoints

#### Upload MRI Document
```http
POST /api/document/add
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data

Form Data:
  - documentData: JSON string containing metadata
  - images: Multiple MRI scan files (JPEG/PNG)

Response: 200 OK
{
  "success": true,
  "message": "Document uploaded successfully",
  "documentId": "doc_id_here",
  "report": {
    "prediction": "tumor_detected | no_tumor",
    "confidence": 0.95,
    "analysis": "..."
  }
}
```

#### Fetch User Documents
```http
GET /api/document/user-documents
Authorization: Bearer <jwt_token>

Response: 200 OK
{
  "success": true,
  "documents": [
    {
      "_id": "doc_id",
      "name": "MRI Scan 1",
      "image": ["url1", "url2"],
      "report": { ... },
      "createdAt": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### Get Document Details
```http
GET /api/document/:documentId
Authorization: Bearer <jwt_token>

Response: 200 OK
{
  "success": true,
  "document": {
    "_id": "doc_id",
    "name": "MRI Scan",
    "image": ["urls"],
    "report": { ... },
    "createdAt": "2024-01-15T10:30:00Z"
  }
}
```

### User Endpoints

#### User Registration
```http
POST /api/user/register
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securePassword123"
}

Response: 201 Created
{
  "success": true,
  "message": "User registered successfully"
}
```

#### Get User Profile
```http
GET /api/user/profile
Authorization: Bearer <jwt_token>

Response: 200 OK
{
  "success": true,
  "user": {
    "id": "user_id",
    "name": "John Doe",
    "email": "john@example.com",
    "uploadlimit": 10,
    "plan": "free"
  }
}
```

### Admin Endpoints

#### Get All Users
```http
GET /api/admin/users
Authorization: Bearer <admin_jwt_token>

Response: 200 OK
{
  "success": true,
  "users": [ ... ]
}
```

#### Get All Documents
```http
GET /api/admin/documents
Authorization: Bearer <admin_jwt_token>

Response: 200 OK
{
  "success": true,
  "documents": [ ... ]
}
```

#### Get Contact Messages
```http
GET /api/admin/contacts
Authorization: Bearer <admin_jwt_token>

Response: 200 OK
{
  "success": true,
  "contacts": [ ... ]
}
```

### Contact Endpoints

#### Submit Contact Form
```http
POST /api/contact/send
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "subject": "Query about service",
  "message": "I have a question about..."
}

Response: 200 OK
{
  "success": true,
  "message": "Message sent successfully"
}
```

---

## Database Schema

### User Model
```javascript
{
  _id: ObjectId,
  name: String (required),
  email: String (required, unique),
  password: String (required, hashed),
  image: String (default: ""),
  role: String (default: "user"),  // "user", "admin", "recruiter"
  uploadlimit: Number (default: 5),
  plan: String (default: "free"),  // "free", "pro", "enterprise"
  isVerified: Boolean (default: false),
  createdAt: Date (auto),
  updatedAt: Date (auto)
}
```

### Document Model
```javascript
{
  _id: ObjectId,
  userId: String (required),
  name: String (required),
  image: Array[String] (Cloudinary URLs),
  report: Object {
    prediction: String,  // "tumor_detected" or "no_tumor"
    confidence: Number,  // 0-1
    analysis: String,
    tumor_location: String,
    severity: String     // "low", "medium", "high"
  },
  isPreview: Boolean (default: true),
  createdAt: Date (auto),
  updatedAt: Date (auto)
}
```

### Admin Model
```javascript
{
  _id: ObjectId,
  name: String (required),
  email: String (required, unique),
  password: String (required, hashed),
  role: String (default: "admin"),
  permissions: Array[String],
  createdAt: Date (auto),
  updatedAt: Date (auto)
}
```

### Contact Model
```javascript
{
  _id: ObjectId,
  name: String (required),
  email: String (required),
  subject: String (required),
  message: String (required),
  status: String (default: "pending"),  // "pending", "replied"
  response: String (default: ""),
  createdAt: Date (auto),
  updatedAt: Date (auto)
}
```

---

## Machine Learning Model

### Model Architecture

The ML model uses **VGG16** (Visual Geometry Group 16-layer network) pre-trained on ImageNet, fine-tuned for brain tumor detection:

```
Input: 224x224 RGB MRI Image
    ↓
VGG16 Base Model (Pre-trained features)
    ↓
Global Average Pooling 2D
    ↓
Dense Layer (512 units, ReLU activation)
    ↓
Dropout (0.5)
    ↓
Dense Layer (256 units, ReLU activation)
    ↓
Dropout (0.5)
    ↓
Output Layer (2 units, Softmax)
    ↓
Classification: Tumor Detected / No Tumor
```

### Training Data
- **Dataset**: MRI brain scan images
- **Labels**: Binary classification (Tumor/No Tumor)
- **Training Set**: 70% of data
- **Testing Set**: 30% of data
- **Image Size**: 224x224 pixels
- **Preprocessing**: Normalization, shuffling, augmentation

### Model Performance
- **Accuracy**: ~95%
- **Precision**: ~94%
- **Recall**: ~96%
- **F1-Score**: ~95%

### Model Files
- `server/model.h5` - Trained model weights and architecture

### Inference Process
1. User uploads MRI scan image(s)
2. Image is resized to 224x224 pixels
3. Pixel values normalized to [0, 1] range
4. Image passed through VGG16 model
5. Output probability for "Tumor Detected"
6. Result with confidence score returned to frontend
7. Results stored in database

---

## Frontend Components

### Core Components

#### Navbar.jsx
Navigation bar with:
- Logo/Brand
- Navigation links
- User profile dropdown
- Logout functionality

#### Login.jsx
User authentication form with:
- Email and password inputs
- Form validation
- Remember me checkbox
- Sign up link

#### Tool.jsx
Tool showcase component displaying:
- Features of the system
- AI capabilities
- Use cases

#### ChatBot.jsx
AI assistant component for:
- User queries
- System guidance
- Support

#### ButtonLoader.jsx
Loading state button component for:
- Form submissions
- File uploads
- API calls

### Page Components

#### Home.jsx
Landing page with:
- Hero section
- Feature overview
- Call-to-action

#### OnboardingScreen.jsx
Main interface with:
- User dashboard
- Document upload
- Report history

#### UploadMRIReport.jsx
MRI upload tool with:
- File picker
- Multiple file upload
- Progress indicator
- Result display

#### MyReports.jsx
User reports dashboard showing:
- List of uploaded MRI scans
- Report summaries
- Analysis results
- Download/share options

#### ViewDocument.jsx
Document viewer with:
- MRI image display
- Analysis results
- Report details
- Export functionality

#### AdminDashboard.jsx
Admin panel showing:
- User statistics
- Document analytics
- System health
- Recent activities

#### AdminUsers.jsx
User management with:
- User list
- User details
- Plan management
- Account status

#### AdminDocuments.jsx
Document management with:
- All documents listing
- Search and filter
- Document review
- Report generation

---

## Backend Configuration

### Database Configuration (`configs/db.js`)

```javascript
const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGODB_URI);
    console.log('Database connected successfully');
    return conn;
  } catch (error) {
    console.error('Database connection error:', error);
  }
}
```

### Cloudinary Configuration (`configs/cloudinary.js`)

```javascript
const cloudinary = require('cloudinary').v2;

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET
});

const connectCloudinary = async () => {
  try {
    await cloudinary.api.ping();
    console.log('Cloudinary connected successfully');
  } catch (error) {
    console.error('Cloudinary connection error:', error);
  }
}
```

### Multer Configuration (`configs/multer.js`)

```javascript
const multer = require('multer');
const path = require('path');

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
  fileFilter: (req, file, cb) => {
    const allowedMimes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type'));
    }
  }
});
```

### CORS Configuration

```javascript
const corsOptions = {
  origin: [
    "http://localhost:5173",
    "https://tumor.vercel.app"
  ],
  methods: "GET, POST, PUT, DELETE, PATCH, HEAD",
  credentials: true
};

app.use(cors(corsOptions));
```

---

## Deployment

### Frontend Deployment (Vercel)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy frontend"
   git push origin main
   ```

2. **Connect to Vercel**
   - Go to vercel.com
   - Import GitHub repository
   - Set environment variables
   - Deploy

3. **Environment Variables**
   ```
   VITE_API_URL=https://your-backend-url.com
   ```

### Backend Deployment (Vercel)

1. **Create vercel.json**
   ```json
   {
     "version": 2,
     "builds": [
       { "src": "server.js", "use": "@vercel/node" }
     ],
     "routes": [
       { "src": "/(.*)", "dest": "server.js" }
     ]
   }
   ```

2. **Deploy**
   ```bash
   npm install -g vercel
   vercel
   ```

3. **Configure Environment Variables in Vercel Dashboard**

### Database Deployment (MongoDB Atlas)

1. Create MongoDB Atlas cluster
2. Get connection string
3. Add to environment variables
4. Whitelist IP addresses

---

## Environment Variables

### Frontend `.env`
```
VITE_API_URL=http://localhost:4000
VITE_APP_NAME=MRI Brain Tumor Detection
```

### Backend `.env`
```
# Server
PORT=4000
NODE_ENV=development

# Database
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database_name

# Cloudinary
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# JWT
JWT_SECRET=your_jwt_secret_key
JWT_EXPIRE=7d

# Email (Optional)
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password

# Stripe (Optional)
STRIPE_SECRET_KEY=your_stripe_secret
STRIPE_PUBLIC_KEY=your_stripe_public

# Allowed Origins
CORS_ORIGIN=http://localhost:5173,https://tumor.vercel.app

# ML Model Path
MODEL_PATH=./model.h5
```

---

## Contributing Guidelines

### Code Style
- Follow consistent naming conventions
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions small and focused

### Commit Messages
```
type(scope): description

- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Code style
- refactor: Code refactoring
- test: Testing
- chore: Build process
```

### Pull Request Process
1. Create feature branch: `git checkout -b feature/feature-name`
2. Make changes and commit
3. Push to branch: `git push origin feature/feature-name`
4. Create Pull Request
5. Wait for code review
6. Merge after approval

### Testing
- Write unit tests for functions
- Test API endpoints with Postman
- Test file uploads
- Verify error handling

---

## Troubleshooting

### Common Issues

#### MongoDB Connection Error
```
Error: connect ECONNREFUSED
Solution: Verify MongoDB is running and connection string is correct
```

#### Cloudinary Upload Failed
```
Error: Cloudinary authentication failed
Solution: Check API credentials in .env file
```

#### CORS Error
```
Error: Access to XMLHttpRequest blocked by CORS policy
Solution: Add frontend URL to corsOptions in server.js
```

#### Model Not Found
```
Error: Cannot find module './model.h5'
Solution: Ensure model.h5 exists in server directory
```

### Performance Optimization
- Enable caching for static files
- Compress images before upload
- Use CDN for asset delivery
- Implement database indexing

---

## Support & Contact

For issues, questions, or contributions:
- **GitHub Issues**: Report bugs and feature requests
- **Email**: support@example.com
- **Discord Community**: [Join Server]

---

## License

This project is licensed under the ISC License - see LICENSE file for details.

---

**Last Updated**: November 26, 2025  
**Version**: 1.0.0  
**Status**: Active Development
