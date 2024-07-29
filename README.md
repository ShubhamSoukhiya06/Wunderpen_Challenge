# Project README

## Overview

This project consists of three main components:

1. **Angular Frontend App**
2. **Spring Boot Backend (Kotlin)**
3. **Python Service**

### Angular Frontend App

The Angular frontend application provides three primary functionalities for managing image uploads related to job numbers:

1. **Single Image Upload**: Upload images for a particular job number individually. Supported image types include Gcode, Blanc, and Written images.
2. **Folder Upload**: Upload entire folders containing Gcode and Blanc & Written images separately.
3. **Drag and Drop**: Utilize the drag-and-drop feature to upload multiple images at once.

**Process Flow**:
- Users upload images via the frontend interface.
- The images are sent to the backend at `http://localhost:8080`.
- Once uploaded, the backend processes the images and communicates with the Python service running at `http://localhost:5000`.

### Backend (Spring Boot with Kotlin)

The backend is developed using the Spring Boot framework with Kotlin. Its responsibilities include:

- Receiving images from the frontend.
- Saving the images locally.
- Sending the image locations to the Python service for prediction.

### Python Service

The Python service handles the image processing and prediction tasks. It performs the following actions:

- Receives image data from the backend.
- Processes the images through a custom pipeline.
- Predicts whether each image is correct or faulty.
- Sends the prediction results back to the backend.

### Data Flow

1. **Upload**: The Angular frontend sends image data to the backend.
2. **Processing**: The backend saves the images and forwards their locations to the Python service.
3. **Prediction**: The Python service processes the images and determines their quality.
4. **Result**: The Python service sends the prediction results back to the backend.
5. **Display**: The backend forwards the results to the Angular frontend, where they are displayed to the user.

## Setup

### Prerequisites

- Node.js and npm (for Angular frontend)
- Java JDK (for Spring Boot backend)
- Python (for Python service)

### Running the Application

1. **Frontend**:
   - Navigate to the `angular-frontend` directory.
   - Install dependencies: `npm install`
   - Start the Angular application: `ng serve`

2. **Backend**:
   - Navigate to the `backend` directory.
   - Build and run the Spring Boot application

3. **Python Service**:
   - Navigate to the `python-service` directory.
   - Install dependencies: `pip install -r requirements.txt`
   - Start the Python service: `python app.py`

## API Endpoints

- **Frontend to Backend**:
  - Upload endpoint: `POST http://localhost:8080/checkImageQuality`
  
- **Backend to Python Service**:
  - Processing endpoint: `POST http://localhost:5000/process_images`