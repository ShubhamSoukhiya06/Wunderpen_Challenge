from flask import Flask, request, jsonify
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
import time
import traceback


app = Flask(__name__)

def detect_belt(bi_images):
    try:
        hsv = cv2.cvtColor(bi_images, cv2.COLOR_BGR2HSV)
        sobelx = cv2.Sobel(hsv[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(hsv[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
        magnitude, _ = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
        threshold = 0.1 * cv2.norm(magnitude, cv2.NORM_INF)
        edges = magnitude > threshold
        edges = cv2.morphologyEx(np.uint8(edges), cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_edges = bi_images.copy()
        image_height = bi_images.shape[0]
        xcrop_end = bi_images.shape[1]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > (0.7 * image_height):
                cv2.rectangle(filtered_edges, (x, y), (x + w, y + h), (0, 255, 0), 2)
                xcrop_end = w

    except Exception as e:
        print(f"An error occurred: {e}")
        xcrop_end = bi_images.shape[1]

    return xcrop_end

def subtract_images(background, foreground):
    if background is None or foreground is None:
        print("Error: Images not loaded correctly.")
        return None

    if background.shape != foreground.shape:
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

    subtracted_image = cv2.absdiff(foreground, background)
    subtracted_gray = cv2.cvtColor(subtracted_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(subtracted_gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean_mask = 255 - clean_mask
    subtracted_image[mask == 0] = [0, 0, 0]

    return subtracted_image

def extract_gcode_text(gi_image):
    gray = cv2.cvtColor(gi_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)
    inverted_bgr = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    return inverted_bgr

def crop_save_images(blanc_path, written_path, gcode_path):
    #bi_image = cv2.imread(blanc_path)
    #wi_image = cv2.imread(written_path)
    #gi_image = cv2.imread(gcode_path)
    bi_image = blanc_path
    wi_image = written_path
    gi_image = gcode_path
    bi_images = bi_image[:,150:,:]
    wi_images = wi_image[:,150:,:]
    xcrop_end = detect_belt(bi_images)
    bi_images = bi_images[:, :xcrop_end, :]
    wi_images = wi_images[:, :xcrop_end, :]
    result = subtract_images(bi_images, wi_images)
    result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    inverted_bgr = extract_gcode_text(gi_image)
    os.makedirs('./test_content', exist_ok=True)
    os.makedirs('./test_content_gcode', exist_ok=True)
    cv2.imwrite('./test_content/written.png', result)
    cv2.imwrite('./test_content_gcode/gcode.png', inverted_bgr)

def resize_and_pad(image, target_size):
    h, w = image.shape[:2]
    sh, sw = target_size
    aspect = w / h
    if aspect > 1:
        new_w = sw
        new_h = int(new_w / aspect)
    else:
        new_h = sh
        new_w = int(new_h * aspect)
    resized = cv2.resize(image, (new_w, new_h))
    delta_w = sw - new_w
    delta_h = sh - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = resize_and_pad(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

def build_cnn(input_shape):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu')
    ])
    return model

def build_siamese_network():
    input_a = Input(shape=(224, 224, 1))
    input_b = Input(shape=(224, 224, 1))
    cnn = build_cnn((224, 224, 1))
    processed_a = cnn(input_a)
    processed_b = cnn(input_b)
    distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
    output = Dense(1, activation='sigmoid')(distance)
    model = Model([input_a, input_b], output)
    return model

def process_image(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # Load image
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")
    
    return image

def predict(bi_image, wi_image, gi_image):
    input_shape = (224, 224, 1)
    crop_save_images(bi_image, wi_image, gi_image)
    siamese_network = build_siamese_network()
    siamese_network.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint_dir = './checkpoints'
    weights_path = os.path.join(checkpoint_dir, 'best_model.weights.h5')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    siamese_network.load_weights(weights_path)
    image_path_written = './test_content/written.png'
    image_path_gcode = './test_content_gcode/gcode.png'
    test_image_written = preprocess_image(image_path_written)
    test_image_gcode = preprocess_image(image_path_gcode)
    test_image_written = np.expand_dims(test_image_written, axis=0)
    test_image_gcode = np.expand_dims(test_image_gcode, axis=0)
    prediction = siamese_network.predict([test_image_written, test_image_gcode])
    return prediction[0][0]

@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        files = request.files
        if 'file1' not in files or 'file2' not in files or 'file3' not in files:
            return jsonify({"error": "Images not provided"}), 400
        # Read files from the request
        image1 = files['file1'].read()
        image2 = files['file2'].read()
        image3 = files['file3'].read()

        # Convert image bytes to numpy arrays
        image1 = cv2.imdecode(np.frombuffer(image1, np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(image2, np.uint8), cv2.IMREAD_COLOR)
        image3 = cv2.imdecode(np.frombuffer(image3, np.uint8), cv2.IMREAD_COLOR)

        if image1 is None or image2 is None or image3 is None:
            return jsonify({"error": "Failed to process images"}), 400 

        result = predict(image1, image2, image3)
        return jsonify({"result": "correct" if result >= 0.5 else "faulty"})

    except Exception as e:
        traceback.print_exception(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)






       
