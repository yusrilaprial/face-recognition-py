from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2

app = Flask(__name__)

# Dictionary untuk menyimpan data wajah terdaftar
registered_faces = {}

def read_image(file):
    """Function to convert uploaded file to an OpenCV image."""
    np_arr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Membaca gambar sebagai BGR

    # Check for valid image
    if image is None:
        raise ValueError("Invalid image format or corrupted file.")

    # Convert BGR (default for OpenCV) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ensure the image is 8-bit per channel
    if image.dtype != np.uint8:
        raise ValueError("Unsupported image type, must be 8bit per channel.")

    return image

@app.route('/register', methods=['POST'])
def register_face():
    if 'name' not in request.form or 'image' not in request.files:
        return jsonify({'status': 'failed', 'message': 'Name and image file are required!'}), 400

    name = request.form['name']
    image_file = request.files['image']

    try:
        image = read_image(image_file)
    except ValueError as e:
        return jsonify({'status': 'failed', 'message': str(e)}), 400

    # Deteksi dan encode wajah
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        return jsonify({'status': 'failed', 'message': 'No face detected in the image!'}), 400

    # Simpan encoding wajah ke dalam dictionary
    registered_faces[name] = face_encodings[0].tolist()

    return jsonify({'status': 'success', 'message': f'Face registered for {name}.'}), 200

@app.route('/match', methods=['POST'])
def match_face():
    if 'image' not in request.files:
        return jsonify({'status': 'failed', 'message': 'Image file is required!'}), 400

    image_file = request.files['image']

    try:
        image = read_image(image_file)
    except ValueError as e:
        return jsonify({'status': 'failed', 'message': str(e)}), 400

    # Deteksi dan encode wajah
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        return jsonify({'status': 'failed', 'message': 'No face detected in the image!'}), 400

    face_encoding_to_check = face_encodings[0]

    # Cek apakah ada wajah yang cocok
    for name, face_encoding in registered_faces.items():
        match = face_recognition.compare_faces([np.array(face_encoding)], face_encoding_to_check)
        if match[0]:
            return jsonify({'status': 'success', 'message': f'Match found for {name}.'}), 200

    return jsonify({'status': 'failed', 'message': 'No match found!'}), 404

if __name__ == '__main__':
    app.run(debug=True)
