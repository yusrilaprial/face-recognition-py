from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import dlib

app = Flask(__name__)

# Load dlib face detector dan predictor untuk landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

# def is_image_blurry(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     variance = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return variance < 75  # Threshold ditingkatkan untuk toleransi

def check_reflection(image):
    # Hitung rata-rata nilai pixel di area tertentu
    avg_color = cv2.mean(image)
    # Penerimaan lebih luas untuk refleksi
    return avg_color[0] > 70 and avg_color[1] > 70 and avg_color[2] > 70

def analyze_landmarks(landmarks):
    # Menghitung jarak antara landmark tertentu
    landmarks_array = np.array([(p.x, p.y) for p in landmarks.parts()])
    distance = np.linalg.norm(landmarks_array[30] - landmarks_array[34])  # Contoh: jarak antara alis
    return distance

def check_image_validity(image):
    # Deteksi wajah
    faces = detector(image, 1)
    
    if len(faces) == 0:
        return "No face detected"
    
    # # Cek apakah gambar buram
    # if is_image_blurry(image):
    #     return "Image is blurry, but could still be a real face"

    # Cek refleksi
    if not check_reflection(image):
        return "Unusual reflection detected, further analysis needed"

    # Ambil landmark wajah
    for face in faces:
        landmarks = predictor(image, face)

        # Analisis jarak antar landmark
        landmark_distance = analyze_landmarks(landmarks)

        # Threshold jarak landmark lebih longgar
        if landmark_distance < 30:  # Threshold ini lebih tinggi
            return "Unnatural proportions detected, further analysis needed"

    return "Face detected. Further analysis needed."

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

@app.route('/match-faces', methods=['POST'])
def match_faces():
    if 'faces' not in request.files or 'image_to_match' not in request.files:
        return jsonify({'status': 'failed', 'message': 'Faces and image_to_match files are required!'}), 400

    # Mengambil daftar wajah untuk didaftarkan
    faces = request.files.getlist('faces')
    image_to_match_file = request.files['image_to_match']

    registered_faces = {}

    # Proses setiap wajah di parameter faces
    for file in faces:
        name = file.name  # Nama file digunakan sebagai nama wajah
        try:
            image = read_image(file)
        except ValueError as e:
            return jsonify({'status': 'failed', 'message': str(e)}), 400

        # Deteksi dan encode wajah
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) == 0:
            return jsonify({'status': 'failed', 'message': f'No face detected in the image for {name}!'}), 400

        # Simpan encoding wajah ke dalam dictionary
        registered_faces[name] = face_encodings[0]

    try:
        # Proses gambar untuk dicocokkan
        image_to_match = read_image(image_to_match_file)
    except ValueError as e:
        return jsonify({'status': 'failed', 'message': str(e)}), 400

    # Deteksi dan encode wajah untuk gambar yang dicocokkan
    face_encodings_to_check = face_recognition.face_encodings(image_to_match)
    if len(face_encodings_to_check) == 0:
        return jsonify({'status': 'failed', 'message': 'No face detected in the image to match!'}), 400

    face_encoding_to_check = face_encodings_to_check[0]

    # Bandingkan setiap encoding wajah yang terdaftar dengan gambar yang akan dicocokkan
    for name, face_encoding in registered_faces.items():
        match = face_recognition.compare_faces([face_encoding], face_encoding_to_check)
        if match[0]:
            return jsonify({'status': 'success', 'message': f'Match found for {name}.'}), 200

    return jsonify({'status': 'failed', 'message': 'No match found!'}), 404

@app.route('/check-face', methods=['POST'])
def check_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    
    # Menggunakan OpenCV untuk membaca gambar dari upload
    npimg = np.fromfile(file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Lakukan pengecekan keaslian gambar
    result = check_image_validity(image)

    # Kembalikan hasil deteksi
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
