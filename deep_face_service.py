from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

# Fungsi untuk membaca gambar dari file yang dikirimkan
def read_image(file):
    np_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

@app.route('/detect-face', methods=['POST'])
def detect_face():
    try:
        # Memeriksa apakah file gambar ada di request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        img = read_image(image_file)
        
        # Deteksi wajah menggunakan DeepFace (auto download model for every actions)
        # result = DeepFace.analyze(img, actions=['age', 'gender', 'emotion', 'race'])
        result = DeepFace.analyze(img_path = img, actions=['age'])
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/anti-spoofing', methods=['POST'])
def anti_spoofing():
    try:
        # Memeriksa apakah file gambar ada di request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        img = read_image(image_file)

        result = DeepFace.represent(img_path = img, anti_spoofing = True)
        # result = DeepFace.extract_faces(img, anti_spoofing = True)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/match-faces', methods=['POST'])
def match_faces():
    try:
        # Memeriksa apakah file gambar ada di request
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Two image files are required'}), 400
        
        image_file1 = request.files['image1']
        image_file2 = request.files['image2']
        
        img1 = read_image(image_file1)
        img2 = read_image(image_file2)
        
        # Menggunakan fungsi verify untuk mencocokkan wajah
        result = DeepFace.verify(img1_path = img1, img2_path = img2, anti_spoofing = True)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/find-best-match', methods=['POST'])
def find_best_match():
    try:
        # Memeriksa apakah file gambar pencarian ada di request
        if 'query_image' not in request.files or 'list_image' not in request.files:
            return jsonify({'error': 'Query image and list_image array are required'}), 400
        
        # Mengambil gambar pencarian
        query_image = read_image(request.files['query_image'])

        # Buat direktori sementara untuk menyimpan kumpulan gambar
        with tempfile.TemporaryDirectory() as temp_dir:
            # Menyimpan gambar target di dalam direktori sementara
            for image_file in request.files.getlist('list_image'):
                image_path = os.path.join(temp_dir, image_file.filename)
                image_file.save(image_path)

            # Mencocokkan gambar menggunakan DeepFace.find()
            result = DeepFace.find(img_path = query_image, db_path = temp_dir, anti_spoofing = True)
            result = result[0].to_dict()

            # Additional
            result['basename'] = {}
            if result['identity']:
                result['basename'][0] = os.path.basename(result['identity'][0])

            return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
