from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import tempfile
import os
import requests
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

# Fungsi untuk membaca gambar dari file yang dikirimkan
def read_image(file):
    np_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

def save_image(image, filename, db_path):
    try:
        # Buat folder jika belum ada
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        # Cek jika gambar adalah base64
        if image.startswith('data:image/'):
            # Pisahkan metadata dan data base64
            header, base64_data = image.split(',', 1)
            image_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_data))
            ext = header.split(';')[0].split('/')[1]  # Mendapatkan ekstensi
        else:
            # Mendownload gambar menggunakan requests
            response = requests.get(image)
            if response.status_code != 200:
                raise ValueError(f"Failed to download image: {response.status_code}")

            # Membaca gambar menggunakan PIL
            image = Image.open(BytesIO(response.content))
            ext = image.format.lower()

        # Cek ekstensi gambar dari formatnya (contoh: jpg, png)
        if ext not in ['jpeg', 'jpg', 'png']:
            raise ValueError("Image Format not supported, must be JPG or PNG.")

        # Simpan gambar dengan nama unik
        filename = f"{filename}.{ext}"
        image_path = os.path.join(db_path, filename)

        # # Cek apakah file sudah ada
        # if os.path.exists(image_path):
        #     raise ValueError(f"Gambar dengan nama '{filename}' sudah ada.")

        # Simpan gambar jika belum ada
        image.save(image_path)

        return image_path
    except Exception as e:
        raise ValueError(f"Failed to save image: {e}")

def delete_image(filename, db_path):
    try:
        if not os.path.exists(db_path):
            raise ValueError(f"Folder '{db_path}' not found.")

        deleted_files = []
        for file in os.listdir(db_path):
            if file.startswith(filename):
                image_path = os.path.join(db_path, file)
                os.remove(image_path)
                deleted_files.append(file)

        if not os.listdir(db_path):
            os.rmdir(db_path)

        return deleted_files
    except Exception as e:
        raise ValueError(f"Failed to delete files: {e}")

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
        result = DeepFace.analyze(img_path = img, actions=['age', 'gender', 'emotion', 'race'])
        
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
        if 'image_query' not in request.files or 'image_list' not in request.files:
            return jsonify({'error': 'image_query and image_list array are required'}), 400
        
        # Mengambil gambar pencarian
        image_query = read_image(request.files['image_query'])

        # Buat direktori sementara untuk menyimpan kumpulan gambar
        with tempfile.TemporaryDirectory() as temp_dir:
            # Menyimpan gambar target di dalam direktori sementara
            for image_file in request.files.getlist('image_list'):
                image_path = os.path.join(temp_dir, image_file.filename)
                image_file.save(image_path)

            # Mencocokkan gambar menggunakan DeepFace.find()
            result = DeepFace.find(img_path = image_query, db_path = temp_dir, anti_spoofing = True)
            result = result[0].to_dict()

            # Additional
            result['basename'] = {}
            if result['identity'] and len(result['identity']) > 0:
                for i in result['identity']:
                    result['basename'][i] = os.path.basename(result['identity'][i])

            return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/find-most-similar', methods=['POST'])
def find_most_similar():
    try:
        # Memeriksa apakah parameter yang dibutuhkan ada di request
        if 'image_query' not in request.json or 'image_list' not in request.json:
            return jsonify({'error': 'image_query and image_list are required'}), 400
        
        # Mengambil file gambar pencarian dalam bentuk base64
        image_query = request.json['image_query']
        image_list = request.json['image_list']

        # Variabel untuk menyimpan hasil terbaik
        best_match = None
        best_score = float('inf')  # Inisialisasi dengan nilai tak terhingga
        
        # Loop untuk membandingkan setiap gambar dalam daftar
        for item in image_list:
            image_id = item['id']
            image_url = item['image']
            
            try:
                verification = DeepFace.verify(img1_path=image_query, img2_path=image_url, enforce_detection = False)
                score = verification['distance']

                # Simpan hasil terbaik jika lebih baik
                if score < best_score:
                    best_score = score
                    best_match = {
                        'id': image_id,
                        'image': image_url,
                        'verification': verification
                    }
                    # if verification['verified']:
                    #     break
            except Exception as e:
                print(f"Error verifying {image_url}: {e}")
                continue

        # Jika tidak ada hasil yang sesuai
        if best_match is None:
            return jsonify({'best_match': None, 'message': 'No matches found.'})
        
        return jsonify({'best_match': best_match})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/face-register', methods=['POST'])
def face_register():
    try:
        if 'id' not in request.json or 'name' not in request.json or 'image' not in request.json or 'db_path' not in request.json:
            return jsonify({'error': 'id, name, image and db_path are required'}), 400
        
        BASE_DB_PATH = 'faces/'

        id = request.json['id']
        name = request.json['name']
        image = request.json['image']
        db_path = request.json['db_path']

        faces = DeepFace.analyze(img_path = image, actions=['emotion'], anti_spoofing = True)
        if len(faces) > 1:
            return jsonify({'error': 'Multiple faces detected in the image'}), 400

        filename = f"{id}_{name}"
        full_path = BASE_DB_PATH + db_path
        image_path = save_image(image, filename, full_path)
        basename = os.path.basename(image_path)

        return jsonify({
            'message': 'Face registered successfully',
            'basename': basename,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete-face', methods=['POST'])
def delete_face():
    try:
        if 'id' not in request.json or 'name' not in request.json or 'db_path' not in request.json:
            return jsonify({'error': 'id, name and db_path are required'}), 400

        BASE_DB_PATH = 'faces/'

        id = request.json['id']
        name = request.json['name']
        db_path = request.json['db_path']

        filename = f"{id}_{name}"
        full_path = BASE_DB_PATH + db_path
        deleted_files = delete_image(filename, full_path)

        if not deleted_files:
            return jsonify({'error': f'No image found with filename "{filename}" in folder "{full_path}"'}), 404

        return jsonify({
            'message': f'Images with filename "{filename}" deleted successfully', 
            'deleted_files': deleted_files
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/best-match', methods=['POST'])
def best_match():
    try:
        if 'image' not in request.json or 'db_path' not in request.json:
            return jsonify({'error': 'image and db_path array are required'}), 400

        BASE_DB_PATH = 'faces/'

        image = request.json['image']
        db_path = BASE_DB_PATH + request.json['db_path']
        # db_path = "C:/Users/binta/Pictures/Kaggle/Faces/Faces"

        thershold = 0.68
        if 'threshold' in request.json and isinstance(request.json['threshold'], (float)):
            thershold = request.json['threshold']

        result = DeepFace.find(img_path = image, db_path = db_path, anti_spoofing = True, threshold = thershold)
        result = result[0].to_dict()

        if result['identity'] and len(result['identity']) > 0:
            for i in result['identity']:
                full_path = result['identity'][i]
                basename = os.path.basename(full_path)
                filename = os.path.splitext(basename)[0]
                [id, name] = filename.split('_', 1)

                result['identity'][i] = {}
                result['identity'][i]['id'] = id
                result['identity'][i]['name'] = name
                # result['identity'][i]['filename'] = filename
                result['identity'][i]['basename'] = basename
                # result['identity'][i]['full_path'] = full_path

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
