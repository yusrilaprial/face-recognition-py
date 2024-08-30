# Set Up
- Install Virtual Environment: `python -m venv .venv`
- Run Virtual Environment: `.venv\Scripts\activate`
- Exit Virtual Environment: `deactivate`
- Install Packages: `pip install -r requirements.txt`
- Run App: `python face_recognition_service.py`

# Error Case
- Download dlib binary and install manual `python -m pip install dlib-19.22.99-cp310-cp310-win_amd64.whl` (this only for python 3.10)
- If you get error (RuntimeError: Unsupported image type, must be 8bit gray or RGB image) just downgrade `python -m pip install numpy==1.26.4`