from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
from process_image import process_image
import json

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

UPLOAD_FOLDER = os.path.abspath(os.getenv("UPLOAD_DIR", "uploads"))
OUTPUT_FOLDER = os.path.abspath(os.getenv("OUTPUT_DIR", "outputs"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/process-image', methods=['POST'])
def upload_and_process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # 파일 경로 설정
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{file.filename}")

    try:
        # 파일 저장
        file.save(file_path)

        # 이미지 처리 및 JSON 데이터 생성
        json_data = process_image(file_path, output_path)

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    # 처리된 이미지 URL과 JSON 데이터를 반환
    return jsonify({
        "processedImageUrl": f"http://{os.getenv('PUBLIC_HOST', '127.0.0.1')}:{os.getenv('PORT', '5000')}/outputs/processed_{file.filename}",  # [변경]
        "result": json.loads(json_data) 
    })

@app.route('/outputs/<filename>', methods=['GET'])
def serve_processed_image(filename):
    """
    처리된 이미지를 반환.
    """
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=True)