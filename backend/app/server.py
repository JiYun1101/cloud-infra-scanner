# file: server.py
from __future__ import annotations

import os
import json
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename  # why: 파일명 정규화
from dotenv import load_dotenv
from process_image import process_image

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.getenv("FRONT_ORIGIN", "http://localhost:3000")}})

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.abspath(os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads")))
OUTPUT_FOLDER = os.path.abspath(os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs")))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/process-image", methods=["POST"])
def upload_and_process_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # 1) 파일명 정규화(공백/한글/특수문자 → 안전한 이름)
    raw_name = file.filename
    name_only, ext = os.path.splitext(raw_name)
    safe_name = secure_filename(name_only) + ext.lower()

    # 2) 경로 생성
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    output_name = f"processed_{safe_name}"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    try:
        # 3) 저장 + 처리
        file.save(file_path)
        json_data = process_image(file_path, output_path)  # 내부에서 유니코드 안전 저장 사용

        # 4) 응답
        public_host = os.getenv("PUBLIC_HOST", "127.0.0.1")
        port = os.getenv("PORT", "5000")
        return jsonify({
            "processedImageUrl": f"http://{public_host}:{port}/outputs/{output_name}",
            "result": json.loads(json_data),
        })
    except Exception as e:
        # why: 실패 지점 파악을 위해 경로 로그 포함
        return jsonify({
            "error": f"Error processing image: {str(e)}",
            "paths": {"upload": file_path, "output": output_path}
        }), 500


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_processed_image(filename: str):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)


@app.after_request
def add_cors_headers(resp):
    # why: 프론트 캔버스에서 이미지 픽셀 읽기 시 CORS 문제 예방
    resp.headers["Access-Control-Allow-Origin"] = os.getenv("FRONT_ORIGIN", "http://localhost:3000")
    resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=True)
