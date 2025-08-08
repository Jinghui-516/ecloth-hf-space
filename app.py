import os
from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import mediapipe as mp
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def process_images(model_path, clothing_path, output_path):
    model_img = cv2.imread(model_path, cv2.IMREAD_UNCHANGED)
    clothing_img = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)

    model_rgb = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
    results = pose.process(model_rgb)
    if not results.pose_landmarks:
        return False

    landmarks = results.pose_landmarks.landmark
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    h, w, _ = model_img.shape
    lx, ly = int(l_shoulder.x * w), int(l_shoulder.y * h)
    rx, ry = int(r_shoulder.x * w), int(r_shoulder.y * h)

    shoulder_width = int(((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5)
    scale_factor = shoulder_width / clothing_img.shape[1]
    new_size = (int(clothing_img.shape[1] * scale_factor), int(clothing_img.shape[0] * scale_factor))
    resized_clothing = cv2.resize(clothing_img, new_size)

    x_offset = lx - resized_clothing.shape[1] // 4
    y_offset = ly - resized_clothing.shape[0] // 4

    for c in range(0, 3):
        model_img[y_offset:y_offset+resized_clothing.shape[0],
                  x_offset:x_offset+resized_clothing.shape[1], c] = \
            resized_clothing[:, :, c] * (resized_clothing[:, :, 3] / 255.0) + \
            model_img[y_offset:y_offset+resized_clothing.shape[0],
                      x_offset:x_offset+resized_clothing.shape[1], c] * (1.0 - resized_clothing[:, :, 3] / 255.0)

    cv2.imwrite(output_path, model_img)
    return True

@app.route("/", methods=["GET", "POST"])
def index():
    result_url = None
    if request.method == "POST":
        model = request.files["model_image"]
        clothing = request.files["clothing_image"]
        if model and clothing:
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model.filename))
            clothing_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(clothing.filename))
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], "result.png")

            model.save(model_path)
            clothing.save(clothing_path)

            success = process_images(model_path, clothing_path, result_path)
            if success:
                result_url = "/static/uploads/result.png"
            else:
                result_url = None
    return render_template("index.html", result_url=result_url)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
