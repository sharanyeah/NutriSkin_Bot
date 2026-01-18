import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# -----------------------------
# APP SETUP
# -----------------------------
app = Flask(__name__)

# -----------------------------
# UPLOAD CONFIG
# -----------------------------
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# LOAD MODEL (ONCE)
# -----------------------------
MODEL_PATH = "mobilnetV3model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -----------------------------
# CLASS LABELS
# -----------------------------
class_labels = [
    "Acne and Rosacea Photos",
    "Atopic Dermatitis Photos",
    "Eczema Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Psoriasis pictures Lichen Planus and related diseases",
]

# -----------------------------
# DISEASE → DEFICIENCY MAP
# -----------------------------
disease_data = {
    "Acne and Rosacea Photos": {
        "deficiency": ["Zinc", "Vitamin A", "Vitamin E", "Omega-3"],
        "advice": "Increase Zinc, Vitamin A, and Omega-3 through nuts, carrots, and fatty fish."
    },
    "Eczema Photos": {
        "deficiency": ["Vitamin D", "Omega-3", "Zinc", "Vitamin B6"],
        "advice": "Include fish, eggs, seeds, and sunlight exposure."
    },
    "Atopic Dermatitis Photos": {
        "deficiency": ["Vitamin D", "Zinc", "Omega-3"],
        "advice": "Consume salmon, flaxseeds, walnuts, and fortified foods."
    },
    "Light Diseases and Disorders of Pigmentation": {
        "deficiency": ["Vitamin B12", "Copper", "Zinc"],
        "advice": "Add dairy, shellfish, leafy greens, and legumes."
    },
    "Psoriasis pictures Lichen Planus and related diseases": {
        "deficiency": ["Vitamin D", "Omega-3", "Selenium", "Vitamin E"],
        "advice": "Eat oily fish, sunflower seeds, nuts, and whole grains."
    }
}

# -----------------------------
# HELPERS
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)
    return class_labels[np.argmax(preds)]


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    disease = predict_disease(file_path)

    disease_info = disease_data.get(
        disease,
        {"deficiency": [], "advice": "No specific advice available."}
    )

    return jsonify({
        "disease": disease,
        "deficiency": disease_info["deficiency"],
        "advice": disease_info["advice"]
    })


@app.route("/get-advice", methods=["GET"])
def get_advice():
    disease = request.args.get("disease")
    if disease not in disease_data:
        return jsonify({"error": "No advice available"}), 400

    return jsonify({"advice": disease_data[disease]["advice"]})


# -----------------------------
# RENDER ENTRY POINT (IMPORTANT)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
