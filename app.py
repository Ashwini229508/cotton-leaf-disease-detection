from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import torch
from swin_transformer_model import SwinTransformerPredictor

app = Flask(__name__)
CORS(app)

EFFICIENTNET_PATH = "cotton_model.keras"
SWIN_TRANSFORMER_PATH = "swin_transformer_cotton.pth"

class_names = [
    "aphids",
    "bacterial_blight",
    "curl_virus",
    "fussarium_wilt",
    "healthy"
]
efficientnet_model = None

try:
    efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_PATH)
    print("EfficientNet model loaded successfully")
except Exception as e:
    print("Error loading EfficientNet:", e)


swin_model = None

try:
    swin_model = SwinTransformerPredictor()
    swin_model.load_custom_weights(SWIN_TRANSFORMER_PATH)
    print("Swin Transformer model loaded successfully")
except Exception as e:
    print("Error loading Swin Transformer:", e)


@app.route("/")
def home():
    return "Cotton Disease Detection API Running"


def predict_with_efficientnet(img):

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = efficientnet_model.predict(img_array)

    class_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]) * 100)

    return {
        "prediction": class_names[class_index],
        "confidence": round(confidence, 2),
        "probabilities": {
            class_names[i]: float(pred * 100)
            for i, pred in enumerate(predictions[0])
        }
    }


def predict_with_swin(img):

    return swin_model.predict(img)



def ensemble_prediction(img):

    eff_result = predict_with_efficientnet(img)
    swin_result = predict_with_swin(img)

    predictions = [
        eff_result["prediction"],
        swin_result["prediction"]
    ]

    confidences = [
        eff_result["confidence"],
        swin_result["confidence"]
    ]

    
    from collections import Counter
    final_prediction = Counter(predictions).most_common(1)[0][0]

    avg_confidence = sum(confidences) / len(confidences)

    return {
        "ensemble_prediction": final_prediction,
        "ensemble_confidence": round(avg_confidence, 2),
        "efficientnet": eff_result,
        "swin_transformer": swin_result
    }


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        file = request.files["file"]

        img = Image.open(file).convert("RGB")

        result = ensemble_prediction(img)

        result["mode"] = "hybrid_ensemble"

        result["models_used"] = [
            "efficientnet",
            "swin_transformer"
        ]

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        })



@app.route("/models", methods=["GET"])
def get_models():

    return jsonify({
        "efficientnet": efficientnet_model is not None,
        "swin_transformer": swin_model is not None,
        "classes": class_names
    })



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)