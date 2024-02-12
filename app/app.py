from flask import Blueprint, request, jsonify
from gesture_recognition.gesture_recognition import GestureRecognizer
import mediapipe as mp
import cv2
import numpy as np
import base64

mp_hands = mp.solutions.hands
MAX_HANDS = 1                   #@param {type: "integer"}
min_detection_confidence = 0.6  #@param {type:"slider", min:0, max:1, step:0.01}
min_tracking_confidence  = 0.5  #@param {type:"slider", min:0, max:1, step:0.01}

MODEL_PATH = "../classifier"
model_letter_path = "/Users/hiepnnguyen/ASL-Finger-Spelling-To-Text/classifier/classify_letter_model.p"
model_number_path = "/Users/hiepnnguyen/ASL-Finger-Spelling-To-Text/classifier/classify_number_model.p"

numberMode = False

main = Blueprint('main', __name__)
@main.route('/process_frame', methods=['POST'])
def process_frame():
    recognizer = GestureRecognizer()
    data = request.get_json()

    # Step 1: Decode the Base64 string to bytes
    if 'image' in data:
        img_str = data['image']
        img_bytes = base64.b64decode(img_str)  # Decode base64 string to bytes
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)  # Convert bytes to a numpy array
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode numpy array to image

        image_output, prediction = recognizer.recognize_gesture(image, numberMode, model_letter_path, model_number_path)

        prediction_str = ''
        if prediction:
            prediction_str = prediction[0]

        #Serialize image_output
        _, buffer = cv2.imencode('.jpg', image_output)
        image_output_str = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string

        return jsonify(
            {'message': 'Image received and processed',
             'image_output_str': image_output_str,
             'prediction': prediction_str
             })
    return jsonify({'error': 'Missing image data'}), 400
