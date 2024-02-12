from gesture_recognition.gesture_recognition import GestureRecognizer
import cv2
import argparse
import numpy as np
import requests
import sys
import base64
import json
import mediapipe as mp

from autocorrect import Speller
from utils import load_model, save_gif, save_video
from utils import calc_landmark_list, draw_landmarks, draw_info_text

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Autocorrect Word
spell = Speller(lang='en')

# Colors RGB Format
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
BLUE   = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_HANDS = 1                   #@param {type: "integer"}
min_detection_confidence = 0.6  #@param {type:"slider", min:0, max:1, step:0.01}
min_tracking_confidence  = 0.5  #@param {type:"slider", min:0, max:1, step:0.01}

MODEL_PATH = "./classifier"
model_letter_path = f"{MODEL_PATH}/classify_letter_model.p"
model_number_path = f"{MODEL_PATH}/classify_number_model.p"

if __name__ == '__main__':
    recognizer = GestureRecognizer()
    opt = recognizer.parse_opt()
    saveGIF = opt.gif
    saveVDO = opt.video
    source = opt.source

    global TIMING, autocorrect
    TIMING = opt.timing
    autocorrect = opt.autocorrect
    print(f"Timing Threshold is {TIMING} frames.")
    print(f"Using Autocorrect: {autocorrect}")

    # Get video source path
    if source == None or source.isnumeric():
        video_path = 0
    else:
        video_path = source

    # Webcam Arguments
    fps = opt.fps
    webcam_width = opt.width
    webcam_height = opt.height

    _output = [[], []]
    output = []
    quitApp = False

    frame_array = []
    current_hand = 0
    numberMode = False

    # Webcam Input
    if video_path == 0:
        capture = cv2.VideoCapture(video_path)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FPS, fps)

        # Press 'r' if you are ready
        while True:
            success, frame = capture.read()
            frame = cv2.flip(frame, 1)

            # setup text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'Ready? Press "R"'

            # get boundary of this text
            textsize = cv2.getTextSize(text, font, 1.3, 3)[0]

            # get coords based on boundary
            textX = (frame.shape[1] - textsize[0]) // 2
            textY = (frame.shape[0] + textsize[1]) // 2

            cv2.putText(
                frame, text, (textX, textY),
                font, 1.3, GREEN, 3,
                cv2.LINE_AA
            )
            cv2.imshow('Gesture Recognition:', frame)

            # User Input from Keyboard
            key = cv2.waitKey(5) & 0xFF
            if key == ord('r'):
                break

            # Press 'Esc' to quit
            if key == 27:
                quitApp = True
                break

        # Remove starter window
        cv2.destroyAllWindows()
        if quitApp == True:
            capture.release()
            quit()

    # Video Input
    else:
        capture = cv2.VideoCapture(video_path)

    with mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=MAX_HANDS
    ) as hands:
        while capture.isOpened():
            success, image = capture.read()
            if not success:
                if video_path == 0:
                    print("Ignoring empty camera frame.")
                    continue
                else:
                    print("Video ends.")
                    break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            try:
                #API Call
                #image, prediction = recognizer.recognize_gesture(img_str, numberMode, model_letter_path, model_number_path)
                # Serialize image
                _, buffer = cv2.imencode('.jpg', image)
                img_str = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string

                payload = json.dumps({"image": img_str})
                response = requests.post('http://127.0.0.1:5000/process_frame',
                                         headers={'Content-Type': 'application/json'}, data=payload)

                if response.status_code == 200:
                    response = response.json()  # Assuming the server returns prediction results in JSON
                    prediction = response.get('prediction')
                    image_output_str = response.get('image_output_str')

                    img_bytes = base64.b64decode(image_output_str)  # Decode base64 string to bytes
                    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)  # Convert bytes to a numpy array
                    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode numpy array to image
                else:
                    print(f"Failed to process frame. Status Code: {response.status_code}")
            except Exception as error:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(f"{error}, line {exc_tb.tb_lineno}")

            # Show output in Top-Left corner
            output_text = str(output)
            output_size = cv2.getTextSize(output_text, FONT, 0.5, 2)[0]
            cv2.rectangle(image, (5, 0), (10 + output_size[0], 10 + output_size[1]), YELLOW, -1)
            cv2.putText(image, output_text, (10, 15), FONT, 0.5, BLACK, 2)

            mode_text = f"Number: {numberMode}"
            mode_size = cv2.getTextSize(mode_text, FONT, 0.5, 2)[0]
            cv2.rectangle(image, (5, 45), (10 + mode_size[0], 10 + mode_size[1]), YELLOW, -1)
            cv2.putText(image, mode_text, (10, 40), FONT, 0.5, BLACK, 2)

            # Save each frames to GIF
            cv2.imshow('American Sign Language', image)
            frame_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(5) & 0xFF

            # Press 'Esc' to quit
            if key == 27:
                break

            # Press 'Backspace' to delete last word
            if key == 8:
                output.pop()

            # Press 's' to save result
            if key == ord('s'):
                saveGIF = True
                saveVDO = True
                break

            # Press 'm' to change mode between alphabet and number
            if key == ord('m'):
                numberMode = not numberMode

            # Press 'c' to clear output
            if key == ord('c'):
                output.clear()

    cv2.destroyAllWindows()
    capture.release()