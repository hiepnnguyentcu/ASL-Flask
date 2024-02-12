import cv2
import argparse
import numpy as np
import mediapipe as mp
import base64

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

MODEL_PATH = "../classifier"
model_letter_path = f"{MODEL_PATH}/classify_letter_model.p"
model_number_path = f"{MODEL_PATH}/classify_number_model.p"

current_hand = 0
numberMode = False
_output = [[], []]
output = []
quitApp = False

frame_array = []
current_hand = 0
numberMode = False
class GestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    # Customize your input
    @staticmethod
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--source', type=str, default=None, help='Video Path/0 for Webcam')
        parser.add_argument('-a', '--autocorrect', action='store_true', help='Autocorrect Misspelled Word')
        parser.add_argument('-g', '--gif', action='store_true', help='Save GIF Result')
        parser.add_argument('-v', '--video', action='store_true', help='Save Video Result')
        parser.add_argument('-t', '--timing', type=int, default=8, help='Timing Threshold')
        parser.add_argument('-wi', '--width',  type=int, default=800, help='Webcam Width')
        parser.add_argument('-he', '--height', type=int, default=600, help='Webcam Height')
        parser.add_argument('-f', '--fps', type=int, default=30, help='Webcam FPS')
        opt = parser.parse_args()
        return opt

    def get_output(self, idx):
        global _output, output, autocorrect, TIMING
        key = []
        for i in range(len(_output[idx])):
            character = _output[idx][i]
            counts = _output[idx].count(character)

            # Add character to key if it exceeds 'TIMING THRESHOLD'
            if (character not in key) or (character != key[-1]):
                if counts > TIMING:
                    key.append(character)

        # Add key character to output text
        text = ""
        for character in key:
            if character == "?":
                continue
            text += str(character).lower()

        # Autocorrect Misspelled Word
        text = spell(text) if autocorrect else text

        # Add word to output list
        if text != "":
            _output[idx] = []
            output.append(text.title())
        return None


    def recognize_gesture(self, image, numberMode, model_letter_path, model_number_path):
        #Image preprocessing
        with mp_hands.Hands(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                max_num_hands=MAX_HANDS
        ) as hands:

            # To improve performance, optionally mark the image as not writeable to pass by reference
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Recognition
            opt = self.parse_opt()
            global TIMING, autocorrect
            TIMING = opt.timing
            autocorrect = opt.autocorrect
            global mp_drawing, current_hand
            global output, _output

            multi_hand_landmarks = results.multi_hand_landmarks
            multi_handedness = results.multi_handedness

            # Load Classification Model
            letter_model = load_model(model_letter_path)
            number_model = load_model(model_number_path)

            _gesture = []
            data_aux = []

            prediction = ''

            # Number of hands
            isIncreased = False
            isDecreased = False

            if current_hand != 0:
                if results.multi_hand_landmarks is None:
                    isDecreased = True
                else:
                    if len(multi_hand_landmarks) > current_hand:
                        isIncreased = True
                    elif len(multi_hand_landmarks) < current_hand:
                        isDecreased = True

            if results.multi_hand_landmarks:
                h, w, _ = image.shape
                for idx in reversed(range(len(multi_hand_landmarks))):
                    current_select_hand = multi_hand_landmarks[idx]
                    handness = multi_handedness[idx].classification[0].label

                    # mp_drawing.draw_landmarks(image, current_select_hand, mp_hands.HAND_CONNECTIONS)
                    landmark_list = calc_landmark_list(image, current_select_hand)
                    image = draw_landmarks(image, landmark_list)

                    # Get (x, y) coordinates of hand landmarks
                    x_values = [lm.x for lm in current_select_hand.landmark]
                    y_values = [lm.y for lm in current_select_hand.landmark]

                    # Get Minimum and Maximum Values
                    min_x = int(min(x_values) * w)
                    max_x = int(max(x_values) * w)
                    min_y = int(min(y_values) * h)
                    max_y = int(max(y_values) * h)

                    # Draw Text Information
                    cv2.putText(image, f"Hand No. #{idx}", (min_x - 10, max_y + 30), FONT, 0.5, GREEN, 2)
                    cv2.putText(image, f"{handness} Hand", (min_x - 10, max_y + 60), FONT, 0.5, GREEN, 2)

                    # Flip Left Hand to Right Hand
                    if handness == 'Left':
                        x_values = list(map(lambda x: 1 - x, x_values))
                        min_x -= 10

                    # Create Data Augmentation for Corrected Hand
                    for i in range(len(current_select_hand.landmark)):
                        data_aux.append(x_values[i] - min(x_values))
                        data_aux.append(y_values[i] - min(y_values))

                    if not numberMode:
                        # Alphabets Prediction
                        prediction = letter_model.predict([np.asarray(data_aux)])
                        gesture = str(prediction[0]).title()
                        gesture = gesture if gesture != 'Unknown_Letter' else '?'
                    else:
                        # Numbers Prediction
                        prediction = number_model.predict([np.asarray(data_aux)])
                        gesture = str(prediction[0]).title()
                        gesture = gesture if gesture != 'Unknown_Number' else '?'

                    # Draw Bounding Box
                    cv2.rectangle(image, (min_x - 20, min_y - 10), (max_x + 20, max_y + 10), BLACK, 4)
                    image = draw_info_text(image, [min_x - 20, min_y - 10, max_x + 20, max_y + 10], gesture)

                    _gesture.append(gesture)

            # Number of hands is decreasing, create "SPACE"
            if isDecreased == True:
                if current_hand == 1:
                    self.get_output(0)

            # Number of hands is the same, append gesture
            else:
                if results.multi_hand_landmarks is not None:
                    _output[0].append(_gesture[0])

            # Track hand numbers
            if results.multi_hand_landmarks:
                current_hand = len(multi_hand_landmarks)
            else:
                current_hand = 0

            return image, prediction

    @staticmethod
    def serialize_landmarks(results):
        if not results.multi_hand_landmarks:
            return []

        hands_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [{'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                         for landmark in hand_landmarks.landmark]
            hands_data.append(landmarks)

        return hands_data



