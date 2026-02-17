from threading import Thread
import os
import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance
from imutils import face_utils
import pygame

#Alarm sound file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_SOUND_PATH = os.path.join(BASE_DIR, "nuclear_alarm.wav")

#Face and eye perception models
LANDMARKS_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
FRONTAL_FACE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_alt.xml")

predictor = dlib.shape_predictor(LANDMARKS_PATH)
face_cascade = cv2.CascadeClassifier(FRONTAL_FACE_PATH)

#Alarm control variables
alarm_playing = False

def sound_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        pygame.mixer.init()
        pygame.mixer.music.load(ALARM_SOUND_PATH)
        pygame.mixer.music.play(-1)

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        alarm_playing = False

def detect(img, cascade=face_cascade, minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise Exception("There was a problem loading your Haar Cascade xml file.")
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)

    if len(rects) == 0:
        return []

    rects[:, 2:] += rects[:, :2]

    return rects

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

def cropEyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    te = detect(gray, minimumFeatureSize=(80, 80))

    if len(te) == 0:
        return frame, None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te

    face_rect = dlib.rectangle(left=int(face[0]), top=int(face[1]),
                               right=int(face[2]), bottom=int(face[3]))

    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(face_rect)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, "Face #1", (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)

    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    return frame, ear

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def monitor_breath_and_drowsiness():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    BREATH_THRESHOLD = 100
    BREATH_INTERVAL_THRESHOLD = 2

    numberOfBreaths = 0
    last_breath_time = None
    noBreathTime = 0
    eye_closed_time = 0
    eye_close_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 2)
        fgmask = fgbg.apply(blur)
        pixelCounter = np.count_nonzero(fgmask)

        current_time = time.time()

        #Breath Detection
        if pixelCounter > BREATH_THRESHOLD:
            if last_breath_time is None or (current_time - last_breath_time) > BREATH_INTERVAL_THRESHOLD:
                numberOfBreaths += 1
                last_breath_time = current_time
                noBreathTime = 0
        else:
            if last_breath_time:
                noBreathTime = current_time - last_breath_time

        frame, ear = cropEyes(frame)

        if ear is not None and ear < 0.25:
            if eye_close_start_time is None:
                eye_close_start_time = current_time
            eye_closed_time = current_time - eye_close_start_time
        else:
            eye_close_start_time = None
            eye_closed_time = 0

        situation = "Awake" if eye_closed_time < 10 else "Sleeping"

        # Alarm conditions
        if noBreathTime > 10 or eye_closed_time >= 5:
            sound_alarm()
        else:
            stop_alarm()

        #Printing information to the monitoring
        cv2.putText(frame, f"Number of Breaths: {numberOfBreaths}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Situation: {situation}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Time not Breathing: {noBreathTime:.1f} sn", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Eye Closure Duration: {eye_closed_time:.1f} sn", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Breath View", fgmask)
        cv2.imshow("AI Processed View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    monitor_breath_and_drowsiness()
