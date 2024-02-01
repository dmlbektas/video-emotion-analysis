import numpy as np
import cv2
from keras.models import load_model
import os
import time
import matplotlib.pyplot as plt

dnnNetwork_face = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10SSD.caffemodel")

model_emotion = load_model('emotion_detection_model_100epochs.h5', compile=False)

#videoCapture = cv2.VideoCapture(0)
videoCapture = cv2.VideoCapture("cindirella.mp4")

shiftValue = 20
resizeX = 460
resizeY = 300
thresholdValue = 0.3

class_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_data = {'Angry': [], 'Fear': [], 'Happy': [], 'Neutral': [], 'Sad': [], 'Surprise': []}
total_emotion_counts = {label: 0 for label in class_labels}

# Dosya
output_folder = 'faces_with_emotion6'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    frame = cv2.resize(frame, (resizeX, resizeY))
    (h, w) = frame.shape[:2]

    dnnBlobObject_face = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnnNetwork_face.setInput(dnnBlobObject_face)
    resultDetections_face = dnnNetwork_face.forward()

    max_emotion_label = None
    max_emotion_confidence = 0.0

    for i in range(0, resultDetections_face.shape[2]):
        confidence_face = resultDetections_face[0, 0, i, 2]

        if confidence_face < thresholdValue:
            continue

        resultArea_face = resultDetections_face[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX_face, startY_face, endX_face, endY_face) = resultArea_face.astype("int")

        # bounding box
        cv2.rectangle(frame, (startX_face, startY_face), (endX_face, endY_face), (0, 255, 255), 2)

        face_roi = frame[startY_face:endY_face, startX_face:endX_face]

        if face_roi is not None and face_roi.size > 0 and face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
            # Duygu analizi
            face_roi_resized = cv2.resize(face_roi, (48, 48))
            face_roi_gray = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)
            face_roi_normalized = np.reshape(face_roi_gray, (1, 48, 48, 1)) / 255.0
            emotion_prediction = model_emotion.predict(face_roi_normalized)

            emotion_label = class_labels[np.argmax(emotion_prediction)]
            emotion_percent = np.max(emotion_prediction) * 100

            #  duygu sınıfı etiketi
            y_face = startY_face - 20 if startY_face - 20 > 20 else startY_face + 20
            cv2.putText(frame, f"{emotion_label} - {emotion_percent:.2f}%", (startX_face, y_face),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)




            current_time = videoCapture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            face_filename = f"{output_folder}/face_{emotion_label}_{current_time:.2f}.png"

            cv2.imwrite(face_filename, face_roi)

            emotion_data[emotion_label].append((videoCapture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, emotion_percent))
            total_emotion_counts[emotion_label] += 1

            if emotion_percent > max_emotion_confidence:
                max_emotion_confidence = emotion_percent
                max_emotion_label = emotion_label



    cv2.imshow("Video Duygu Analizi", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Grafik
fig, ax = plt.subplots(figsize=(10, 6)) #w,h

for emotion_label, data_points in emotion_data.items():
    if not data_points:
        continue  # Skip empty data points

    timestamps, confidences = zip(*data_points)
    ax.scatter(timestamps, confidences, label=f"{emotion_label} ({sum(confidences) * 0.001:.2f}%)")

ax.set(xlabel='Time (seconds)', ylabel='Emotion Confidence (%)', title='Emotion Analysis Over Time')
ax.legend()

fig, ax = plt.subplots()

ax.bar(total_emotion_counts.keys(), total_emotion_counts.values(), color='skyblue')
ax.set(xlabel='Emotion Labels', ylabel='Total Count', title='Total Emotion Counts')

most_detected_emotion = max(total_emotion_counts, key=total_emotion_counts.get)

if most_detected_emotion in ['Happy', 'Surprise']:
    predicted_genre = "Komedi"
elif most_detected_emotion in ['Sad', 'Fear', 'Angry']:
    predicted_genre = "Dram/Macera"
elif most_detected_emotion in ['Sad', 'Fear', 'Surprise', 'Angry', 'Happy']:
    predicted_genre = "Macera"
elif most_detected_emotion in ['Fear','Angry', 'Surprise', 'Sad']:
    predicted_genre = "Gerilim"
elif most_detected_emotion in ['Neutral', 'Happy', 'Fear','Sad', ]:
    predicted_genre = "fantastik"
else:
    predicted_genre = "Tanımlanamadı"

print(f"Film Türü Tahmini: {predicted_genre}")

plt.show()

cv2.destroyAllWindows()
videoCapture.release()
