# import sys
# out = open('/Users/prince_13/Documents/projects/try/bakwass/final_/ output1.txt', 'w')
# sys.stdout = out
# sys.stderr = out

import cv2
import face_recognition
from ultralytics import YOLO
import os
import time
import pickle
import numpy as np
import datetime
import csv
from people_data import record

with open('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl', 'rb') as f:
    data = pickle.load(f)
    known_faces = [item['encodings'] for item in data]
    known_names = [item['name'] for item in data]
if len(np.array(known_faces).shape) == 3:
    known_faces = np.squeeze(known_faces)   
# print(known_names,known_faces)    
model = YOLO('/Users/prince_13/Documents/projects/try/bakwass/final_/yolov8n-face-lindevs.pt')
input_video = cv2.VideoCapture(0)
# desired_fps = 60
# input_video.set(cv2.CAP_PROP_FPS, desired_fps)
count = 0

while True:
    current_time = datetime.datetime.now()
    # t2 = time.time()
    count += 1
    ret, frame = input_video.read()
    if not ret:
        break
    # t = time.time()
    result = model.predict(frame,verbose = False)
    # print(time.time()-t)
    # print("here")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    count_known = 0
    count_unknown = 0
    match = [False]*len(known_names)
    for box in result[0].boxes:
        if box.cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # t1 = time.time()
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y1, x2, y2, x1)])[0]
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            # print(time.time()-t1)
            name = "Unknown"
            count_unknown += 1
            if np.any(matches):
                first_match_index = np.where(matches)[0][0]
                match[first_match_index] = True
                name = known_names[first_match_index]
                count_known += 1
                count_unknown -= 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/people_data/data.csv','a+') as f:
        writer = csv.writer(f)
        match.append(current_time)
        writer.writerow(match)
    record()

    cv2.putText(frame, f'Known: {count_known}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.putText(frame, f'Unknown: {count_unknown}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.putText(frame, f'Total: {count_known + count_unknown}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('Video', frame)
    # print(time.time()-t2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        with open('people_data/in_out_present_status.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow([False]*len(known_names))
        with open('people_data/data.csv','w') as f :
            pass
        break
    
input_video.release()
cv2.destroyAllWindows()