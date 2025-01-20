from ultralytics import YOLO
import cv2
import pickle
import numpy as np
import time  
import datetime   
import face_recognition
import csv
from people_data import record
with open('faces_input/common_encodings.pkl', 'rb') as f:
    data = pickle.load(f)
    known_faces = [item['encodings'] for item in data]
    known_names = [item['name'] for item in data]
if len(np.array(known_faces).shape) == 3:
    known_faces = np.squeeze(known_faces) 
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n-face-lindevs.pt')

input_video = cv2.VideoCapture(0)

total_people = 0
total_faces = 0

def calculate_iou(box1, box2):
    # print(box1[0],box2[0])
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0
    return iou

presence_list = [False]*(len(known_names)+1)
count = 0
while True:
    current_time = datetime.datetime.now()
    # t = time.time()
    presence_list[-1] = current_time    
    dict_people = {}
    ret, frame = input_video.read()
    if not ret:
        break
    # tim = time.time()
    result1 = model1.predict(frame,verbose = False)
    # print('1',time.time()-tim)
    # tim = time.time()
    result2 = model2.predict(frame,verbose = False)
    # print('2',time.time()-tim)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    unknown_count = 0
    people  = 0
    ppl = []
    faces = []
    faces_count = 0
    # tim = time.time()
    #making list
    for box in result1[0].boxes:
        if box.cls == 0:
            people += 1
            ppl.append(box.xyxy[0].tolist())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for box in result2[0].boxes:
        if box.cls == 0:
            faces.append(box.xyxy[0].tolist())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # print(time.time()-tim)
    flag = False
    # finding bounding boxes matching

    for i,a in enumerate(ppl) :
        # print(i,a)
        max1 = 0
        a1 = 0
        for j in faces :
            if calculate_iou(a,j) >= max1:
                max1 = calculate_iou(a,j)
                flag = True
                a1 = j
        if flag == True :
            faces_count += 1
            dict_people[faces_count] = [a1,i]
            x1, y1 = int(a[0]), int(a[1])
            x2, y2 = int(a1[0]), int(a1[1])
        
        # Draw line with integer coordinates
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            flag = False
        else :
            unknown_count += 1
            

    temp_list = [False]*len(known_names)
    if people != total_people or faces_count != total_faces :
        count = 0
    # tim = time.time()
    if people != total_people or faces_count != total_faces or count<=5:
        count += 1
        print('not equal')
        for i , lst in  enumerate(dict_people.values()):
            x1, y1, x2, y2 = map(int, lst[0])
            # t1 = time.time()
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y1, x2, y2, x1)])[0]
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            print(matches)
            name = "Unknown"
            unknown_count += 1
            if np.any(matches):
                first_match_index = np.where(matches)[0][0]
                temp_list[first_match_index] = True
                name = known_names[first_match_index]
                # count_known += 1
                unknown_count -= 1
            print (name)
    
        total_people = people
        total_faces = faces_count
        presence_list = temp_list
        presence_list.append(current_time)
    # print('see',temp_list)
    # print(time.time()-tim)
    with open('log/unknown.txt','a') as f:
        f.write(f'{unknown_count} unknown people at {current_time}\n')
    with open('people_data/data.csv','a+') as f:
        writer = csv.writer(f)
        writer.writerow(presence_list)
    record()
    cv2.imshow('frame', frame)
    # print('3',time.time()-t)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        with open('people_data/in_out_present_status.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow([False]*len(known_names))
        with open('people_data/data.csv','w') as f :
            pass
        break
input_video.release()
cv2.destroyAllWindows()          
    
