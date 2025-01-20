from database import create_database
from extract_frames import extract_frames
from detector import detect_and_recognize_faces
from tracker import track_faces

db,collection=create_database()
changes,bbox_list=extract_frames("vid.mp4")
face_data=detect_and_recognize_faces("vid.mp4",changes,collection)
bbox_list = {i: item for i, item in enumerate(bbox_list)}
track_faces(face_data,"vid.mp4",yolo_detections=bbox_list)