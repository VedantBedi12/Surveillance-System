from ultralytics import YOLO
import av
import cv2
import torch
import os

def extract_frames(video):
    model = YOLO("yolov8n.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device=device)



    # Initialize bbox_list as a list of dictionaries
    bbox_list = []
    changes = []
    prev_count = None
    person_frames_count = 0
    frame_index = 0

    container = av.open(video)
    stream = container.streams.video[0]
    stream.thread_type = "SLICE"

    for frame in container.decode(stream):
        frame = frame.reformat(640, 480, format="bgr24")
        frame_array = frame.to_ndarray()
        #cv2.imwrite(f"{output_folder}/frame_{frame_index}.jpg", frame_array)
        frame_index += 1

        # Initialize lists to store bboxes and scores for the current frame
        bboxes = []
        scores = []

        results = model.predict(frame_array, stream=True)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                cl = boxes.cls
                conf = boxes.conf
                val = (cl == 0) & (conf > 0.5)  # Filter for persons (class 0) with confidence > 0.5

                # Get bounding boxes and confidence scores for persons
                for i in range(len(val)):
                    if val[i]:  # If the detection is a person with confidence > 0.5
                        bbox = boxes.xyxy[i].cpu().numpy()  # Get bbox in [x1, y1, x2, y2] format
                        score = conf[i].cpu().numpy()  # Get confidence score
                        bboxes.append(bbox.tolist())  # Convert to list and add to bboxes
                        scores.append(float(score))  # Convert to float and add to scores

                current_count = torch.sum(val).item()  # Count number of people (class 0)
                if prev_count is None and current_count >= 1:
                    changes.append({
                        'previous_count': prev_count,
                        'current_count': current_count,
                        'frame': frame_index
                    })

                if prev_count is not None and current_count != prev_count:
                    changes.append({
                        'previous_count': prev_count,
                        'current_count': current_count,
                        'frame': frame_index
                    })

                prev_count = current_count

        # Store bboxes and scores in a dictionary for the current frame
        if not bboxes:
            bbox_list.append(None)  # No persons detected
        else:
            bbox_list.append({
                'bboxes': bboxes,
                'scores': scores
            })

    fps = stream.average_rate  # Frame rate of the video

    return changes, bbox_list
