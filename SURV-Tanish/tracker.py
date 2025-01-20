import av
import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# Helper function to calculate IoU
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def track_faces(face_data, video_path, yolo_detections, output_video_path="output_video.mp4"):
    # Initialize the tracker
    tracker = DeepSort(max_age=20)  # Set max_age for track lifetime

    # Sort face data by frame index
    face_data_sorted = sorted(face_data, key=lambda x: x['frame_index'])

    # Open the video file using PyAV
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "SLICE"
    fps = stream.base_rate  # Get the frame rate

    # Get video properties
    width =640
    height =480
    count=0
    # Prepare video writer
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (width, height))

    # Initialize variables
    current_face_data_index = 0  # Index to track position in face_data_sorted
    name_to_track_id = {}  # Map names to track IDs
    track_id_to_name = {}  # Map track IDs to names
    frame_index = 0  # Track the current frame index

    # Process each frame
    for frame in container.decode(stream):
        # Convert PyAV frame to OpenCV format
        count+=1

        frame = frame.reformat(640, 480, format="bgr24")
        frame = frame.to_ndarray()

        if frame is None or frame.shape[0]==0 or frame.shape[1]==0 or frame.shape[2]==0 or frame.size==0:
            print(f"Could not read frame: {frame_index}")
            continue

        # Get YOLO detections for the current frame
        if yolo_detections is not None:
            if yolo_detections.get(frame_index, {}) is not None:
                yolo_bboxes = yolo_detections.get(frame_index, {}).get('bboxes', [])
                yolo_scores = yolo_detections.get(frame_index, {}).get('scores', [])
            else:
                yolo_bboxes = []
                yolo_scores = []
        else:
            yolo_bboxes = []
            yolo_scores = []

        # Initialize detections list
        detections = []

        # Add RetinaFace detections (if available for this frame)
        if current_face_data_index < len(face_data_sorted) and frame_index == face_data_sorted[current_face_data_index]['frame_index']:
            face_data_for_frame = face_data_sorted[current_face_data_index]
            bboxes = face_data_for_frame['bboxes']
            embeddings = face_data_for_frame['embeddings']
            scores = face_data_for_frame['confidence']
            names = face_data_for_frame.get('names', [])  # Get names if available

            for bbox, embedding, score, name in zip(bboxes, embeddings, scores, names):
                x1, y1, x2, y2 = bbox
                detections.append(([x1, y1, x2, y2], score, embedding, name))  # Include name in detections

            # Move to the next face data entry
            current_face_data_index += 1

        # Add YOLO detections (if available for this frame)
        for bbox, score in zip(yolo_bboxes, yolo_scores):
            x1, y1, x2, y2 = bbox
            detections.append(([x1, y1, x2, y2], score, None, None))  # No embeddings or names for YOLO detections

        # Update tracker with detections
      
        tracks = tracker.update_tracks(detections, frame=frame)
        


        # Process tracks to combine those with the same name and remove those with no name
        updated_tracks = []
        for track in tracks:
            # Get the name for this track (if available)
            track_id = track.track_id
            name = track_id_to_name.get(track_id, None)

            # If the track has no name, try to associate it with a named track using IoU
            if name is None:
                for detection in detections:
                    if detection[3] is not None:  # Named detection (RetinaFace)
                        detection_bbox = detection[0]
                        track_bbox = track.to_tlbr()

                        # Calculate IoU between detection and track bounding boxes
                        iou = calculate_iou(detection_bbox, track_bbox)
                        
                        if iou > 0.006:  # Threshold for association
                            # Associate the track with the name
                            name = detection[3]
                            track_id_to_name[track_id] = name
                            name_to_track_id[name] = track_id
                            break

            # If the track still has no name, remove it
            if name is None:
                continue

            # If the track has a name, check if it matches an existing track
            if name in name_to_track_id:
                # Combine with the existing track
                existing_track_id = name_to_track_id[name]
                if existing_track_id != track_id:
                    # Merge the two tracks (keep the existing track and update its bounding box)
                    existing_track = next((t for t in tracks if t.track_id == existing_track_id), None)
                    if existing_track:
                        existing_track.to_tlbr = track.to_tlbr  # Update bounding box
                        continue  # Skip adding this track
            else:
                # Add this track to the name-to-track-ID mapping
                name_to_track_id[name] = track_id
                track_id_to_name[track_id] = name

            updated_tracks.append(track)

        # Draw bounding boxes and track IDs on the frame
        for track in updated_tracks:
            bbox = track.to_tlbr().astype(int)  # Get bounding box
            track_id = track.track_id  # Get track ID
            name = track_id_to_name.get(track_id, "Unknown")  # Get name

            # Draw bounding box
            color = (0, 255, 0) if track.time_since_update == 0 else (0, 0, 255)  # Green for active, red for inactive
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Label track ID and name
            cv2.putText(frame, f" ({name})", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame to the output video
        video_writer.write(frame)

        # Increment frame index
        frame_index += 1

    # Release video writer
    video_writer.release()
    print(f"Output video saved to {output_video_path}")
