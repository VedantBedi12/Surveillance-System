import av
import torch
import numpy as np
import faiss
import cv2
from face_detection import RetinaFace
from facenet_pytorch import InceptionResnetV1
import os

def detect_and_recognize_faces(output_folder, changes,collection):
    # Initialize models
    face_detector = RetinaFace(gpu_id=0 if torch.cuda.is_available() else -1)
    face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()
    face_recognizer.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Prepare data storage
    face_data = []

    # Get all frame files in the output folder


    # Initialize frame counter
    frame_counter = 0
    embeddings_list = []
    names_list = []



    for doc in collection.find({}):
        embeddings_list.append(np.array(doc["embedding"], dtype=np.float32))
        names_list.append(doc["name"])

    # Convert embeddings to a NumPy array
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    embeddings_array = np.squeeze(embeddings_array, axis=1)

    # Build a FAISS index
    dimension = embeddings_array.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    faiss_index.add(embeddings_array)  # Add embeddings to the index

    # Open the video file using PyAV
    container = av.open(output_folder)  # Replace with your video path
    stream = container.streams.video[0]
    stream.thread_type = "SLICE"  # Enable slice threading for faster decoding
    fps = stream.base_rate  # Get the frame rate

    # Process all frames
    for frame in container.decode(stream):
        # Get the frame index

        # Check if face detection should run:
        # 1. Every 30 frames, or
        # 2. When there is a change in the number of persons
        should_detect_faces = (frame_counter % 30 == 0) or any(change['frame'] == frame_counter for change in changes)

        if not should_detect_faces:
            frame_counter += 1
            continue  # Skip this frame

        # Convert PyAV frame to OpenCV format
        frame = frame.reformat(640, 480, format="bgr24")  # Resize if needed
        frame = frame.to_ndarray()

        if frame is None:
            print(f"Frame {frame_counter} not found or could not be loaded.")
            frame_counter += 1
            continue

        # Detect faces using RetinaFace
        faces = face_detector(frame)
        while faces is None:
            frame_counter += 1
            frame = next(container.decode(stream))  # Move to the next frame
            frame = frame.reformat(640, 480, format="bgr24")
            frame = frame.to_ndarray()
            faces = face_detector(frame)

        # Extract face embeddings using FaceNet
        embeddings = []
        bboxes = []
        scores = []
        for face in faces:
            box, landmark, score = face
            if score > 0.5:
                x1, y1, x2, y2 = map(int, box.tolist())
                face_img = frame[y1:y2, x1:x2]

                # Preprocess face image for FaceNet
                face_img = cv2.resize(face_img, (160, 160))
                face_img = face_img / 255.0
                face_img = torch.tensor(face_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                face_img = face_img.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                # Get embedding
                with torch.no_grad():
                    embedding = face_recognizer(face_img)
                embeddings.append(embedding.cpu().numpy())
                bboxes.append((x1, y1, x2, y2))
                scores.append(score)

        # Recognize faces using FAISS
        names = []
        for embedding in embeddings:
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)  # Reshape for FAISS
            distances, indices = faiss_index.search(embedding, k=1)  # Find the closest match
            if distances[0][0] < 1.0:  # Threshold for similarity (adjust as needed)
                names.append(names_list[indices[0][0]])
            else:
                names.append("Unknown")  # No match found

        # Store frame data
        face_data.append({
            'frame_index': frame_counter,
            'confidence': scores,
            'bboxes': bboxes,
            'embeddings': embeddings,
            'names': names
        })

        # Increment frame counter
        frame_counter += 1

    return face_data
