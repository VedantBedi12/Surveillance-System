from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
import numpy as np
import faiss
import cv2
import os
import time

class SURV3:
    def __init__(self, yolo_model_path, facenet_model='vggface2', faiss_dim=512, index_path='faiss_index.bin'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.yolo_model = YOLO(yolo_model_path)
        self.facenet = InceptionResnetV1(pretrained=facenet_model).eval().to(self.device)
        self.index = faiss.IndexFlatIP(faiss_dim)
        self.index_path = index_path

        self.load_index()

    def normalize_embeddings(self, embeddings):
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"Index loaded from {self.index_path}. Number of vectors: {self.index.ntotal}")
        else:
            print(f"No index file found at {self.index_path}. Starting with an empty index.")

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        print(f"Index saved to {self.index_path}.")

    def clear_index(self):
        self.index.reset()
        print("All vectors cleared from the index.")

    def remove_vector(self, index_to_remove):
        try:
            self.index.remove_ids(np.array([index_to_remove], dtype=np.int64))
            print(f"Vector at index {index_to_remove} removed from the index.")
        except Exception as e:
            print(f"Error removing vector at index {index_to_remove}: {e}")

    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_face_embeddings(self, image_path):
        try:
            img = cv2.imread(image_path)
            results = self.yolo_model(img)
            detections = results[0].boxes

            if detections is not None and len(detections) > 0:
                embeddings = []
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    face = img[y1:y2, x1:x2]
                    face = self.convert_to_grayscale(face)
                    face = cv2.resize(face, (160, 160))
                    face = Image.fromarray(face).convert('RGB')
                    face = np.transpose(np.array(face), (2, 0, 1)) / 255.0
                    face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(self.device)
                    face_embedding = self.facenet(face_tensor)
                    embeddings.append(face_embedding.detach().cpu().numpy())
                return self.normalize_embeddings(np.vstack(embeddings))
            else:
                print("No faces detected in the image.")
                return None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def add_embeddings_to_faiss(self, image_paths):
        for image_path in image_paths:
            embeddings = self.get_face_embeddings(image_path)
            if embeddings is not None:
                self.index.add(embeddings)
        print(f"Number of vectors in the index: {self.index.ntotal}")

    def find_closest_match(self, image_path, threshold=0.9):
        query_embedding = self.get_face_embeddings(image_path)
        if query_embedding is not None:
            distances, indices = self.index.search(query_embedding, k=1)
            for dist, idx in zip(distances[0], indices[0]):
                similarity = dist
                if similarity > threshold:
                    print(f"Match found! Similarity: {similarity}, Index: {idx}")
                    return idx
                else:
                    print("No match found.")
                    return None
        else:
            print("No embeddings generated for the query image.")
            return None

    def webcam_inference(self, threshold=0.9, target_fps=0.5):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        frame_interval = 1.0 / target_fps
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            results = self.yolo_model(frame)
            detections = results[0].boxes

            if detections is not None and len(detections) > 0:
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()

                    face = frame[y1:y2, x1:x2]
                    face = self.convert_to_grayscale(face)
                    face = cv2.resize(face, (160, 160))
                    face = Image.fromarray(face).convert('RGB')
                    face = np.transpose(np.array(face), (2, 0, 1)) / 255.0
                    face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(self.device)
                    face_embedding = self.facenet(face_tensor).detach().cpu().numpy()
                    face_embedding = self.normalize_embeddings(face_embedding)
                    distances, indices = self.index.search(face_embedding, k=1)

                    for dist, idx in zip(distances[0], indices[0]):
                        similarity = dist
                        label = f"Match (Similarity: {similarity:.2f}, Index: {idx})" if similarity > threshold else "No Match"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Face Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Webcam Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def pipeline(self, mode, image_paths=None, query_image=None, video_path=None, threshold=0.9, index_to_remove=None, output_path='output_video.mp4'):
        if mode == 'feed':
            if image_paths is not None:
                self.add_embeddings_to_faiss(image_paths)
                self.save_index()
            else:
                print("No images provided for feeding.")
        elif mode == 'infer':
            if query_image is not None:
                return self.find_closest_match(query_image, threshold)
            else:
                print("No query image provided for inference.")
        elif mode == 'webcam_infer':
            self.webcam_inference(threshold)
        elif mode == 'clear':
            self.clear_index()
            self.save_index()
        elif mode == 'remove':
            if index_to_remove is not None:
                self.remove_vector(index_to_remove)
                self.save_index()
            else:
                print("No index specified for removal.")
        else:
            print("Invalid mode. Choose 'feed', 'infer', 'webcam_infer', 'clear', or 'remove'.")
