import cv2
import face_recognition
import pickle
import os
import csv

def encode_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings

def save_encodings(image_path, person_name):
    encodings = encode_faces(image_path)
    if os.path.exists('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl'):
        with open('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        data = []
    data.append({'name': person_name, 'encodings': encodings})
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl', 'rb') as f:
        data = pickle.load(f)
        known_names = [item['name'] for item in data]
        with open('/Users/prince_13/Documents/projects/try/bakwass/final_/people_data/in_out_present_status.csv','w') as f:
            writer  = csv.writer(f)
            a = [[False]]*len(known_names)
            writer.writerow(a)
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/people_data/data.csv','w') as f :
            pass

    print(f"Encodings for {person_name} saved to common_encodings.pkl")

def delete_person_encoding(person_name):
    if not os.path.exists('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl'):
        print("No encodings file found.")
        return

    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data)
    data = [entry for entry in data if entry['name'] != person_name]
    print(f'New data: {data}')
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl', 'rb') as f:
        data = pickle.load(f)
        known_names = [item['name'] for item in data]
        with open('/Users/prince_13/Documents/projects/try/bakwass/final_/people_data/in_out_present_status.csv','w') as f:
            writer  = csv.writer(f)
            a = [[False]]*len(known_names)
            writer.writerow(a)
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/people_data/data.csv','w') as f :
            pass
        
            

    print(f"Encodings for {person_name} deleted from common_encodings.pkl")

def main():
    action = input("Enter 'save' to save encodings or 'delete' to delete encodings: ").strip().lower()
    person_name = input("Enter the name of the person: ").strip()

    if action == 'save':
        image_path = input("Enter the path to the image: ").strip()
        save_encodings(image_path, person_name)
    elif action == 'delete':
        delete_person_encoding(person_name)
    else:
        print("Invalid action.")

if __name__ == "__main__":
    main()