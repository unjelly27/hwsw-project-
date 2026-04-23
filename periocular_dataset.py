import os
import cv2
from tqdm import tqdm

input_dir = "../facedata/lfw-deepfunneled/lfw-deepfunneled"
output_dir = "../periocular_dataset"

os.makedirs(output_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for person in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person)
    if not os.path.isdir(person_path):
        continue

    out_person_path = os.path.join(output_dir, person)
    os.makedirs(out_person_path, exist_ok=True)

    images = os.listdir(person_path)
    for img_name in tqdm(images, desc=person):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        (x, y, w, h) = faces[0]

        # Crop upper half of face
        eye_region = img[y:y + h//2, x:x + w]

        # Resize
        eye_region = cv2.resize(eye_region, (224, 224))

        save_path = os.path.join(out_person_path, img_name)
        cv2.imwrite(save_path, eye_region)

print("Periocular dataset created successfully!")
