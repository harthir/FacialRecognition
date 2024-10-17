import cv2
import mediapipe as mp
import face_recognition
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load reference image and get face encodings
reference_image_path = "IMG_9289.jpg"
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encodings = face_recognition.face_encodings(reference_image)

# Check if at least one encoding is found
if len(reference_encodings) == 0:
    raise ValueError("No faces found in the reference image.")

reference_encoding = reference_encodings[0]  # Use the first face found

# For webcam input
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

                # Extract the bounding box for the face
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h), 
                                       int(bboxC.width * w), int(bboxC.height * h))

                # Get the face image
                face_image = image[y:y + height, x:x + width]
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                # Get face encodings for the detected face
                face_encodings = face_recognition.face_encodings(face_image_rgb)

                if face_encodings:  # If a face is found
                    face_encoding = face_encodings[0]

                    # Compare with the reference encoding
                    results = face_recognition.compare_faces([reference_encoding], face_encoding)

                    if results[0]:  # If the face matches
                        print("MATCH!")
                    else:
                        print("No Match")
                    

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()