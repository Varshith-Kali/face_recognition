import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime 



source_path = "Images"
images_list = []
images_wextension = []
list = os.listdir(source_path)

for x in list:
    current_image = cv2.imread(f"{source_path}/{x}")
    images_list.append(current_image)
    images_wextension.append(x[:-4])  # made changes : Used slicing to remove extension


def encode_image(images_list):
    encoded_list = []

    for op in images_list:
        op = cv2.cvtColor(op, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(op)[0]
        encoded_list.append(encode)
    return encoded_list

known_faces = encode_image(images_list)
print("Encoding Completed !!!")


# Caputuring webcam image

cam = cv2.VideoCapture(0)

while True:

    success, img = cam.read()
    reimage = cv2.resize(img, (0,0), None, 0.25, 0.25)
    reimage = cv2.cvtColor(reimage, cv2.COLOR_BGR2RGB)

    cam_frame_location = face_recognition.face_locations(reimage)
    cam_frame_encode = face_recognition.face_encodings(reimage) # made changes : Didn't give the arguments same a the guy.

    
    for face_loc, encode_face in zip (cam_frame_location, cam_frame_encode):

        compare = face_recognition.compare_faces(known_faces,encode_face)
        face_dis = face_recognition.face_distance(known_faces, encode_face)
        index = np.argmin(face_dis)

        if compare[index]:
            name = images_wextension[index].upper()

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y1-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x+6, y+6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            # attendance(name)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)



