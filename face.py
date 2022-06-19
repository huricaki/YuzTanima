import face_recognition
import cv2
import os
import glob
import numpy as np
import pandas as pd
from simple_facerec import SimpleFacerec
#import RPi.GPIO as GPIO

#def lock(pin):
    # GPIO.output(pin, GPIO.LOW)
    print('Kilitlendi!')


# Kilit Aç
#def unlock(pin):
    # GPIO.output(pin, GPIO.HIGH)
#    print('Kilit Açıldı!')
#GPIO26 pinini seçtiğimizi belirtiyoruz
#role_pini = [26]
# Gelebilecek gereksiz uyarıları devre dışı bırakıyoruz
#GPIO.setwarnings(False)
# GPIO numaralarına göre seçim yaptık
#GPIO.setmode(GPIO.BCM)
# Seçtiğimiz röle pinini çıkış olarak ayarlıyoruz.
#GPIO.setup(role_pini, GPIO.OUT)
#lock(role_pini)
sfr=SimpleFacerec()
sfr.load_encoding_images("images/")
cap=cv2.VideoCapture(0)
while True:
	ret,frame=cap.read()
	face_locations,face_names=sfr.detect_known_faces(frame)
	for face_loc, name in zip(face_locations,face_names):
		y1, x1, y2, x2=face_loc[0], face_loc[1],face_loc[2],face_loc[3]
		#if name=="Unknown":
			lock(role_pini)
		#else:
			#def dosya():
		#	reader=pd.read_excel('./deneme.xlsx')
			
				#writer = pd.ExcelWriter('./deneme.xlsx', engine='xlsxwriter')
		#	kolon=["Gelenler"]
			#df=pd.DataFrame(columns=kolon)
			#isim=name
			#print(isim)
			#for i in face_locations :
				#for index in format(len("./images/")):
		#	df=pd.DataFrame([name], columns=kolon)
		#	df=pd.concat([df])

				#df=df.append({'Gelenler':name}, ignore_index=True)
			#df.to_excel(writer)
			#unlock(role_pini) 
		#	print(df)
			
		cv2.putText(frame, name,(x1,y1-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)
		cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),4)
		
	cv2.imshow("Frame",frame)
	key=cv2.waitKey(1)
	if key==27:

		#writer.save()
		break

cap.release()
cv2.destroyAllWindows()
class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
       
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
