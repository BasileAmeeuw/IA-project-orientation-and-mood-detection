# import the necessary packages
import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model, model_from_json
import matplotlib.pyplot as plt
from skimage import img_as_float,img_as_ubyte
import dlib
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
from argparse import ArgumentParser
import ctypes
import random




user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=0,
					help="Vous pouvez insérez après --video le path de la vidéo que vous voudriez rentrer dans l'application")
parser.add_argument("--image", type=str, default=None,
					help="Vous pouvez insérez après --image le path de l'image que vous voudriez rentrer dans l'application")
parser.add_argument("--savePath", type=str, default="photo",
					help="Vous pouvez insérez après --savePath le path de pour enregistrer l'image")
args = parser.parse_args()

with open("models/model.json", "r") as json_file:
	model_json = json_file.read()
	model = model_from_json(model_json)
model.load_weights("models/model_weights.h5")



# face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

detector = dlib.get_frontal_face_detector()
predector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def add_photo(img,pt2,mask):
	mask=img_as_float(mask)
	img=img_as_float(img)
	pt1=np.float32([[0,0],
				  [mask.shape[1],0],
				  [0,mask.shape[0]],
				  [mask.shape[1],mask.shape[0]]
				  ])
	mat = cv2.getPerspectiveTransform(pt1,pt2)

	res=cv2.warpPerspective(mask,mat,(img.shape[1],img.shape[0]),cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,borderValue=(-1, -1, -1))

	return res

def pro(img,mask,draw_rect1=True,draw_rect2=True,draw_lines=True,draw_mask=True):
	copy = img.copy()
	colImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	faces=detector(img, 0)
	i=-1
	for face in faces:
		i=int((i+1)/480)
		x1 = face.left()

		y1 =face.top()

		x2= face.right()


		y2= face.bottom()


		landmarks = predector(colImg,face)

		size = copy.shape
		#2D image points. If you change the image, you need to change vector
		image_points = np.array([
								(landmarks.part(33).x,landmarks.part(33).y),     # Nose tip
								(landmarks.part(8).x,landmarks.part(8).y),       # Chin
								(landmarks.part(36).x,landmarks.part(36).y),     # Left eye left corner
								(landmarks.part(45).x,landmarks.part(45).y),     # Right eye right corne
								(landmarks.part(48).x,landmarks.part(48).y),     # Left Mouth corner
								(landmarks.part(54).x,landmarks.part(54).y)      # Right mouth corner
							], dtype="double")

		# 3D model points.
		model_points = np.array([
								(0.0, 0.0, 0.0),             # Nose tip
								(0.0, -330.0, -65.0),        # Chin
								(-225.0, 170.0, -135.0),     # Left eye left corner
								(225.0, 170.0, -135.0),      # Right eye right corne
								(-150.0, -150.0, -125.0),    # Left Mouth corner
								(150.0, -150.0, -125.0)      # Right mouth corner

							])
		# Camera internals
		focal_length = size[1]
		center = (size[1]/2, size[0]/2)
		camera_matrix = np.array(
							[[focal_length, 0, center[0]],
							[0, focal_length, center[1]],
							[0, 0, 1]], dtype = "double"
							)

		dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
		(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

		(b1, jacobian) = cv2.projectPoints(np.array([(350.0, 270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		(b2, jacobian) = cv2.projectPoints(np.array([(-350.0, -270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		(b3, jacobian) = cv2.projectPoints(np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		(b4, jacobian) = cv2.projectPoints(np.array([(350.0, -270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

		(b11, jacobian) = cv2.projectPoints(np.array([(500.0, 450.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		(b12, jacobian) = cv2.projectPoints(np.array([(-400.0, -450.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		(b13, jacobian) = cv2.projectPoints(np.array([(-400.0, 450, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		(b14, jacobian) = cv2.projectPoints(np.array([(500.0, -450.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

		b1 = ( int(b1[0][0][0]), int(b1[0][0][1]))
		b2 = ( int(b2[0][0][0]), int(b2[0][0][1]))
		b3 = ( int(b3[0][0][0]), int(b3[0][0][1]))
		b4 = ( int(b4[0][0][0]), int(b4[0][0][1]))

		b11 = ( int(b11[0][0][0]), int(b11[0][0][1]))
		b12 = ( int(b12[0][0][0]), int(b12[0][0][1]))
		b13 = ( int(b13[0][0][0]), int(b13[0][0][1]))
		b14 = ( int(b14[0][0][0]), int(b14[0][0][1]))

		if draw_rect1 ==True:
			cv2.line(copy,b1,b3,(255,255,0),10)
			cv2.line(copy,b3,b2,(255,255,0),10)
			cv2.line(copy,b2,b4,(255,255,0),10)
			cv2.line(copy,b4,b1,(255,255,0),10)

		if draw_rect2 ==True:
			cv2.line(copy,b11,b13,(255,255,0),10)
			cv2.line(copy,b13,b12,(255,255,0),10)
			cv2.line(copy,b12,b14,(255,255,0),10)
			cv2.line(copy,b14,b11,(255,255,0),10)

		if draw_lines == True:
			cv2.line(copy,b11,b1,(0,255,0),10)
			cv2.line(copy,b13,b3,(0,255,0),10)
			cv2.line(copy,b12,b2,(0,255,0),10)
			cv2.line(copy,b14,b4,(0,255,0),10)

		if draw_mask ==True:
			pt=np.float32([b11,b13,b14,b12])
			try:
				maskImage=cv2.imread("imageSmiley/"+mask[i]+".jpg")
			except:
				maskImage=cv2.imread("imageSmiley/"+"neutral"+".jpg")
			ty=add_photo(copy,pt,maskImage)
			tb= img_as_ubyte(ty)
			for i in range(0,ty.shape[0]):
				for j in range(0,ty.shape[1]):
					k=ty[i,j]
					if k[0] != -1 and k[1] != -1 and k[2] != -1:
						copy[i,j] = tb[i,j]

	return copy

def proOne(frame,locs,mask,draw_rect1=True,draw_rect2=True,draw_lines=True,draw_mask=True):
	copy = frame
	colImg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	(x,y,w,h)=locs

	test=frame[y:h,x:w]
	face=detector(frame[y:h,x:w],0)

	succeed=True
	try:
		if face[0]:
			for face in face:
					
				face=dlib.rectangle(x+face.left(),y+face.top(),face.right()+x,face.bottom()+y)
				# cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), thickness=7)
				print(face)
				
				landmarks = predector(colImg,face)
				size = copy.shape
				#2D image points. If you change the image, you need to change vector
				image_points = np.array([
										(landmarks.part(33).x,landmarks.part(33).y),     # Nose tip
										(landmarks.part(8).x,landmarks.part(8).y),       # Chin
										(landmarks.part(36).x,landmarks.part(36).y),     # Left eye left corner
										(landmarks.part(45).x,landmarks.part(45).y),     # Right eye right corne
										(landmarks.part(48).x,landmarks.part(48).y),     # Left Mouth corner
										(landmarks.part(54).x,landmarks.part(54).y)      # Right mouth corner
									], dtype="double")

				# 3D model points.
				model_points = np.array([
										(0.0, 0.0, 0.0),             # Nose tip
										(0.0, -330.0, -65.0),        # Chin
										(-225.0, 170.0, -135.0),     # Left eye left corner
										(225.0, 170.0, -135.0),      # Right eye right corne
										(-150.0, -150.0, -125.0),    # Left Mouth corner
										(150.0, -150.0, -125.0)      # Right mouth corner

									])
				# Camera internals
				focal_length = size[1]
				center = (size[1]/2, size[0]/2)
				camera_matrix = np.array(
									[[focal_length, 0, center[0]],
									[0, focal_length, center[1]],
									[0, 0, 1]], dtype = "double"
									)

				dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
				(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

				(b1, _) = cv2.projectPoints(np.array([(350.0, 270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
				(b2, _) = cv2.projectPoints(np.array([(-350.0, -270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
				(b3, _) = cv2.projectPoints(np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
				(b4, _) = cv2.projectPoints(np.array([(350.0, -270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

				(b11, _) = cv2.projectPoints(np.array([(500.0, 450.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
				(b12, _) = cv2.projectPoints(np.array([(-400.0, -450.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
				(b13, _) = cv2.projectPoints(np.array([(-400.0, 450, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
				(b14, _) = cv2.projectPoints(np.array([(500.0, -450.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

				b1 = ( int(b1[0][0][0]), int(b1[0][0][1]))
				b2 = ( int(b2[0][0][0]), int(b2[0][0][1]))
				b3 = ( int(b3[0][0][0]), int(b3[0][0][1]))
				b4 = ( int(b4[0][0][0]), int(b4[0][0][1]))

				b11 = ( int(b11[0][0][0]), int(b11[0][0][1]))
				b12 = ( int(b12[0][0][0]), int(b12[0][0][1]))
				b13 = ( int(b13[0][0][0]), int(b13[0][0][1]))
				b14 = ( int(b14[0][0][0]), int(b14[0][0][1]))

				if draw_rect1 ==True:
					cv2.line(copy,b1,b3,(255,255,0),10)
					cv2.line(copy,b3,b2,(255,255,0),10)
					cv2.line(copy,b2,b4,(255,255,0),10)
					cv2.line(copy,b4,b1,(255,255,0),10)

				if draw_rect2 ==True:
					cv2.line(copy,b11,b13,(255,255,0),10)
					cv2.line(copy,b13,b12,(255,255,0),10)
					cv2.line(copy,b12,b14,(255,255,0),10)
					cv2.line(copy,b14,b11,(255,255,0),10)

				if draw_lines == True:
					cv2.line(copy,b11,b1,(0,255,0),10)
					cv2.line(copy,b13,b3,(0,255,0),10)
					cv2.line(copy,b12,b2,(0,255,0),10)
					cv2.line(copy,b14,b4,(0,255,0),10)

				if draw_mask ==True:
					pt=np.float32([b11,b13,b14,b12])
					# try:
					maskImage=cv2.imread("imageSmiley/"+mask+".jpg")
					# except:
					# 	maskImage=cv2.imread("imageSmiley/"+"neutral"+".jpg")
					ty=add_photo(copy,pt,maskImage)
					tb= img_as_ubyte(ty)
					for i in range(0,ty.shape[0]):
						for j in range(0,ty.shape[1]):
							k=ty[i,j]
							if k[0] != -1 and k[1] != -1 and k[2] != -1:
								copy[i,j] = tb[i,j]
							

	except:
		succeed=False
		
		
		
	return (succeed,copy)

##### PARTIE EMOTION Recognition

# load our serialized face detector model from disk
prototxtPath = r"models\deploy.prototxt"
weightsPath = r"models\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("models/mask_detector.model")

# the video

cap = cv2.VideoCapture(args.video)
if args.image is not None:
	cap = cv2.VideoCapture(args.image)
	ret,frame=cap.read()
if (cap.isOpened() == False):
	print("Unable to read")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
predicted_emotions=[]
toBeExcluded=[]
counter=0
videoNumber=str(random.randint(0,1000))
while(True):
	if args.image is None:
		ret, frame = cap.read()
	else:
		if counter==15:
			cv2.imwrite('image_saved/'+args.savePath+".jpg", copy)
			break
		counter+=1
	if not ret:
		break
	else:
		copy=frame
		
		emotions = ('angry','disgust','fear','happy','neutral','sad','surprise')
		# faces_detected = face_haar_cascade.detectMultiScale(frame, 1.2, 5)
		
		# print(faces_detected)
		font = cv2.FONT_HERSHEY_SIMPLEX
		predicted_emotions=[]
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
			(104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		faceNet.setInput(blob)
		detections = faceNet.forward()
		

		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]
			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				locs=box.astype("int")
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
				(startXOr, startYOr) = (max(0, startX-40), max(0, startY-40))	
				(endXOr, endYOr) = (min(w - 1, endX+40), min(h - 1, endY+40))
				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				locsForOr=(startXOr,startYOr,endXOr,endYOr)
				# cv2.imshow("yes",cv2.resize(face,(700,700)))
				# cv2.imshow("ydses",cv2.resize(faceForOr,(700,700)))
				if face.any():
					facesMask=[]
					faceMask = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
					faceMask = cv2.resize(faceMask, (224, 224))
					faceMask = img_to_array(faceMask)
					faceMask = preprocess_input(faceMask)
					facesMask.append(faceMask)
					facesMask = np.array(facesMask, dtype="float32")
					prediction=maskNet.predict(facesMask, batch_size=32)[0]
					# print(prediction)
					if prediction[0]>0.7:
						# print(prediction)
						predicted_emotion="mask"
						# cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), thickness=7)

					else:
						# cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), thickness=7)
						faceEmotion = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
						faceEmotion = cv2.resize(faceEmotion, (48, 48))
						predictions = model.predict(faceEmotion[np.newaxis, :, :, np.newaxis])
						max_index = np.argmax(predictions[0])
						# print(predictions)
						predicted_emotion=emotions[max_index]

					(succeed,copy)=proOne(copy,locsForOr,predicted_emotion,draw_rect1=False,draw_rect2=False,draw_lines=False,draw_mask=True)
					if not succeed:
						cv2.putText(frame, "pas d'orientation mais " + predicted_emotion + " devine", (startX, startY - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
						cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

		resized_img = cv2.resize(copy, (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)))
		if args.image is not None:
			cv2.imshow('image'+str(random.randint(0,1000)),resized_img)
		else:
			cv2.imshow('votre video'+videoNumber,resized_img)
		# Write the frame into the file 'output.avi'
		out.write(copy)

	key = cv2.waitKey(10)
	# Escキーが押されたら
	if key == 27:
		cv2.destroyAllWindows()
		break

# When everything done, release the video capture and video write objects
if args.image is not None:
	time.sleep(10)
cap.release()
out.release()