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
from imutils.video import VideoStream
import imutils
import time



# # load model
# model_json_file="model.json"
# model_weights_file="model_weightsTest.h5"

# # load model
# with open(model_json_file, "r") as json_file:
#     loaded_model_json = json_file.read()
#     model = model_from_json(loaded_model_json)
# model.load_weights(model_weights_file)

# load models
with open("models/model.json", "r") as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)
model.load_weights("models/model_weights.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


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

    #dst = cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]] )
        # src: input image
        # M: Transformation matrix
        # dsize: size of the output image
        # flags: interpolation method to be used
    #Sert a faire une transformation dans la perspective
    res=cv2.warpPerspective(mask,mat,(img.shape[1],img.shape[0]),cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,borderValue=(-1, -1, -1))

    return res

def detect_and_predict_mask(frame, faceNet, maskNet):
    	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

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
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def pro(img,mask,draw_rect1=True,draw_rect2=True,draw_lines=True,draw_mask=True):
    copy = img.copy()
    colImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    faces=detector(img, 0)
    print("\nfacesMask\n")
    print(faces)
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

##### PARTIE EMOTION Recognition

#photo
mask=cv2.imread("testHAPPY.jpg")

# load our serialized face detector model from disk
prototxtPath = r"models\deploy.prototxt"
weightsPath = r"models\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("models/mask_detector.model")

# the video
cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
predicted_emotions=[]
toBeExcluded=[]
while(True):
    ret, frame = cap.read()
    if not ret:
        break
    else:
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        if preds != []:
            toBeExcluded=[]
            index=0
            for pred in preds:
                if pred[0]*100>95:
                    predicted_emotion="mask"
                    print(locs)
                    toBeExcluded.append(((locs[index][0]+locs[index][2])//2,(locs[index][1]+locs[index][3])//2))
                    index=index+1
        print(toBeExcluded)
        emotions = ('angry','disgust','fear','happy','neutral','sad','surprise')
        color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(color_img, 1.2, 5)

        font = cv2.FONT_HERSHEY_SIMPLEX
        predicted_emotions=[]
        print("\nfacesdetected\n")
        for (x, y, w, h) in faces_detected:
            for j in range(0,len(toBeExcluded)):
                print(toBeExcluded)
                print(x,y,w,h)
                if toBeExcluded[j][0] in range(x,w+x) and toBeExcluded[j][1] in range(y,h+y):
                    predicted_emotions.append("mask")
                    break
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi = color_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi = cv2.resize(roi, (48, 48))

            predictions = model.predict(roi[np.newaxis, :, :, np.newaxis])

            # find max indexed array
            max_index = np.argmax(predictions[0])


            predicted_emotions.append(emotions[max_index])

            # cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(predicted_emotions)
        res=pro(frame,predicted_emotions,draw_rect1=False,draw_rect2=False,draw_lines=False,draw_mask=True)
        resized_img = cv2.resize(frame, (1000, 700))
        cv2.imshow('head',res)
        # Write the frame into the file 'output.avi'
        out.write(res)

    key = cv2.waitKey(10)
    # Escキーが押されたら
    if key == 27:
        cv2.destroyAllWindows()
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()