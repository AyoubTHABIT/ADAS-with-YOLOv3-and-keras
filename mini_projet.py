
# Here we import OpenCV 
import cv2 as cv
# We import Numpy; The fundamental package for scientific computing with Python.
import numpy as np
#The keras's model is already trained, we just need to load it
from keras.models import load_model
# create a deque
from collections import deque

confThreshold = 0.25
nmsThreshold = 0.40  #non-maximum suppression threshold
inpWidth = 416  #Width of network4s input image
inpHeight = 416  #Height of network4s input image


#Name the files 
classesFile = "coco.names"
modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#We load the model
net = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)

#we set the backend; it refers to the implementation
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

#target refers to the processor
#here we use CPU as a processing unit
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


#we load the distance model
distance_model_path = "model_3.keras"
distance_model = load_model(distance_model_path, compile=True)


# Get the names of the output layers
def getOutputsNames(net):
    
    #get the names of all the output layers
    layersNames = net.getLayerNames()
# Get the names of all the layers in the network
# Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#yhe input to the model is a vector calculated here
def load_dist_input( predict_box, classID, img_width, img_height):
    
        predict_class = classes[classID]
        cc = ['person','bus', 'truck', 'car', 'bicycle', 'motorbike', 'cat', 'dog', 'horse', 'sheep', 'cow']
        if str(predict_class) in cc:
            
            top, left, bottom, right = predict_box
            width = float(right - left) / img_width
            height = float(bottom - top) / img_height
            diagonal = np.sqrt(np.square(width) + np.square(height))
            class_h, class_w, class_d = set_class_size[predict_class]
            dist_input = [1 / width, 1 / height, 1 / diagonal, class_h, class_w, class_d]
            
        else:
           
           dist_input = [0, 0, 0, 0, 0, 0]
           
        return np.array(dist_input)
    
    
    
def getDirection(image, pointList):
    
    dx = 0
    dy = 0

    for x in range(len(pointList)-1):
        
        cv.line(image, pointList[x], pointList[x+1], [0, 255, 0], 10)
        
        dx += pointList[x+1][0] - pointList[x][0]  
        dy += pointList[x+1][1] - pointList[x][1] 

    x = ""
    y = ""

    if(dx < 0):
        x = "right"

    if(dx > 0):
        x = "left"

    if(dy < 0):
        y = "down"

    if(dy > 0):
        y = "up"

    return (x,y)
        



Classes = ['person', 'bus', 'truck', 'car', 'bicycle', 'motorbike', 'cat', 'dog', 'horse', 'sheep', 'cow']
class_shape = [[115, 45, 10], [300, 250, 1200], [400, 350, 1500], [160, 180, 400], [110, 50, 180],
                       [110, 50, 180], [40, 20, 50], [50, 30, 60], [180, 60, 200], [130, 60, 150], [170, 70, 200]]
set_class_size = dict(zip(Classes, class_shape))
#print("Load Class size!")
        

def postprocess(frame, outs):
    
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
    
        classIDs = []
        confidences = []
        boxes = []
    
    #scan through all the bounding boxes output and keep only the ones with high confidence scores
        for out in outs:
        
            for detection in out:
                
                print(len(detection))
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
            
                 # the bounding boxes kept are determined by the confThreshold = 0.25
                if confidence > confThreshold:
                    
                    centerX = int(detection[0]* frameWidth)
                    centerY = int(detection[1]* frameHeight)
                
                    width = int(detection[2]* frameWidth)
                    height = int(detection[2]* frameWidth)
                
                    left = int(centerX - width/2)
                    top = int(centerY - height/2)
                
                    classIDs.append(classID)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                
        #perform non maximum suppression to eliminate redundant overlapping boxes with lower confidence
        indices = cv.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold ) 
        
        for i in indices:
          
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
        
          #calculate center of object
            center = (round(left + (width/ 2)), round(top + (height / 2)))
            
            
            #here we determine the direction based on the center of the box
            nom = classes[classID]
            if nom in pointsDict:
                 pointsDict[nom].appendleft(tuple(center))

            else:
            
                 pointsDict[nom] = deque(maxlen=25)
                 pointsDict[nom].appendleft(tuple(center))

            if len(pointsDict[nom]) > 6:
                
                xdir, ydir = getDirection(frame, pointsDict[nom])
            
            else:
                xdir = ".."
                ydir = ".."
                        
        
            topp = max(0, np.floor(top + 0.5).astype('int32'))
            leftt = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(frameWidth, np.floor(top+height + 0.5).astype('int32'))
            right = min(frameHeight, np.floor(left+width + 0.5).astype('int32'))
            boxx = [topp, leftt, bottom, right] 
    
              #estimate distance  
            distance_input = load_dist_input(boxx, classIDs[i], frameWidth, frameHeight)
            distance = distance_model.predict(np.array([distance_input]).reshape(-1, 6))
        
        
        # draw prediction
            drawPred( i, classIDs[i], confidences[i], left, top, left + width, top + height, center, distance, xdir, ydir)


            
def drawPred(i, classId, conf, left, top, right, bottom, center, distance, xdir, ydir):
    
    #draw the rectangle
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50))
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))

        label = '%s%s:%s' % (classes[classId],int(i+1), label)
        
        #the class of the object detected
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        #the direction
        cv.putText(frame, xdir  + " - " + ydir, (left, top - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        #draw the circle
        cv.circle(frame, center, 5, (0, 0, 255), 5)
        
        predict_class = classes[classId]
        cc = ['person', 'bus', 'truck', 'car', 'bicycle', 'motorbike', 'cat', 'dog', 'horse', 'sheep', 'cow']
        if str(predict_class) in cc:
            
            label_left = '{} {} Distance: {:.2f}M'.format(classes[classId], int(i+1), float(np.squeeze(distance)))
        
        #the distance
            cv.putText(frame , label_left, (50,50+i*20), cv.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255), 1)
        
        # the risk of collision
            if distance <1.70 :
                mid_x = center[0]
                if mid_x >0.3 and mid_x <1:
                    # the ridke of the collision
                     cv.putText(frame, "Warniiiing!!!", (left + 20, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255)) 
                    
        

# we create a window where we will display the detction, we name and resize it
winName = 'DL OD with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000, 1000)

cap = cv.VideoCapture('.mp4')
pointsDict = {}


while cv.waitKey(1) < 0:
    
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    
    net.setInput(blob)
    
    #runs the forward pass
    outs = net.forward(getOutputsNames(net))
    
    postprocess(frame, outs)
    
    cv.imshow(winName, frame)
        


