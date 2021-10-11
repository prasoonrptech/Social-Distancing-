#importing the required libraries
import cv2
import datetime
import imutils
import numpy as np
from itertools import combinations
from tracker.centroidtracker import CentroidTracker
import math
import json
import os



gpu=True

try:
    #get the model files for caffee model
    protopath = "/home/jet245/Downloads/social_dist/model/MobileNetSSD_deploy.prototxt.txt"
    modelpath = "/home/jet245/Downloads/social_dist/model/MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
except:
    print("Model Files missing!")
    pass

if gpu == True:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


#list of classes available
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


tracker = CentroidTracker(maxDisappeared=20)


# non_max_suppression function to avoid the over lapping boxes
def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))



#function for generating social_distance alert  and registering the IDs
def social_dist_alert(distance, id1, id2, socialDist_alert):
    if distance < 60:
        if id1 not in socialDist_alert:
            print(f"social Distance violated ID:{id1}")
            socialDist_alert.append(id1)
        if id2 not in socialDist_alert:
            print(f"social Distance violated ID:{id2}")
            socialDist_alert.append(id2)


#main driver function
def main():
    # get the video from the source
    cap = cv2.VideoCapture("/home/jet245/Downloads/testvideo2.mp4")

    #list for appending the Ids
    socialDist_alert= []


    total_frames = 0
    lpc_count = 0
    # object_id_list = []

    #checking if the feed is live from the source
    if (cap.isOpened() == False):
        print("Error opening video stream")


    while True:
        #read the incoming files
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        #generate blob for each frame
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        # blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        #pass each blob through the model
        detector.setInput(blob)
        person_detections = detector.forward()

        #for storing the coordinates of the bounding box
        rects = []
        #iterate over the detection
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            #confidence threshold
            if confidence > 0.65:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        #convert the box-coordinates to numpy array and pass them into non_max_supression
        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        #dictionary for appending the rect
        centroid_dict = dict()

        #updating the tracker with new non_max_suppressed coordinates
        #object: gives list of all the object_id present in the frame
        objects = tracker.update(rects)

        #iterate over the trackers w.r.t objectIDs and bbox coordinates
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            #calculate the centroid for each box
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)

            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)

            #for calculating the distance between centroids and check which centroids are violating social distance
            for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
                dx, dy = p1[0] - p2[0], p1[1] - p2[1]
                #calculate the distance
                distance = math.sqrt(dx * dx + dy * dy)
                #for running the social_distance alert
                social_dist_alert(distance, id1, id2, socialDist_alert)
                

            for id, box in centroid_dict.items():
                if id in socialDist_alert:
                    cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)


        #for getting the person_count
        person_count = len(objects)

        #print on the frame
        lpc_txt = "Live count: {}".format(person_count)
        cv2.putText(frame, lpc_txt, (1, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)


        #display the feed
        cv2.imshow("Application", frame)
        key= cv2.waitKey(13)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

main()
