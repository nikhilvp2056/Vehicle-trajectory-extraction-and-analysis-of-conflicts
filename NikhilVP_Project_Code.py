#VEHICLE TRAJECTORY DETECTION AND ANALYSIS OF CONFLICTS


import numpy as np
import cv2


# load the video
cap = cv2.VideoCapture('data_video.MTS')

min_width_react = 60 #min width rectangle
min_hieght_react = 60 #min width rectangle

count_line_position = 450

#Initialize Substructor Algorithm
algo = cv2.bgsegm.createBackgroundSubtractorMOG()


def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx= x+x1
    cy= y+y1
    return cx,cy

detect = []
offset=6 #Allowable error between pixel
counter=0

while True:
    # Read a new frame.
    ret,frame1= cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    # apply the background object on each frame
    img_sub = algo.apply(blur)
    # Apply some morphological operations to make sure you have a good masking
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    # Detect contours in the frame.
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(0,50),(20000,10000),(255,127,0),3)

    # loop over each contour found in the frame.
    for (i,c) in enumerate(counterShape):
        # Retrieve the bounding box coordinates from the contour.
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_react) and (h>= min_hieght_react)
        if not validate_counter:
            continue

        # Draw a bounding box around the vehicle.
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)

        # Write Car Detected near the bounding box drawn.
        #cv2.putText(frame1,"Truck"+str(counter),(x, y-100),cv2.FONT_HERSHEY_TRIPLEX, 1,(255,244,0),2)
        #cv2.putText(frame1,"Car"+str(counter),(x, y-40),cv2.FONT_HERSHEY_TRIPLEX, 1,(255,244,0),2)
        #cv2.putText(frame1,"Bus"+str(counter),(x, y-80),cv2.FONT_HERSHEY_TRIPLEX, 1,(255,244,0),2)
        #cv2.putText(frame1,"Motobike"+str(counter),(x, y-60),cv2.FONT_HERSHEY_TRIPLEX, 1,(255,244,0),2)
        
        # List for store vehicle count information
        #temp_up_list = []
        #temp_down_list = []
        #up_list = [0, 0, 0, 0]
        #down_list = [0, 0, 0, 0]

        #font_color = (0, 0, 255)
        #font_size = 0.5
        #font_thickness = 2

        #cv2.putText(frame1, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        #cv2.putText(frame1, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        #cv2.putText(frame1, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        #cv2.putText(frame1, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)



        center= center_handle(x,y,w,h)
        detect.append(center) 
        cv2.circle(frame1,center,4, (0,0,255),-1)


        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
                cv2.line(frame1,(0,50),(20000,10000),(0,127,255),3)
                detect.remove((x,y))
                print("Vehicle Counter:"+str(counter))


    cv2.putText(frame1,"VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)


   
    #cv2.imshow('Detecter',dilatada)

    #Display the output video
    cv2.imshow('Video Original',frame1)

    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xff
    
    # Check if 'q' key is pressed.
    if k == ord('q'):
        # Break the loop.
        break


# Release the VideoCapture Object.
cap.release()
# Close the windows.q
cv2.destroyAllWindows()
