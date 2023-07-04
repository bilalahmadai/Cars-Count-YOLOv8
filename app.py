import cv2 
from ultralytics import YOLO
from sort import *

tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

model = YOLO("yolov8n.pt")
video_path="input.mp4"
video_path_out = 'OutPut.mp4'

vid=cv2.VideoCapture(video_path)
ret,frame= vid.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(vid.get(cv2.CAP_PROP_FPS)), (W, H))

mask=cv2.imread('mask.png')
road_mask=cv2.bitwise_and(frame,mask)

# limit=[180,505,1275,505]
limit=[410,364,721,364]
vechicalCounter=[]

while True:
    ret,frame= vid.read()
    road_mask=cv2.bitwise_and(frame,mask)
   
    print(ret)
    results = model(road_mask)
    result = results[0]

    detections=np.empty((0, 5))

    lineColor=(245, 150, 108)


    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        
        threshold=0.5
        x1,y1,x2,y2=cords
        if conf > threshold:
            
            if class_id=='car' or class_id=='truck' or class_id=='motorcycle' or class_id=='bus':
                # cv2.putText(frame, f"{class_id}", (x1+10, (y2 - 20)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 1, cv2.LINE_AA)
  
                
                curr_array=[x1,y1,x2,y2,conf]
                detections=np.vstack((detections,curr_array))
            
            
    
    reult_tracker = tracker.update(detections)
    print("vechicalCounter: ",vechicalCounter)
    for track in reult_tracker:
        x1,y1,x2,y2,id=track
        x1,y1,x2,y2,id=int(x1),int(y1),int(x2),int(y2),int(id)
        w,h=x2-x1,y2-y1
        cx,cy=x1+w//2 , y1+h//2

        # print(track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,2), 2)
       
        cv2.putText(frame, f"id: {int(id)}", (x1, (y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,2), 1, cv2.LINE_AA)
        cv2.circle(frame,(cx,cy),5,(255,0,0),cv2.FILLED)
        if limit[0]<cx<limit[2] and limit[1]-20<cy<limit[1]+20:
            lineColor=(108, 66, 245)
            if id not in vechicalCounter:
                vechicalCounter.append(id)

        
    cv2.line(frame,(limit[0],limit[1]),(limit[2],limit[3]),lineColor,4)
    cv2.putText(frame,f"Count: {len(vechicalCounter)}",(1080,52),cv2.FONT_HERSHEY_COMPLEX,1,lineColor,1)
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cv2.imshow("out",frame)
    # cv2.imshow("out2",road_mask)
    # cv2.imwrite("maskedFrame.png",road_mask)
    # out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
vid.release()
out.release()
cv2.destroyAllWindows()