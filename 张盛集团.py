from ultralytics.models import YOLO
import numpy as np
import cv2

def trans(tensor):
    return np.array(tensor.numpy())

dt = {0:"zyx",1:"sl"}
color = {"zyx":(0,0,255),"sl":(255,255,0)}
camera = cv2.VideoCapture(0)
count = 0
model = YOLO(model='E:/python objects/YOLO-Project/ultralytics-main/datasets/人脸识别/runs/train/exp7/weights/best.pt')
while (1):
    ret, img = camera.read()  # 读取每一帧
    results = model(img)
    img = np.array(img)
    for result in results:
        boxes = result.boxes
        if list(result.boxes.shape)[0] == 2:
            for i in range(2):
                x=int(trans(boxes.xywh[i][0].cpu())-trans(boxes.xywh[i][2].cpu())/2)
                y=int(trans(boxes.xywh[i][1].cpu())-trans(boxes.xywh[i][3].cpu())/2)
                w=int(trans(boxes.xywh[i][2].cpu()))
                h=int(trans(boxes.xywh[i][3].cpu()))
                a = dt[int(trans(boxes.cls[i].cpu()))]  # 类别名称
                b = format(trans(boxes.conf[i].cpu()) , '.2f') # 置信度
                font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
                y_top = int(y-(h/6))
                cv2.rectangle(img, (x, y), (x + w, y + h),color[a], 2)
                cv2.putText(img, '   {} {}'.format(a, b), (x, y_top), font, 0.8, (0, 0, 0), 2)
        elif list(result.boxes.shape)[0] == 1:
            x=int(trans(boxes.xywh[0][0].cpu())-trans(boxes.xywh[0][2].cpu())/2)
            y=int(trans(boxes.xywh[0][1].cpu())-trans(boxes.xywh[0][3].cpu())/2)
            w=int(trans(boxes.xywh[0][2].cpu()))
            h=int(trans(boxes.xywh[0][3].cpu()))
            a = dt[int(trans(boxes.cls[0].cpu()))]  # 类别名称
            b = format(trans(boxes.conf[0].cpu()) , '.2f') # 置信度
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            y_top = int(y-(h/6))
            cv2.rectangle(img, (x, y), (x + w, y + h), color[a], 2)
            cv2.putText(img, '   {} {}'.format(a, b), (x, y_top), font, 0.8, (0, 0, 0), 2)
        else:
            continue
    cv2.imshow("face",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count+=1
camera.release()
cv2.destroyAllWindows()

