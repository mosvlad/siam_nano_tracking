import cv2
import datetime

cap = cv2.VideoCapture(0)

params = cv2.TrackerNano_Params()
params.backbone = 'nanotrack_backbone_sim.onnx'
params.neckhead = 'nanotrack_head_sim.onnx'
tracker = cv2.TrackerNano_create(params)

_, frame = cap.read()

box = cv2.selectROI("frame", frame)
tracker.init(frame, box)

while True:
    _, frame = cap.read()
    start = datetime.datetime.now()
    flag, box = tracker.update(frame)
    stop = datetime.datetime.now()
    delta = stop - start
    print(int(delta.total_seconds() * 1000), " ms")
    if flag:
        frame = cv2.rectangle(frame, box, (255, 0, 0), 2)

    cv2.imshow("img", frame)
    cv2.waitKey(1)