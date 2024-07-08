import cv2
min_area = 500
ovp = "moutput.avi"
video_source = 0 
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("There is a error, unable to access video source...")
    exit()

fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(ovp, fc, fps, (fw, fh))
bg = cv2.createBackgroundSubtractorMOG2()

md = False
mcounter = 0

while cap.isOpened():
    rt, fm = cap.read()
    if not rt:
        break

    fgm = bg.apply(fm)
    krnl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    fgm = cv2.morphologyEx(fgm, cv2.MORPH_OPEN, krnl)

    cntrs, _ = cv2.findContours(fgm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cntrs:
        if cv2.contourArea(contour)<min_area:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(fm, (x, y), (x+w, y+h), (0,165,255), 2)

        md = True
        mc = fps

    if md:
        out.write(fm)
        mc -= 1
        if mc <= 1:
            md = False

    cv2.imshow('Motion Detection', fm)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()