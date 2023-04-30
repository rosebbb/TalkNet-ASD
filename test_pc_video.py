import cv2

print("enter cam id")
cam_id = int(input())
print("recording via index "+str(cam_id))

cap = cv2.VideoCapture(cam_id)
ret, frame = cap.read()
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite('frame3.jpg', frame)
cap.release()
cv2.destroyAllWindows()