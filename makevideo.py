import cv2
import glob
import os

image_folder = '/data/Projects/TalkNet-ASD/demo/presenter_3/tracking'
video_name = '/data/Projects/TalkNet-ASD/demo/presenter_3/tracking.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# print(imgs)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 25, (width,height))

for i in range(25, 202):
    video.write(cv2.imread(os.path.join(image_folder, str(i)+'.jpg')))

cv2.destroyAllWindows()
video.release()