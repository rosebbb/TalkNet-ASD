import torch, pyaudio, cv2, python_speech_features
from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
import numpy as np
from scipy.io import wavfile
import scipy.io.wavfile as wav

import warnings
from threading import Thread
import cv2, time
from collections import deque

warnings.filterwarnings("ignore")

# Camera setup
hardwareWidth = 1080
hardwareHeight = 1920
# cam_id = 0
cam_id = '/data/Projects/TalkNet-ASD/demo/camera2.mp4'
# cam_id = 'videos/camera2_short.mp4'

# Load Pretrained Model
s = talkNet()
s.loadParameters("pretrained/pretrain_TalkSet.model")
s.eval()

# IoU for face tracking/ identification
def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def get_bbox(bbox, frame):
    xMin = max(0,int(bbox[0]))
    xMax = min(int(bbox[2]), width)
    yMin = max(int(bbox[1]), 0)
    yMax = min(int(bbox[3]), height)
    face = frame[yMin: yMax, xMin : xMax, :]
    face = cv2.resize(face, (112,112))
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    return face
            
# callback function for audio streaming thread
def callback(in_data, frame_count, time_info, status):
    global old, mfcc, audioFeature, AudioCounter
    data = np.frombuffer(in_data, dtype=np.int16)
    if((old == data).all() == False):
        mfcc = python_speech_features.mfcc(data.astype(np.int16), 16000, numcep = 13, winlen = 0.025, winstep = 0.010) # (N_frames, 13)   [1s = 100 frames]
        audioFeature = np.append(audioFeature, mfcc, axis=0)
        audioFeature = audioFeature[-96:,:]
        print('audioFeature', audioFeature)
        old = data
        AudioCounter = AudioCounter + 1
    return (in_data, pyaudio.paContinue)

# Thread for face detection and tracking
class FaceDetectTrack(object):
    def __init__(self, frame, seq_boxes, seq_faces, frame_no):            
        self.frame = frame
        self.stopped = False
        self.seq_boxes = seq_boxes
        self.seq_faces = seq_faces # initialize with the first 25 frames
        self.frame_no = frame_no
        print('-------------------initialize facetracker:')

    # method to start thread 
    def start(self):
        print('----------start facetracker:')
        Thread(target=self.process_face, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def match_bbox(self, bboxes):
        self.frame_no.append(max(self.frame_no)+1)
        for iface, bbox in enumerate(bboxes):
            face = get_bbox(bbox, frame)
            for jface, boxes_seq in enumerate(self.seq_boxes):
                last_box = boxes_seq[-1]
                if (bb_intersection_over_union(last_box, bbox) > 0.5):
                    self.seq_boxes[jface].append(bbox)
                    self.seq_faces[jface].append(face) # assuming no new face, i.e. ignore new face
                    
                    if len(self.seq_boxes[jface]) > 24:
                        self.seq_boxes[jface].pop(0)
                        self.seq_faces[jface].pop(0)
                        self.frame_no.pop(0)


    def process_face(self):
        # Face detection
        print('------------running facetracker:')

        bboxes = DET.detect_faces(self.frame, conf_th=0.9, scales=[0.25])
        self.match_bbox(bboxes)
        

class ASD(object):
    def __init__(self, seq_boxes, seq_faces):
        self.stopped = False
        self.seq_boxes = seq_boxes
        self.seq_faces = seq_faces # initialize with the first 25 frames
        num_person = len(self.seq_boxes)
        self.scores = [0] * num_person

    # method to start thread 
    def start(self):
        Thread(target=self.asd_process, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def asd_process(self):
        global audioFeature
        bboxes_temp = bboxes.copy()
        num_person = len(self.seq_boxes)
        # Inference model for each bbox person
        for iface in range(num_person):
            tempVideoFeature = np.array(self.seq_faces[iface])
            inputA = torch.FloatTensor(audioFeature).unsqueeze(0).cuda()
            inputV = torch.FloatTensor(tempVideoFeature).unsqueeze(0).cuda()
            embedA = s.model.forward_audio_frontend(inputA)
            embedV = s.model.forward_visual_frontend(inputV)
            embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
            out = s.model.forward_audio_visual_backend(embedA, embedV)
            score = s.lossAV.forward(out, labels = None)
            # Use 4/24 sec delay ~ 0.16 sec
            # lastScoreArray[iface] = np.mean(score[20])
            #print(str(iface) + " Score : " + str(lastScoreArray[iface]))

            self.scores[iface] = score
            # # Threshold for determination of active/ inactive speaker
            # if lastScoreArray[faceIdx] > -0.5:
            #     colorArray[faceIdx] = (0,255,0)
            # else:
            #     colorArray[faceIdx] = (0,0,255)

class FrameGetter(object):
    def __init__(self, cap, frames_queue, frames_processed):
        self.frames_queue = frames_queue 
        self.frames_processed = frames_processed
        self.frame_no = 0
        self.cap = cap
        print('-------------------initialize video capture:')

    def start(self):
        print('----------start video capture:')
        Thread(target=self.run, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def run(self):
        cv2.namedWindow('input', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        if not self.cap.isOpened():
            print('Video is not captured')
        while self.cap.isOpened():
            print('Getting Frame: ', self.frame_no)
            ret, frame = self.cap.read()
            # lock.acquire()            
            self.frames_queue.append(frame)
            self.frames_processed.append(0)
            self.frame_no += 1
            # lock.release()
        #     cv2.imshow('input', frame)
        #     if cv2.waitKey(1) == ord('q'):
        #         break
        # print("实时读取线程退出！！！！")
        # cv2.destroyWindow('ip_camera')
        self.frames_queue.clear()    # 读取进程结束时清空队列
        self.cap.release()

# Video Thread Setup, Assume 1920x 1080 output
videoBuffer = []
oldVideo = np.zeros((hardwareHeight, hardwareWidth, 3))
bboxes = np.zeros((1,1))

#Audio Thread Setting
CHUNK = 3840 # number of data points to read at a time (0.24 sec)
RATE = 16000 # time resolution of the recording device (Hz)

# 0.96 sec of audio (0.96 * 16000) < - > 24 frame of images (25 fps)
AudioCounter = 0
AudioBuffer = np.zeros(15360)
audioFeature = np.zeros((96,13))
old = np.zeros(3840)

# ASD model input setup (buffer)

lastScoreArray = []
flagArray = []

# #Assume at most 10 people in conference room
# for step in range(10):
#     videoFeature.append([])
#     videoBbox.append([])
#     colorArray.append((0,0,255))
#     lastScoreArray.append(-100)
#     flagArray.append(False)

# Face Detector Setup
DET = S3FD(device='cuda')

cap = cv2.VideoCapture(cam_id)
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, hardwareWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hardwareHeight)

# Initilaization for person identification & buffering for ASD inference
inf_length = 24
videoFeature = [] # len = number of person, videoFeature[0]: last 24 faces of the person 0
videoBbox = []
frames_queue = deque([], inf_length)
frames_processed = deque([], inf_length)
num_person = 0
for step in range(inf_length):
    ret, frame = cap.read()
    frames_queue.append(frame)
    frames_processed.append(1)
    cv2.imwrite('frane.jpg', frame)
    height, width, _ = frame.shape
    bboxes = DET.detect_faces(frame, conf_th=0.9, scales=[0.25]) # person1, person2, person3 
    for iperson, bbox in enumerate(bboxes):
        matched = 0
        face = get_bbox(bbox, frame)
        cv2.imwrite(f'face{iperson}.jpg', face)
        if step == 0:
            num_person = len(bboxes)
            videoBbox.append(deque([], inf_length))
            videoBbox[-1].append(bbox)
            videoFeature.append(deque([], inf_length))
            videoFeature[-1].append(face)
        else:
            for jperson in range(num_person):
                trackbBox = videoBbox[jperson][-1]
                if (bb_intersection_over_union(trackbBox, bbox) > 0.5):
                    matched = 1
                    videoBbox[jperson].append(bbox)
                    videoFeature[jperson].append(face)
                    break
            if matched == 0:
                videoBbox.append([bbox])
                videoFeature.append([face])
                num_person += 1

# print(len(videoBbox))
# for iperson in range(len(videoBbox)):
#     print(f'\nvideoBbox[{iperson}], ', videoBbox[iperson])
#     for iframe in range(len(videoFeature[iperson])):
#         cv2.imwrite(f'temp/{iperson}_{iframe}.jpg', videoFeature[iperson][iframe])
# Start the PyAudio class
p=pyaudio.PyAudio()

# Initiate all threads
# face detection 
(grabbed, frame) = cap.read()
frame_no = list(range(24))
frames = FrameGetter(cap, frames_queue, frames_processed).start()
# face_tracker = FaceDetectTrack(frame, videoBbox, videoFeature, frame_no).start()
# active_speaker_detector = ASD(videoBbox, videoFeature).start()
# stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
            #   frames_per_buffer=CHUNK, stream_callback=callback) #uses default input device

# SaveCounter = 0
# while True:
#     (grabbed, frame) = cap.read()
#     face_tracker.frame = frame
#     # active_speaker_detector.seq_boxes = face_tracker.seq_boxes
#     # active_speaker_detector.seq_faces = face_tracker.seq_faces
#     # scores = active_speaker_detector.scores

#     # print(face_tracker.seq_boxes)
#     for iface, bboxes in enumerate(face_tracker.seq_boxes):
#         bbox = bboxes[-1]
#         xMin = max(0,int(bbox[0]))
#         xMax = min(int(bbox[2]), width)
#         yMin = max(int(bbox[1]), 0)
#         yMax = min(int(bbox[3]), height)

#         cv2.rectangle(frame,(xMin,yMin),(xMax,yMax),(0,255,0),2)

#         # #cv2.putText(frame_temp, str(bboxIdx), (xMin,yMin-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,colorArray[bboxIdx], 2, cv2.LINE_AA, False)
#         # cv2.putText(frame, str(lastScoreArray[bboxIdx]), (xMin,yMin), cv2.FONT_HERSHEY_SIMPLEX, 0.5,colorArray[bboxIdx], 2, cv2.LINE_AA, False)
#         # if score[bboxIdx]>0.5:
#         #     cv2.rectangle(frame,(xMin,yMin),(xMax,yMax),(0,255,0),2)
#         # else:
#         #     cv2.rectangle(frame,(xMin,yMin),(xMax,yMax),(0.0,255),2)
#     cv2.imshow('video', frame)

#     c = cv2.waitKey(1)
#     if c == 27:
#         face_tracker.stop()
#         active_speaker_detector.stop()
#         break

# face_tracker.stop()
# active_speaker_detector.stop()
# stream.stop_stream()
# stream.close()
# p.terminate()

# print("Finished")