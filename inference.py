import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
from collections import deque
import numpy as np

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="demo",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')
parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

args = parser.parse_args()
args.pretrainModel = '/data/Projects/TalkNet-ASD/exps/exp1/model/model_0025.model'

if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)

args.videoName = 'test_2'
# args.audioFilePath = f'/data/Projects/TalkNet-ASD/demo/{args.videoName}/pyavi/audio.wav'
# args.videoFilePath = f'/data/Projects/TalkNet-ASD/demo/{args.videoName}/pyavi/video.avi'
args.audioFilePath = f'/data/Projects/TalkNet-ASD/demo/test_2/pyavi/audio.wav'
args.videoFilePath = f'/data/Projects/TalkNet-ASD/demo/test_2/pyavi/video.avi'
args.savePath = os.path.join(args.videoFolder, args.videoName)
os.makedirs(args.savePath, exist_ok=True)
os.makedirs(args.savePath+'/', exist_ok=True)
video_only_path = args.savePath+'/inference_result_only_exp1.avi'
video_out_path = args.savePath+'/inference_result_exp1.avi'

DET = S3FD(device='cuda')

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

def crop_face(frame, cs, bbox):
    bs  = max((bbox[3]-bbox[1]), (bbox[2]-bbox[0]))/2  # Detection box size
    bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
    frame = numpy.pad(frame, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
    my  = (bbox[1]+bbox[3])/2 + bsi  # BBox center Y
    mx  = (bbox[0]+bbox[2])/2 + bsi  # BBox center X
    # print(int(my-bs),int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)),int(mx+bs*(1+cs)))
    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
    # cv2.imwrite('face.jpg', face)
    # cy, cx = bbox['y'], bbox['x']
    # face = image[int(cy-bs):int(cy+bs),int(cx-bs):int(cx+bs)]
    # cv2.imwrite('face0.jpg', face)
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
    return face

def asd(s, videoFeature, audioFeature):  # 1 sec of audio and video
    with torch.no_grad():
        inputA = torch.FloatTensor(audioFeature).unsqueeze(0).cuda()
        inputV = torch.FloatTensor(videoFeature).unsqueeze(0).cuda()
        # print(inputA.shape, inputV.shape)# --> torch.Size([1, 100, 13]) torch.Size([1, 25, 112, 112])
        embedA = s.model.forward_audio_frontend(inputA)
        embedV = s.model.forward_visual_frontend(inputV)	
        embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
        out = s.model.forward_audio_visual_backend(embedA, embedV)
        score = s.lossAV.forward(out, labels = None)
    return score
        
def box_color(iperson):
    color = (iperson, iperson*30, iperson*2*30)
    return color

# def check(seq_faces, seq_boxes):
    # pass
    # print('Length of seq_boxes: ', len(seq_boxes))
    # print('Length of seq_faces: ', len(seq_faces))

    # for iface, seq_box in enumerate(seq_boxes):
        # print('iface, number of boxes: ', iface, len(seq_box))

    # for iface, seq_faces in enumerate(seq_faces):
        # print('iface, number of faces: ', iface, len(seq_faces))

def process(video_file, audio_file, args):
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
    s.eval()

    iouThres  = 0.5     # Minimum IOU between consecutive face detections
    cs  = args.cropScale
    seq_faces = []
    seq_boxes = []
    seq_audio = []

    _, audio = wavfile.read(audio_file)
    audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010) # len: 29995

    cap = cv2.VideoCapture(video_file)
    max_person_id = 0
    iframe = 0

    while cap.isOpened():
        print('frame: ', iframe)
        ret, frame = cap.read()

        if ret is False:
            break

        if iframe == 0:
            h, w, _ = frame.shape
            vOut = cv2.VideoWriter(video_only_path,  cv2.VideoWriter_fourcc(*'DIVX'), 20.0,(w, h))


        # Step 1: Face detection
        imageNumpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        # First frame
        if iframe == 0: 
            for iperson, bbox in enumerate(bboxes):
                bbox = bbox.tolist()
                box_per_person = deque([], 25)
                box_per_person.append({'bbox': bbox, 'id': iperson, 'frameidx': iframe})
                seq_boxes.append(box_per_person)

                face = crop_face(frame, cs, bbox)
                face_per_person = deque([], 25)
                face_per_person.append(face)
                seq_faces.append(face_per_person)
                max_person_id += 1
            iframe += 1
            continue

        # check(seq_faces, seq_boxes)
        # Step 2: remove all discontinued faces
        iperson = 0
        while True:
            if iperson >= len(seq_boxes):
                break
            seq_box = seq_boxes[iperson]
            seq_face = seq_faces[iperson]
            frameidx = seq_box[-1]['frameidx']
            if iframe - frameidx > 5: # discontinued, remove
                seq_boxes.remove(seq_box) ### not sure
                seq_faces.remove(seq_face) ### not sure
                # print(len(seq_boxes), num_person)
                assert(len(seq_boxes) == num_person-1)
                num_person = len(seq_boxes)
            else:
                iperson += 1
            
        # Step 3: Match
        for ibox, bbox in enumerate(bboxes):
            bbox = bbox.tolist()
            face = crop_face(frame, cs, bbox)
            matched = False
            num_person = len(seq_boxes)
            for iperson, seq_box in enumerate(seq_boxes):
                if matched == True: # no more matching, assuming no overlapping
                    break
                last_box = seq_box[-1]['bbox']
                person_id = seq_box[-1]['id']
                frameidx = seq_box[-1]['frameidx']
                iou = bb_intersection_over_union(bbox, last_box)
                if iou > iouThres:
                    matched = True
                    seq_boxes[iperson].append({'bbox': bbox, 'id': person_id,'frameidx': iframe})
                    seq_faces[iperson].append(face)
                    cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), box_color(person_id),10)
                    cv2.putText(frame,'%d'%(person_id), (int(bbox[0]),int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, box_color(person_id),5)
                
            if matched == False: # new person
                box_per_person = deque([], 25)
                box_per_person.append({'bbox': bbox, 'id': max_person_id,'frameidx': iframe})
                seq_boxes.append(box_per_person)

                face_per_person = deque([], 25)
                face_per_person.append(face)
                seq_faces.append(face_per_person)
                max_person_id += 1
                cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), box_color(max_person_id+1),10)
                cv2.putText(frame,'%d'%(max_person_id), (int(bbox[0]),int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, box_color(max_person_id+1),5)

        # check(seq_faces, seq_boxes)
        if iframe < 25:
            iframe+=1
            continue
        for iface, seq_box in enumerate(seq_boxes):
            if len(seq_box) == 25:
                # print('asd activated ', iframe)
                videoFeature_seq = seq_faces[iface]
                videoFeature_seq = numpy.array(videoFeature_seq)
                videoFeature_seq = videoFeature_seq.squeeze()
                audioFeature_seq = audioFeature[(iframe-25)*4:(iframe-1)*4+4]
                scores = asd(s, videoFeature_seq, audioFeature_seq)
                # print('person id, scores ', seq_box[-1]['id'], scores)
                bbox = seq_box[-1]['bbox']
                final_score = np.mean(scores[-5:])
                if final_score >= 0:
                    cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 255, 0),15)

        vOut.write(frame)
        cv2.imwrite(f'{args.savePath}/tracking/{iframe}.jpg', frame)
        iframe+=1
    vOut.release()
    cap.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
        (video_only_path, args.audioFilePath, \
        args.nDataLoaderThread, video_out_path)) 
    output = subprocess.call(command, shell=True, stdout=None)

# Main function
def main():
    # Active Speaker Detection by TalkNet
    scores = process(args.videoFilePath, args.audioFilePath, args)


if __name__ == '__main__':
    main()
