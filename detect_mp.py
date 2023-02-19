from multiprocessing import Process, Queue, Pipe
import cv2
import time
import numpy as np
from model.faceDetector.s3fd import S3FD
import os
import warnings
from collections import deque
from talkNet import talkNet
import torch

warnings.filterwarnings("ignore")

def asd(s, videoFeature, audioFeature):  # 1 sec of audio and video
    with torch.no_grad():
        inputA = torch.FloatTensor(audioFeature).unsqueeze(0).cuda()
        inputV = torch.FloatTensor(videoFeature).unsqueeze(0).cuda()
        print(inputA.shape, inputV.shape)# --> torch.Size([1, 100, 13]) torch.Size([1, 25, 112, 112])
        embedA = s.model.forward_audio_frontend(inputA)
        embedV = s.model.forward_visual_frontend(inputV)	
        embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
        out = s.model.forward_audio_visual_backend(embedA, embedV)
        score = s.lossAV.forward(out, labels = None)
    return score

def crop_face(frame, bbox, cs = 0.40):
    bs  = max((bbox[3]-bbox[1]), (bbox[2]-bbox[0]))/2  # Detection box size
    bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
    frame = np.pad(frame, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
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

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
    
def box_color(iperson):
    color = (iperson, iperson*30, iperson*2*30)
    return color


def person_detect_mp(Original_frames, Predicted_boxes, Processing_times): # needs to run on every frame
    DET = S3FD(device='cuda')
    times = []
    while True:
        if Original_frames.qsize()>0:
            item = Original_frames.get()
            frame = item['frame']
            frame_idx = item['frame_idx']
            print('-------1------- Running person detect on frame:', frame_idx)
            t1 = time.time()
            Processing_times.put(time.time())
            pred_bboxes = DET.detect_faces(frame, conf_th=0.9, scales=[0.25])
            Predicted_boxes.put({'frame': frame, 'pred_bboxes': pred_bboxes, 'frame_idx': frame_idx})

def postprocess_mp(Predicted_boxes, Processed_frames):
    while True:
        if Predicted_boxes.qsize()>0:
            item = Predicted_boxes.get()
            frame_idx = item['frame_idx']
            frame = item['frame']
            pred_bboxes = item['pred_bboxes']
            print('-------3------- Writing boxes on frame:', frame_idx)

            for iface, bbox in enumerate(pred_bboxes):
                bbox = bbox.tolist()
                cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 255, 0),10)

            Processed_frames.put({'frame': frame, 'frame_idx': frame_idx})

# def asd_mp(Original_frames):
#     global max_person_id, seq_faces, seq_boxes, Predicted_asd

#     s = talkNet()
#     s.loadParameters('pretrain_TalkSet.model')
#     s.eval()

#     for iface, seq_box in enumerate(seq_boxes):
#         if len(seq_box) == 25:
#             print('asd activated ')
#             videoFeature_seq = seq_faces[iface]
#             videoFeature_seq = np.array(videoFeature_seq)
#             videoFeature_seq = videoFeature_seq.squeeze()
#             audioFeature_seq = audioFeature[(iframe-25)*4:(iframe-1)*4+4]
#             scores = asd(s, videoFeature_seq, audioFeature_seq)
#             print('person id, scores ', seq_box[-1]['id'], scores)
#             bbox = seq_box[-1]['bbox']
#             face_id = seq_box[-1]['face_id']
#             final_score = np.mean(scores[-5:])
#             Predicted_asd[face_id] = {'bbox': bbox, 'score': final_score}

def detect_video_realtime_mp(cam_id, output_path, show=True, realtime=False):
    cap = cv2.VideoCapture(cam_id)

    # by default VideoCapture returns float instead of int
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')    
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Original_frames = Queue()
    Predicted_boxes = Queue()
    Processed_frames = Queue()
    Tracked_frames = Queue()
    Processing_times = Queue()
    Final_frames = Queue()
    Predicted_asd = {}
    seq_faces = []
    seq_boxes = []
    max_person_id = -1    
    p1 = Process(target=person_detect_mp, args=(Original_frames, Predicted_boxes, Processing_times))
    # p2 = Process(target=tracking_mp, args=(Predicted_boxes, Tracked_frames))
    p3 = Process(target=postprocess_mp, args=(Predicted_boxes, Processed_frames))
    p1.start()
    # p2.start()
    p3.start()

    started = False

    iframe = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        os.makedirs('temp/Original_frames', exist_ok=True)
        cv2.imwrite(f'temp/Original_frames/{iframe}.jpg', frame)

        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Original_frames.put({'frame': original_frame, 'frame_idx': iframe})
        print(f'-------0------- put frame {iframe} in Original_frames at: ',  time.time())
        iframe += 1

        if Processed_frames.qsize()>0:
            item = Processed_frames.get()
            frame = item['frame']
            frame_idx = item['frame_idx']
            Final_frames.put({'frame': frame, 'frame_idx': frame_idx})
            if show:
                print(show)
                cv2.imshow('output', frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

        while started == False and Original_frames.qsize()>0:
            if Processed_frames.qsize() == 0:
                time.sleep(0.1)
                print("wait")
                continue
            else:
                started = True
                start_time = time.time()
                print("break")
                break

    while True:
        if Original_frames.qsize() == 0 and Predicted_boxes.qsize() == 0  and Processed_frames.qsize() == 0  and Processing_times.qsize() == 0 and Final_frames.qsize() == 0:
            p1.terminate()
            # p2.terminate()
            p3.terminate()
            break
        elif Final_frames.qsize()>0:
            image = Final_frames.get()
            # if output_path != '': out.write(image)

    end_time = time.time()
    # print("total_duration", end_time-start_time) 
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    # cam_id = '/data/Projects/TalkNet-ASD/demo/test/pyavi/video.avi'
    cam_id= 1
    output_path = './temp/'
    detect_video_realtime_mp(cam_id, output_path, show=True, realtime=False)
