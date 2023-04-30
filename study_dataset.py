import os
import random
import cv2
import glob
import subprocess
import pandas
import tqdm

def videos():
    folder = '/home/xingliu/Datasets/AVADataPath/clips_videos/train'

    video_names = os.listdir(folder)

    clip_names = []
    for video in video_names:
        # print(len(os.listdir(os.path.join(folder, video))))
        temp_names = os.listdir(os.path.join(folder, video))
        temp_names = [os.path.join(folder, video, x) for x in temp_names]
        clip_names += temp_names
    return clip_names

def audios():
    folder = '/home/xingliu/Datasets/AVADataPath/clips_audios/train'

    audio_names = os.listdir(folder)

    clip_names = []
    for audio in audio_names:
        temp_names = os.listdir(os.path.join(folder, audio))
        temp_names = [os.path.join(folder, audio, x) for x in temp_names]
        clip_names  += temp_names
    return clip_names

def get_num_clips():
    audio_clips = audios()
    video_clips = videos()
    for audio_clip in audio_clips:
        if audio_clip.split('.')[0] not in video_clips:
            print(audio_clip)

def check_ann(clip_name):
    ann_file = '/home/xingliu/Datasets/AVADataPath/csv/train_orig.csv'
    df = pandas.read_csv(os.path.join(ann_file))
    df = df.groupby('entity_id')

    insData = df.get_group(clip_name)
    print(len(insData))
    labels = []
    image_list = []
    for _, row in insData.iterrows():
        labels.append(int(row['label_id']))
        image_list.append(str("%.2f"%row['frame_timestamp'])+'.jpg')

    return labels, image_list
def gen_sample_clips():
    audio_clips = audios()
    video_clips = videos()

    audio_clips.sort()
    video_clips.sort()


    clips = random.choices(video_clips, k=30)	

    for clip in clips:
        
        clip_name = os.path.basename(clip)
        labels, image_list = check_ann(clip_name)
        print(clip_name)
        print(image_list)
        video_name = '/data/Projects/TalkNet-ASD/figures/sample_training_clips/'+ clip_name+'.avi'

        frames = glob.glob(clip +'/*.jpg')
        frames.sort()
        min_h, min_w = 10000, 10000
        for frame_file in frames:
            frame = cv2.imread(frame_file)
            height, width, layers = frame.shape
            min_h = min(height, min_h)
            min_w = min(width, min_w)

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 25, (min_w,min_h))
        assert(len(frames) == len(labels))
        for i, frame_file in enumerate(frames):
            print(frame_file)
            frame = cv2.imread(frame_file)

            h, w, _ = frame.shape
            cut_w, cut_h =0 ,0
            if h > min_h:
                cut_h = (h-min_h)//2
            if w > min_w: 
                cut_w = (w - min_w)//2

            frame_crop = frame[cut_h:cut_h+min_h, cut_w:cut_w+min_w, :]
            frame_crop = cv2.putText(frame_crop, str(labels[i]), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite('image.jpg', frame_crop)
            video.write(frame_crop)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        video.release()
    
        # Combine video and audio
        audio_file = clip.replace('video', 'audio')+'.wav'
        command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
            (video_name, audio_file, 10, os.path.dirname(video_name)+'/out_'+os.path.basename(video_name)))
        output = subprocess.call(command, shell=True, stdout=None)


gen_sample_clips()