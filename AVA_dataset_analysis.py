import glob
import os
import random
import cv2
import subprocess
import ffmpeg
# (
#     ffmpeg
#     .input('/path/to/jpegs/*.jpg', pattern_type='glob', framerate=25)
#     .output('movie.mp4')
#     .run()
# )

#     os.system("ffmpeg -r 1 -i img%01d.png -vcodec mpeg4 -y movie.mp4")

def make_video(video_path, image_files, audio_path, output_file,h,w):
    video = cv2.VideoWriter(video_path, 0, 25, (w,h))
    image_files.sort()
    for image_file in image_files:
        face = cv2.imread(image_file)
        cv2.imwrite('face.jpg', face)
        video.write(face)
    video.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
        (video_path, audio_path, 10, output_file)) 
    output = subprocess.call(command, shell=True, stdout=None)
    cv2.destroyAllWindows()

dataset_path = '/home/xingliu/Datasets/AVADataPath/clips_videos/train/'
audio_path = '/home/xingliu/Datasets/AVADataPath/clips_audios/train/'
out_path = '/home/xingliu/Datasets/AVADataPath/sample_clips'
movie_files = glob.glob(dataset_path + '*')
print(len(movie_files))

training_files = []
for movie_file in movie_files:
    movie_clips = glob.glob(movie_file+'/*')
    training_files += movie_clips

print(len(training_files))

selected = random.sample(training_files, 50)

for clip_file in selected:
    video_name = os.path.dirname(clip_file).split('/')[-1]
    clip_name = os.path.basename(clip_file)
     
    image_files = glob.glob(clip_file+'/*.jpg')
    image = cv2.imread(image_files[0])
    h,w,_ = image.shape

    audio_path = os.path.join(audio_path, video_name, clip_name+'.wav')
    video_path = os.path.join(out_path, clip_name+'_video_only.avi')
    output_file = os.path.join(out_path, clip_name+'.avi')
    make_video(video_path, image_files, audio_path, output_file,h,w)