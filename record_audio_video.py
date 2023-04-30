import cv2
import os
import sounddevice as sd
import soundfile as sf

cap = cv2.VideoCapture(2)
stream_in = sd.InputStream(
    device=40,
    samplerate=16000,
    channels=5)
stream_in.start()

iframe = 0 
length = 16000
# video_name = '/temp/videosvideo_only.avi'

audio_5s = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    os.makedirs('temp/Original_frames', exist_ok=True)
    cv2.imwrite(f'temp/Original_frames/{iframe}.jpg', frame)
    print('save frame: ', iframe)
    original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    audio_frame, overflow = stream_in.read(length)
    # if iframe%1 == 0:
    #     if len(audio_5s) > 0:
    #         print(len(audio_5s))
    sf.write(f'temp/Original_frames/{iframe}.wav', audio_frame, 16000)
    #         print('save audio')
    #         audio_5s = []
    # audio_5s.append(audio_frame)
    
    iframe += 1