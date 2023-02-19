import pyaudio
import wave
 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 5
device_index = 2

audio = pyaudio.PyAudio()
    
# Get audio device info
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
print('numdevices: ', numdevices)

device_dict = {}
for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
            device_dict[i] = audio.get_device_info_by_host_api_device_index(0, i).get('name')

print(device_dict)
index = int(input())
print("recording via index "+str(index))

stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,input_device_index = index,
                frames_per_buffer=CHUNK)
print ("recording started")
Recordframes = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    Recordframes.append(data)
print ("recording stopped")
 
stream.stop_stream()
stream.close()
audio.terminate()

WAVE_OUTPUT_FILENAME = f"test_pc_audio/Poly_maxhub_plugged_Device_id_{index}_{device_dict[index]}.wav"

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(Recordframes))
waveFile.close()