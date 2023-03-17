import whisper
import numpy as np
import math
import os
from pydub import AudioSegment
import math
import datetime, time


class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="flac")

    def multiple_split(self, min_per_split):
        counter = 0
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename.split(".")[0] + '.flac'
            self.single_split(i, i + min_per_split, split_fn)
            print(str(i) + ' Done')
            counter = counter + 1
            if i == total_mins - min_per_split:
                print('All splited successfully')
                return counter + 1


inital_time = datetime.datetime.now()
#split_flac = SplitWavAudioMubin('splits', 'speech.wav')
#splits_amount = split_flac.multiple_split(min_per_split=1)
print('Successfully splitted')
#print(whisper.available_models())
model = whisper.load_model("large", in_memory=False)
options=whisper.DecodingOptions(language='uk',without_timestamps=True,fp16=False)
print(f"Decoding options: {options}")
file_object = open('Interview.txt', 'a')
for i in range(0, 1):  # splits_amount
    # audio,sr=librosa.load(f'splits/{i}_speech.flac', sr=None, mono=True)
    #audio=whisper.load_audio(f'splits/{i}_speech.flac')
    audio = whisper.load_audio('speech.flac')
    print(f"loaded file {audio}")

    # _,probs=model.detect_language(mel)
    # print(f"Detected language: {max(probs,key=probs.get)}")
    result = whisper.transcribe(model, audio, language="Ukrainian", without_timestamps=True, verbose="DEBUG")
    print(result['text'])
    # result=whisper.decode(model,mel,options)

    # result = model.transcribe("speech.flac")
    file_object.write('\n' + result['text'])
file_object.close()

final_time = datetime.datetime.now()
time_elapsed = final_time - inital_time
print(time_elapsed, "Seconds")
