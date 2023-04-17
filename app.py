import whisper
import os
import pathlib
import numpy as np
from math import inf
import os
from pydub import AudioSegment
import math
import datetime, time
import uvicorn
from fastapi import FastAPI, Request, Response, File, Form, UploadFile
from fastapi.responses import FileResponse

from loguru import logger

'''
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
'''

# split_flac = SplitWavAudioMubin('splits', 'speech.wav')
# splits_amount = split_flac.multiple_split(min_per_split=1)
print('Successfully splitted')


def splitter(audio_file_path: str):
    # print(whisper.available_models())
    initial_time = datetime.datetime.now()
    model = whisper.load_model("base", in_memory=False)
    options = whisper.DecodingOptions(language='uk', without_timestamps=True, fp16=False)
    print(f"Decoding options: {options}")
    transcribed_text = open('text/Interview.txt', 'a')
    for i in range(0, 1):  # splits_amount
        # audio,sr=librosa.load(f'splits/{i}_speech.flac', sr=None, mono=True)
        # audio=whisper.load_audio(f'splits/{i}_speech.flac')
        audio = whisper.load_audio(audio_file_path)
        print(f"loaded file {audio}")

        # _,probs=model.detect_language(mel)
        # print(f"Detected language: {max(probs,key=probs.get)}")
        result = whisper.transcribe(model, audio, language="Ukrainian", without_timestamps=True, verbose="DEBUG")
        print(result['text'])
        # result=whisper.decode(model,mel,options)

        # result = model.transcribe("speech.flac")
        transcribed_text.write('\n' + result['text'])
    transcribed_text.close()
    os.remove(audio_file_path)
    final_time = datetime.datetime.now()
    time_elapsed = final_time - initial_time
    print(time_elapsed, "Seconds")
    filename = 'Interview.txt'
    file_path = 'text/'
    return file_path, filename


app = FastAPI(timeout=inf)


@app.post("/trans")
async def transcribe(file: UploadFile = File(...)):
    """
    Handle incoming webhook requests with file uploads
    """

    contents = await file.read()
    file_name = pathlib.Path(file.filename)
    with open(os.path.join("audios/", file_name), "wb") as f:
        f.write(contents)

    logger.info(file_name)
    # logger.info(audio_file)
    file_path, filename = splitter(f'audios/{file_name}')

    # do something with the file, such as processing it or storing it in a database
    return FileResponse(f'{file_path}/{filename}')


@app.get('/')
async def basic(request: Request, response: Response):
    return {"message": "YA RABOTAYU BLYAT'"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
