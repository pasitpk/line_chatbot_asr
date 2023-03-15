import json
import io
import torch
import pydub
import numpy as np
from transformers import pipeline

from typing import List, Optional

from fastapi import HTTPException, Header, Request, FastAPI
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import AudioMessage, MessageEvent, TextSendMessage
from pydantic import BaseModel

with open('.json', 'r') as f:
    config = json.load(f)

line_bot_api = LineBotApi(config['LINE_CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(config['LINE_CHANNEL_SECRET'])
asr_pipe = config['ASR_PIPE']

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline("automatic-speech-recognition",
                    model=asr_pipe,
                    max_new_tokens=50,
                    device=device,
                    )

app = FastAPI()

class Line(BaseModel):
    destination: str
    events: List[Optional[None]]


@app.get("/")
async def root():
    return {"message": "Hello, DeepGI"}


@app.post("/linehook")
async def callback(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="chatbot handle body error.")
    return 'OK'


@handler.add(MessageEvent, message=AudioMessage)
def message_text(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    text = transcribe(message_content)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=text)
    )


def transcribe(message_content):
    audio, sampling_rate = get_audio(message_content)
    audio = audio.mean(1)
    text = pipe({'raw': audio, 'sampling_rate': sampling_rate})['text']
    return text


def get_audio(message_content):
    with io.BytesIO() as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)
        fd.seek(0)
        audio = pydub.AudioSegment.from_file(fd)
    audio, sampling_rate = pydub_to_np(audio)
    return audio, sampling_rate
    

def pydub_to_np(audio):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate
