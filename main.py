import pickle
import json
import io
import torch
import pydub
import numpy as np
from transformers import pipeline

from typing import List, Optional

from fastapi import HTTPException, Header, Request, FastAPI, File, UploadFile
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import AudioMessage, MessageEvent, TextSendMessage
from pydantic import BaseModel

from corrector import DeepGICorrector

with open('.json', 'r') as f:
    config = json.load(f)

LINE_BOT_API = LineBotApi(config['LINE_CHANNEL_ACCESS_TOKEN'])
HANDLER = WebhookHandler(config['LINE_CHANNEL_SECRET'])
ASR_PIPE = config['ASR_PIPE']  # for Line Messaging API
ASR_PIPE2 = config['ASR_PIPE2']  # for regular requests
CORRECTION = config['CORRECTION']

with open(config['CUSTOM_DICT'], 'rb') as f:
    custom_dict = pickle.load(f)

corrector = DeepGICorrector(custom_dict, lower_case=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline("automatic-speech-recognition",
                    model=ASR_PIPE,
                    max_new_tokens=50,
                    device=device,
                    )

pipe2 = pipeline("automatic-speech-recognition",
                    model=ASR_PIPE2,
                    device=device,
                    )

app = FastAPI()

class Line(BaseModel):
    destination: str
    events: List[Optional[None]]


@app.get("/")
async def root():
    return {"message": "Hello, DeepGI"}


@app.post("/transcribe/")
async def transcribe_file(file: UploadFile = File(...)):
    file.file.seek = lambda *args: None
    audio = pydub.AudioSegment.from_file(file.file)
    text = pipe2(audio.export(format='wav').read())['text']
    if CORRECTION:
        text = corrector.correct(text)
    res = {
        'text': text
    }
    return res


@app.post("/linehook")
async def callback(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    try:
        HANDLER.handle(body.decode("utf-8"), x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="chatbot handle body error.")
    return 'OK'


@HANDLER.add(MessageEvent, message=AudioMessage)
def message_text(event):
    message_content = LINE_BOT_API.get_message_content(event.message.id)
    text = transcribe(message_content)
    LINE_BOT_API.reply_message(
        event.reply_token,
        TextSendMessage(text=text)
    )


def transcribe(message_content):
    audio = get_audio(message_content)
    text = pipe(audio.export(format='wav').read())['text']
    if CORRECTION:
        text = corrector.correct(text)
    return text


def get_audio(message_content):
    with io.BytesIO() as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)
        fd.seek(0)
        audio = pydub.AudioSegment.from_file(fd)
    return audio
    
