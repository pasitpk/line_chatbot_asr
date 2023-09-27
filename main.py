import pickle
import json
import io
import torch
import pydub
import numpy as np
from transformers import pipeline, WhisperForConditionalGeneration,WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

from typing import List, Optional

from fastapi import HTTPException, Header, Request, FastAPI, File, UploadFile, Query
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import AudioMessage, MessageEvent, TextSendMessage
from pydantic import BaseModel

from corrector import DeepGICorrector

with open('.json', 'r') as f:
    config = json.load(f)

LINE_BOT_API = LineBotApi(config['LINE_CHANNEL_ACCESS_TOKEN'])
HANDLER = WebhookHandler(config['LINE_CHANNEL_SECRET'])
ASR_PIPE = config['ASR_PIPE']
MAX_TOKENS = config['MAX_TOKENS']
CORRECTION = config['CORRECTION']
MAX_LOG_SIZE = config['MAX_LOG_SIZE']
transcript_log = []

with open(config['CUSTOM_DICT'], 'rb') as f:
    custom_dict = pickle.load(f)

corrector = DeepGICorrector(custom_dict, lower_case=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ORTModelForSpeechSeq2Seq.from_pretrained(ASR_PIPE,use_io_binding=True,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
processor = WhisperProcessor.from_pretrained(ASR_PIPE) 
pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    chunk_length_s=30,
    max_new_tokens=MAX_TOKENS,
    device=device,
)

app = FastAPI()

class Line(BaseModel):
    destination: str
    events: List[Optional[None]]


@app.get("/")
async def root():
    return {"message": "Hello, DeepGI"}


@app.get("/transcripts/")
async def get_transcripts(n: int = Query(ge=1)):
    return {"transcripts": transcript_log[-n:]}


@app.post("/transcribe/")
async def transcribe_file(file: UploadFile = File(...)):
    file.file.seek = lambda *args: None
    audio = pydub.AudioSegment.from_file(file.file)
    text = transcribe(audio)
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
    audio = get_audio(message_content)
    text = transcribe(audio)
    LINE_BOT_API.reply_message(
        event.reply_token,
        TextSendMessage(text=text)
    )


def transcribe(audio):
    text = pipe(audio.export(format='wav').read())['text']
    if CORRECTION:
        text = corrector.correct(text)
    transcript_log.append(text)
    transcript_log[:] = transcript_log[-MAX_LOG_SIZE:]
    return text


def get_audio(message_content):
    with io.BytesIO() as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)
        fd.seek(0)
        audio = pydub.AudioSegment.from_file(fd)
    return audio
    
