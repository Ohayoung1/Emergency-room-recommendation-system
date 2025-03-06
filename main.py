import os, sys
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from warnings import filterwarnings
FutureWarning
filterwarnings('ignore')
import sqlite3
from datetime import datetime
from emergency import load_key_file, audio2text, text_summary, model_prediction, predict, get_dist, recommendation, create_table, insert_data, find_df, get_emergency_data
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel


class Data(BaseModel):
    # filename: str  # 오디오 파일명
    text: str  # 텍스트, 프론트로 음성대신 입력할때
    start_lat: float  # 시작 위도
    start_lng: float  # 시작 경도
    output_count: int # 출력할 개수



app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/hello")
def read_root2():
    return {"Hello2": "World2"}

@app.post('/api/hello')
def process(request: Data):
    # filename = request.filename
    start_lat = request.start_lat
    start_lng = request.start_lng
    text = request.text
    output_count = request.output_count
    tmp = output_count
    
    if (output_count < 3):
        output_count = 3
    
    # api_key, c_id, c_key = load_key_file()

    
    c_id = os.environ.get('NAVER_ID')  # 환경변수에서 Client ID 가져오기
    c_key = os.environ.get('NAVER_KEY')  # 환경변수에서 Client Key 가져오기

    
    # text = audio2text(filename) ## 클라이언트 요청 변수

    answer = text_summary(text)
    model, tokenizer = model_prediction()
    predicted_class, probabilities = predict(answer, model, tokenizer)

    grade = predicted_class + 1

    df = get_emergency_data()
    new_df = recommendation(start_lat, start_lng, df, c_id, c_key, predicted_class, output_count)
    new_df = new_df.head(output_count)

    create_table()
    insert_data(answer, start_lat, start_lng, grade, new_df)
    find_df()
    
    new_df = new_df.head(tmp)
    df_json = new_df.to_dict(orient="records")

    return {
        "text": text,
        "grade": grade,
        "start_lat": start_lat,
        "start_lng": start_lng,
        "emergency_data": df_json
    }

