
import os,sys
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 새로 추가함
import sqlite3
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



# 0. load key file------------------
def load_key_file ():
    """
    파일 경로에서 GPT API 키를 로드하고 환경 변수에 설정,
    txt파일에서 NAVER MAP API 키 로드

    """
    
    path = './key/'
    
    try:
        # GPT API
        filepath1 = path + 'api_key.txt'
        with open(filepath1, 'r') as file:
            api_key = file.readline().strip()

        # OpenAI 및 환경 변수 설정
        openai.api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key


        # NAVER MAP API
        filepath2 = path + 'map_key.txt'

        with open(filepath2, 'r') as file:
            data = json.loads(file.readline().strip())

        c_id = data['c_id']
        c_key = data['c_key']

        # api 이용에 필요한 값 리턴
        # api_key는 키값 확인용 사용 X
        return api_key, c_id, c_key

    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath1}")
    except Exception as e:
        raise RuntimeError(f"API 키 로드 중 에러 발생: {str(e)}")

# 1-1 audio2text--------------------
def audio2text(filename):
    """
    기본경로, 파일이름을 받아 텍스트로 반환,
    만약, 코드로 음성을 수집할 시 수정필

    """
    path = './audio/'
    
    try:
        # OpenAI 클라이언트 생성
        client = OpenAI()

        # 오디오 파일을 읽어서, 위스퍼를 사용한 변환
        audio_file = open(path+filename, 'rb')

        text = client.audio.transcriptions.create(
            file=audio_file,
            model='whisper-1',
            language='ko',
            response_format='text'
        )

        return text

    except Exception as e:
        print(f"파일을 찾을 수 없습니다.")
        sys.exit(1)

# 1-2 text2summary------------------
def text_summary(input_text):
    """
    input_text에서 핵심을 요약해서 리턴

    """

    client = OpenAI()

    system_role = '''이 상황을 단어로 요약하고 응급실 단어 제외해
    '''

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": input_text}
            ]
        )

        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        return f'요약 실패: {str(e)}'


# 2-1. model prediction------------------
def model_prediction():
    """
    model, tokenizer 리턴
    """
    
    
    try:
        save_directory = './fine_tuned_bert'

        model = AutoModelForSequenceClassification.from_pretrained(save_directory)
        tokenizer = AutoTokenizer.from_pretrained(save_directory)

        return model, tokenizer

    except Exception as e:
      return f'모델 로드 실패: {str(e)}'


# 2-2. 데이터 예측 함수------------------
def predict(text, model, tokenizer):
    # 입력 문장 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value for key, value in inputs.items()}  # 각 텐서를 GPU로 이동

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)

    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities, dim=-1).item()

    return pred, probabilities


# 3-1. get_distance------------------
def get_dist(start_lat, start_lng, dest_lat, dest_lng, c_id, c_key):
    """
    사용자의 시작지점 (위도, 경도), 목적지점(위도, 경도)
    NAVER MAP 관련 키 아이디, 키 값

    """

    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": c_id,
        "X-NCP-APIGW-API-KEY": c_key,
    }
    params = {
        "start": f"{start_lng},{start_lat}",  # 출발지 (경도, 위도)
        "goal": f"{dest_lng},{dest_lat}",    # 목적지 (경도, 위도)
        "option": "trafast"  # 실시간 빠른 길 옵션
    }

    # 요청하고, 답변 받아오기
    response = requests.get(url, headers=headers, params=params)

    # JSON 응답 파싱
    if response.status_code == 200:
        data = response.json()
        try:
            # 데이터에서 거리 추출
            dist = data['route']['trafast'][0]['summary']['distance']  # m(미터)
            return dist
        except KeyError as e:
            print(f"KeyError: Missing key {e}")
            return None
        except ValueError as e:
            print(f"ValueError: {e}")
            return None
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None


# 3-2. recommendation------------------
def recommendation(start_lat, start_lng, df, c_id, c_key, predicted_class, output_count):
    """
    사용자의 시작지점 (위도, 경도),

    df는 응급실 정보 데이터프레임,

    데이터프레임에 따라 df['위도'], df['경도'] 등 컬럼명 변경필요

    >> df = pd.read_csv(path+'응급실 정보.csv')

    get_dist를 위한 NAVER MAP 관련 키 아이디, 키 값

    웹 서비스를 위해서 df2 리턴

    """

    my_location = (start_lat, start_lng)
    limit = 0.005        # 범위 지정, 위도, 경도 각 각 0.005씩 +- 해서 지정
    emergency_count = 0  # 응급실 개수
    # output_count = 3     # 출력할 개수

    while emergency_count < output_count:
        # 현재 limit 값으로 범위 내 병원들 선택
        df2 = df.loc[ ((start_lat - limit <= df['위도']) & (df['위도'] <= start_lat + limit)) &
                      ((start_lng - limit <= df['경도']) & (df['경도'] <= start_lng + limit)) ]

        # predicted_class에 따른 응급의료기관 종류 필터링
        if predicted_class in [0]:
            df2 = df2[df2['응급의료기관 종류'] == '권역응급의료센터']
        elif predicted_class in [1]:
            df2 = df2[df2['응급의료기관 종류'].isin(['권역응급의료센터', '지역응급의료센터'])]
        elif predicted_class in [2]:
            df2 = df2[df2['응급의료기관 종류'] == '지역응급의료센터']
        else:
            df2 = df2[df2['응급의료기관 종류'].isin(['지역응급의료기관', '응급실운영신고기관'])]

        emergency_count = len(df2)  # 범위 내 병원 개수 계산

        # 병원 개수가 3개 미만일 경우 범위 늘리기
        if emergency_count < output_count:
            limit += 0.005  # 범위를 0.1씩 증가시킴

    dsts = []

    for index, row in df2.iterrows():
        dst = get_dist(start_lat, start_lng, row['위도'], row['경도'], c_id, c_key)
        dsts.append(dst)

    df2.loc[:, '거리'] = dsts
    df2.sort_values(by='거리', inplace=True)
    df2.reset_index(drop=True, inplace=True)

    return df2





# 새로 추가한 함수

# 테이블 생성함수
def create_table():
    path = './db/'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    path += 'em.db'
    conn = sqlite3.connect(path)

    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datetime TEXT NOT NULL,
        input_text TEXT NOT NULL,
        input_latitude REAL NOT NULL,
        input_longitude REAL NOT NULL,
        em_class INTEGER NOT NULL,
        hospital1 TEXT,
        addr1 TEXT,
        tel1 TEXT,
        hospital2 TEXT,
        addr2 TEXT,
        tel2 TEXT,
        hospital3 TEXT,
        addr3 TEXT,
        tel3 TEXT
    )
    ''')

    conn.commit()
    conn.close()
    
    
    
    
def insert_data(answer, start_lat, start_lng, grade, new_df):
    dt = datetime.now()
    dt = dt.strftime('%Y-%m-%d %H:%M:%S')

    path = './db/em.db'
    conn = sqlite3.connect(path)

    selected_data = pd.DataFrame({
        'datetime':dt,
        'input_text': answer, # 이건 answer
        'input_latitude': start_lat, # 위도 경도 아까 그걸로 대체체
        'input_longitude': start_lng,
        'em_class': grade,
    
        'hospital1': [new_df['병원이름'].iloc[0]],  # 0번 행의 병원이름
        'addr1': [new_df['주소'].iloc[0]],          # 0번 행의 주소
        'tel1': [new_df['전화번호 1'].iloc[0]],     # 0번 행의 전화번호1
        'hospital2': [new_df['병원이름'].iloc[1]],  # 1번 행의 병원이름
        'addr2': [new_df['주소'].iloc[1]],          # 1번 행의 주소
        'tel2': [new_df['전화번호 1'].iloc[1]],     # 1번 행의 전화번호1
        'hospital3': [new_df['병원이름'].iloc[2]],  # 2번 행의 병원이름
        'addr3': [new_df['주소'].iloc[2]],          # 2번 행의 주소
        'tel3': [new_df['전화번호 1'].iloc[2]]      # 2번 행의 전화번호1
    })


    selected_data.to_sql('log', conn, if_exists='append', index=False)
    conn.close()



def find_df():
    path = './db/em.db'
    conn = sqlite3.connect(path)

    df = pd.read_sql('SELECT * FROM log', conn)
    # display(df)

    conn.close()
    
    return df

# 응급실정보 DB 읽기 함수
def get_emergency_data():
    path = './db/emergency_info.db'

    try:
        # 데이터베이스 연결
        conn = sqlite3.connect(path)
        
        # 데이터 조회
        query = "SELECT * FROM emergency_info"
        df = pd.read_sql(query, conn)
        
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()
