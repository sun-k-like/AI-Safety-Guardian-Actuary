import os
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 설정 로드
CONFIG = {
    "VISION_ENDPOINT": os.getenv("VISION_ENDPOINT"),
    "VISION_KEY": os.getenv("VISION_KEY"),
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT")
}

# 클라이언트 초기화
vision_client = ImageAnalysisClient(endpoint=CONFIG["VISION_ENDPOINT"], credential=AzureKeyCredential(CONFIG["VISION_KEY"]))
llm_client = AzureOpenAI(api_key=CONFIG["AZURE_OPENAI_API_KEY"], api_version="2024-12-01-preview", azure_endpoint=CONFIG["AZURE_OPENAI_ENDPOINT"])

@app.post("/analyze_full_integrated")
async def analyze_full_integrated(
    site_img: UploadFile = File(...),      # 현장 사진
    insur_doc: UploadFile = File(...),     # 산재보험 증명원
    machine_doc: UploadFile = File(...)    # 프레스 안전점검표
):
    try:
        # 1. 모든 파일 데이터 읽기
        site_content = await site_img.read()
        insur_content = await insur_doc.read()
        machine_content = await machine_doc.read()

        # 2. Azure AI Vision 분석
        # 현장 사진: 캡션 및 태그 분석
        site_result = vision_client.analyze(image_data=site_content, visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS])
        
        # 보험 서류 & 기계 점검표: OCR 텍스트 추출
        insur_ocr = vision_client.analyze(image_data=insur_content, visual_features=[VisualFeatures.READ])
        machine_ocr = vision_client.analyze(image_data=machine_content, visual_features=[VisualFeatures.READ])
        
        insur_text = " ".join([line.text for block in insur_ocr.read.blocks for line in block.lines])
        machine_text = " ".join([line.text for block in machine_ocr.read.blocks for line in block.lines])

        # 3. LLM 통합 분석 프롬프트 (현장 + 보험 + 기계)
        prompt = f"""
        당신은 산업안전 및 보험 계리 전문가입니다. 제공된 세 가지 데이터를 통합 분석하여 최종 보고서를 작성하세요.
        **기준 날짜: 2026-02-24**

        [1. 현장 데이터]
        - 상황: {site_result.caption.text}
        - 감지된 요소: {', '.join([tag.name for tag in site_result.tags.list])}

        [2. 산재보험 서류 텍스트]
        {insur_text}

        [3. 프레스 안전점검표 텍스트]
        {machine_text}

        [엄격한 분석 및 계산 규칙]
        1. 현장 위험도(S_Risk): 0~100점 (현장 상황 분석 기반, 위험할수록 높음)
        2. 보험 신뢰도(I_Trust): 0~100점 (가입여부, 납부상태 분석 기반)
        3. 기계 안전 점수(M_Safety): 
           - 최종점검일로부터 6개월(180일) 초과 시 무조건 0점.
           - 기한 내라도 'X' 또는 '불량' 항목 1개당 30점 감점.
        4. 날짜 산출: 프레스의 '차기 점검 예정일(최종+6개월)'과 '보험료 인상 예고일(차기-1개월)' 명시.
        5. 최종 할인율: 7.0% * (I_Trust/100) * (M_Safety/100) * (1 - S_Risk/100)

        [JSON 응답 포맷]
        {{
            "site_analysis": {{
                "risk_score": 0,
                "description": "현장 위험 분석 내용"
            }},
            "insurance_analysis": {{
                "trust_score": 0,
                "status": "보험 상태 요약"
            }},
            "machine_report": {{
                "safety_score": 0,
                "last_inspection": "YYYY-MM-DD",
                "next_inspection": "YYYY-MM-DD",
                "hike_warning_date": "YYYY-MM-DD",
                "status_summary": "기한 준수 및 불량 여부"
            }},
            "final_impact": {{
                "final_discount_rate": "0.00%",
                "calculation_logic": "계산 과정 설명",
                "total_improvement": "현장/보험/기계 전체에 대한 조치 제안"
            }}
        }}
        """

        response = llm_client.chat.completions.create(
            model=CONFIG["AZURE_OPENAI_DEPLOYMENT"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))