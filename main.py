import os
import io
import base64
import time
import uuid
import logging
import asyncio
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import replicate
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

# 간소화된 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("image-api")

app = FastAPI(title="이미지 통합 API", description="사람과 동물 이미지로 자연스러운 상호작용 장면 생성")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 설정
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN 환경 변수를 설정해주세요.")

# 이미지 생성 모델 설정
SCENE_GENERATION_MODEL = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"

# 이미지 크기 제한
MAX_IMAGE_SIZE = 1024

# Replicate 클라이언트 설정
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# 예측 작업 저장소
predictions = {}

# 리액트 클라이언트용 모델
class PredictionInput(BaseModel):
    input: Dict[str, str]

class PredictionCreate(BaseModel):
    id: str
    status: str
    created_at: str

class PredictionStatus(BaseModel):
    id: str
    status: str
    output: Optional[List[str]] = None
    error: Optional[str] = None

def resize_image_if_needed(img: Image.Image, max_size: int) -> Image.Image:
    """이미지 크기가 제한을 초과하면 비율을 유지하며 리사이징합니다."""
    width, height = img.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img

@app.post("/predictions", response_model=PredictionCreate)
async def create_prediction(
    background_tasks: BackgroundTasks,
    prediction_input: PredictionInput
):
    """예측 작업을 생성합니다."""
    try:
        # Base64 이미지 디코딩
        person_image_base64 = prediction_input.input.get("person_image")
        pet_image_base64 = prediction_input.input.get("pet_image")
        scene_type = prediction_input.input.get("scene_type", "walking together in a park")
        
        if not person_image_base64 or not pet_image_base64:
            raise HTTPException(status_code=400, detail="사람과 동물 이미지가 모두 필요합니다")
        
        # 이미지 로드
        try:
            # Base64 데이터에 prefix가 없는 경우 추가
            if not person_image_base64.startswith('data:'):
                person_image_base64 = f"data:image/jpeg;base64,{person_image_base64}"
            if not pet_image_base64.startswith('data:'):
                pet_image_base64 = f"data:image/jpeg;base64,{pet_image_base64}"
                
            person_img = decode_base64_to_image(person_image_base64)
            pet_img = decode_base64_to_image(pet_image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {str(e)}")
        
        # 예측 ID 생성
        prediction_id = str(uuid.uuid4())
        current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        # 초기 상태 저장
        predictions[prediction_id] = {
            "id": prediction_id,
            "status": "starting",
            "created_at": current_time,
            "output": None,
            "error": None
        }
        
        # 백그라운드에서 이미지 처리
        background_tasks.add_task(
            process_images_in_background,
            prediction_id=prediction_id,
            person_img=person_img,
            pet_img=pet_img,
            scene_type=scene_type
        )
        
        return PredictionCreate(
            id=prediction_id,
            status="starting",
            created_at=current_time
        )
        
    except Exception as e:
        logger.error(f"예측 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"예측 생성 오류: {str(e)}")

@app.get("/predictions/{prediction_id}", response_model=PredictionStatus)
async def get_prediction_status(prediction_id: str):
    """예측 상태를 확인합니다."""
    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="예측 작업을 찾을 수 없습니다")
    return PredictionStatus(**predictions[prediction_id])

@app.get("/predictions", response_model=Dict[str, PredictionStatus])
async def get_all_predictions():
    """모든 예측 작업의 상태를 반환합니다."""
    return {pid: PredictionStatus(**pred) for pid, pred in predictions.items()}

async def process_images_in_background(
    prediction_id: str, 
    person_img: Image.Image, 
    pet_img: Image.Image,
    scene_type: str
):
    """백그라운드에서 이미지 처리 및 통합 장면 생성"""
    try:
        # 상태 업데이트
        predictions[prediction_id]["status"] = "processing"
        
        # 이미지 크기 제한
        person_img = resize_image_if_needed(person_img, MAX_IMAGE_SIZE)
        pet_img = resize_image_if_needed(pet_img, MAX_IMAGE_SIZE)
        
        # 이미지를 임시 저장하고 처리
        person_features = extract_person_features(person_img)
        pet_features = extract_pet_features(pet_img)
        
        # 장면 생성 프롬프트 구성
        prompt = create_scene_prompt(person_features, pet_features, scene_type)
        
        # 장면 생성 모델 실행
        scene_output = await run_scene_generation(prompt, prediction_id)
        
        if not scene_output or not isinstance(scene_output, list) or len(scene_output) == 0:
            raise Exception("장면 생성 모델이 유효한 결과를 반환하지 않았습니다")
        
        # 결과 이미지 URL 처리
        result_url = scene_output[0]
        result_image = await download_image(result_url)
        
        if not result_image:
            raise Exception("결과 이미지를 다운로드할 수 없습니다")
            
        # 결과 이미지 인코딩
        encoded_result = encode_image_to_base64(result_image)
        
        # 작업 완료
        predictions[prediction_id]["status"] = "succeeded"
        predictions[prediction_id]["output"] = [encoded_result]
        
    except Exception as e:
        logger.error(f"[{prediction_id}] 처리 오류: {str(e)}")
        predictions[prediction_id]["status"] = "failed"
        predictions[prediction_id]["error"] = str(e)

def extract_person_features(img: Image.Image) -> Dict[str, Any]:
    """사람 이미지에서 주요 특징을 추출합니다."""
    # 실제 구현에서는 얼굴 인식이나 특징 추출 라이브러리 사용 가능
    # 여기서는 간단히 이미지 특성만 반환
    return {
        "estimated_gender": "person",
        "estimated_age": "adult",
        "hair_color": "natural",
        "clothing": "casual"
    }

def extract_pet_features(img: Image.Image) -> Dict[str, Any]:
    """동물 이미지에서 주요 특징을 추출합니다."""
    # 실제 구현에서는 동물 종 인식 등의 기능 추가 가능
    # 여기서는 간단히 기본값 반환
    return {
        "type": "pet",
        "color": "natural"
    }

def create_scene_prompt(person_features: Dict, pet_features: Dict, scene_type: str) -> str:
    """장면 생성 프롬프트를 구성합니다."""
    # 기본 장면 유형
    scene_settings = {
        "walking together": "a person and their pet walking together in a beautiful park, sunny day",
        "sitting on couch": "a person and their pet sitting together on a comfortable couch, home setting",
        "playing": "a person playing with their pet in a backyard",
        "beach": "a person and their pet enjoying time at a beach"
    }
    
    # 사용자 지정 장면 또는 기본 장면 설정
    base_scene = scene_settings.get(scene_type, f"a person and their pet {scene_type}")
    
    # 상세한 프롬프트 구성
    prompt = f"{base_scene}, natural interaction, maintaining original appearances, photorealistic, " \
             f"high quality, detailed, 8k resolution, natural lighting, " \
             f"not merged or morphed together, side by side"
    
    # 부정적 프롬프트 추가 (모델에 따라 다를 수 있음)
    negative_prompt = "distorted, blurry, low quality, unnatural poses, morphed faces, merged bodies"
    
    return prompt

async def run_scene_generation(prompt: str, prediction_id: str) -> List[str]:
    """장면 생성 모델을 비동기적으로 실행합니다."""
    try:
        logger.info(f"[{prediction_id}] 장면 생성 시작: {prompt[:50]}...")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: replicate_client.run(
                SCENE_GENERATION_MODEL,
                input={
                    "prompt": prompt,
                    "negative_prompt": "distorted, blurry, low quality, unnatural, artificial, merged bodies, merged faces",
                    "width": 1024,
                    "height": 768,
                    "num_outputs": 1,
                    "num_inference_steps": 40,
                    "guidance_scale": 7.5,
                    "scheduler": "K_EULER_ANCESTRAL"
                }
            )
        )
    except Exception as e:
        logger.error(f"[{prediction_id}] 장면 생성 오류: {str(e)}")
        raise

async def download_image(url: str) -> Optional[Image.Image]:
    """URL에서 이미지를 다운로드합니다."""
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(url, timeout=30)
        )
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        return None
    except Exception as e:
        logger.error(f"이미지 다운로드 오류: {str(e)}")
        return None

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Base64 문자열을 PIL 이미지로 디코딩합니다."""
    try:
        # data:image/png;base64, 접두사 제거
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        img_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(img_data))
        return image
    except Exception as e:
        logger.error(f"Base64 디코딩 오류: {str(e)}")
        raise

def encode_image_to_base64(image: Image.Image) -> str:
    """PIL 이미지를 Base64 문자열(data URL)로 인코딩합니다."""
    try:
        buffer = io.BytesIO()
        # RGBA 이미지를 RGB로 변환 (투명도 제거)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format="JPEG", quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Base64 인코딩 오류: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    logger.info("서버 시작")
    uvicorn.run(app, host="0.0.0.0", port=5050)