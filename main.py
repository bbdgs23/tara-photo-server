from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Optional, List
import os
import base64
import logging
import asyncio
import json
import httpx
import uuid
from PIL import Image, ImageDraw, ImageFilter
import io
import replicate
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 정의 - 모든 클래스를 app 정의 전에 위치시킵니다
class PredictionRequest(BaseModel):
    input: Dict[str, str]
    scene_type: Optional[str] = "park"  # 기본값: "park", 선택사항: "sofa"

class PredictionResponse(BaseModel):
    id: str
    status: str = "processing"

# FastAPI 앱 인스턴스 생성 - 모델 정의 후에 위치시킵니다
app = FastAPI(title="반려동물-사람 합성 API")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 필요한 디렉토리 생성
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# 상태 데이터 저장
prediction_data: Dict[str, Dict] = {}

# Replicate API 키 설정
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    logger.warning("REPLICATE_API_TOKEN 환경변수가 설정되지 않았습니다.")
replicate.api_token = REPLICATE_API_TOKEN

# 모델 ID
SAM_MODEL = "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"
GROUNDING_DINO_MODEL = "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa"
SD_INPAINT_MODEL = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"

# 캐시 설정 - 모델 Cold Start 방지
model_cache = {}
last_prediction_time = {}

def update_prediction(prediction_id: str, status: str, progress: int = 0, output: Optional[list] = None, error: Optional[str] = None):
    """예측 상태 업데이트"""
    if prediction_id not in prediction_data:
        prediction_data[prediction_id] = {}
        
    prediction_data[prediction_id].update({
        "status": status,
        "progress": progress
    })
    
    if output is not None:
        prediction_data[prediction_id]["output"] = output
        
    if error is not None:
        prediction_data[prediction_id]["error"] = error
        
    logger.info(f"Prediction {prediction_id} updated: {status} - {progress}%")

async def base64_to_image(base64_string: str) -> Image.Image:
    """Base64 문자열을 PIL Image로 변환"""
    try:
        # Base64 디코딩
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Base64 이미지 변환 오류: {str(e)}")
        raise ValueError(f"잘못된 Base64 형식: {str(e)}")

async def base64_to_url(base64_string: str, file_prefix: str, prediction_id: str) -> str:
    """Base64 문자열을 임시 파일로 저장하고 URL 반환"""
    try:
        # Base64 디코딩
        image_data = base64.b64decode(base64_string)
        
        # 이미지 파일로 저장
        file_path = f"temp/{file_prefix}_{prediction_id}.jpg"
        with open(file_path, "wb") as f:
            f.write(image_data)
            
        # 로컬 파일 경로 반환
        return file_path
    except Exception as e:
        logger.error(f"Base64 이미지 변환 오류: {str(e)}")
        raise ValueError(f"잘못된 Base64 형식: {str(e)}")

def file_to_data_uri(file_path: str, mime_type: str = "image/jpeg") -> str:
    """파일을 데이터 URI로 변환"""
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            base64_encoded = base64.b64encode(file_bytes).decode('utf-8')
            return f"data:{mime_type};base64,{base64_encoded}"
    except Exception as e:
        logger.error(f"파일을 데이터 URI로 변환 중 오류: {str(e)}")
        raise ValueError(f"파일 변환 실패: {str(e)}")

async def http_file_to_url(file_url: str, file_prefix: str, prediction_id: str) -> str:
    """HTTP URL의 파일을 다운로드하여 로컬에 저장하고 경로 반환"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"파일 다운로드 실패: {response.status_code}")
            
            file_path = f"temp/{file_prefix}_{prediction_id}.jpg"
            with open(file_path, "wb") as f:
                f.write(response.content)
                
            return file_path
    except Exception as e:
        logger.error(f"파일 다운로드 오류: {str(e)}")
        raise ValueError(f"파일 다운로드 실패: {str(e)}")

async def run_replicate_async(model_id: str, input_data: Dict) -> Dict:
    """Replicate API 비동기 호출 - 캐싱 개선"""
    current_time = asyncio.get_event_loop().time()
    model_name = model_id.split(':')[0]
    
    # 캐시 워밍업 확인 및 유지
    if model_name in last_prediction_time:
        time_since_last_call = current_time - last_prediction_time[model_name]
        # 15분 이내에 호출이 없었으면 워밍업 필요 (Replicate 모델은 보통 10-15분 후 cold)
        if time_since_last_call > 900:  # 15분 = 900초
            logger.info(f"{model_name} 모델 re-warming 필요 (마지막 호출 후 {time_since_last_call:.1f}초 경과)")
    
    # 현재 시간 업데이트
    last_prediction_time[model_name] = current_time
    
    try:
        # 로컬 파일 경로를 데이터 URI로 변환
        if "image" in input_data and isinstance(input_data["image"], str) and input_data["image"].startswith("temp/"):
            input_data["image"] = file_to_data_uri(input_data["image"])
        
        if "mask" in input_data and isinstance(input_data["mask"], str) and input_data["mask"].startswith("temp/"):
            input_data["mask"] = file_to_data_uri(input_data["mask"], "image/png")
        
        # httpx를 사용한 비동기 API 호출
        headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        model_version = model_id.split(':')[1]
        
        logger.info(f"Replicate API 호출: {model_name}")
        
        async with httpx.AsyncClient(timeout=180.0) as client:  # 타임아웃 최적화
            # 예측 생성 요청
            response = await client.post(
                f"https://api.replicate.com/v1/predictions",
                headers=headers,
                json={
                    "version": model_version,
                    "input": input_data
                }
            )
            
            if response.status_code != 201:
                logger.error(f"Replicate API 오류: {response.text}")
                raise HTTPException(status_code=response.status_code, 
                                   detail=f"Replicate API 오류: {response.text}")
            
            prediction = response.json()
            prediction_id = prediction["id"]
            logger.info(f"Replicate 예측 생성됨: {prediction_id}")
            
            # 결과가 준비될 때까지 폴링 - 더 효율적인 폴링
            poll_count = 0
            max_polls = 60  # 최대 폴링 횟수
            
            while poll_count < max_polls:
                # 단계적 지연 - 처음엔 짧게, 나중엔 길게
                await asyncio.sleep(1.0 if poll_count < 5 else 2.0)
                poll_count += 1
                
                response = await client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers=headers
                )
                prediction = response.json()
                
                if prediction["status"] == "succeeded":
                    logger.info(f"Replicate 예측 성공: {prediction_id}")
                    return prediction["output"]
                elif prediction["status"] == "failed":
                    logger.error(f"Replicate 예측 실패: {prediction.get('error', '')}")
                    raise HTTPException(status_code=500, 
                                       detail=f"Replicate 모델 실행 실패: {prediction.get('error', '')}")
                
            # 최대 폴링 횟수 초과
            raise HTTPException(status_code=504, detail="Replicate API 응답 시간 초과")
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Replicate API 타임아웃")
    except Exception as e:
        logger.error(f"Replicate API 호출 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Replicate API 오류: {str(e)}")

# 합성 이미지 준비 부분 수정 (process_images 함수 내부)
async def process_images(prediction_id: str, person_base64: str, pet_base64: str, scene_type: str = "park"):
    """이미지 처리 및 합성 작업 실행 - 원본 유지 최적화"""
    try:
        # 초기 상태 업데이트
        update_prediction(prediction_id, "starting", 10)
        
        # 1. 이미지를 임시 파일로 저장하고 경로 반환
        person_path = await base64_to_url(person_base64, "person", prediction_id)
        pet_path = await base64_to_url(pet_base64, "pet", prediction_id)
        
        # 2. 데이터 URI 생성
        person_data_uri = file_to_data_uri(person_path)
        pet_data_uri = file_to_data_uri(pet_path)
        
        # 3. GroundingDINO로 사람 객체 정밀 감지 (임계값 최적화)
        update_prediction(prediction_id, "detecting_person", 20)
        person_detection = await run_replicate_async(
            GROUNDING_DINO_MODEL,
            {
                "image": person_data_uri,
                "query": "person, human, people",
                "box_threshold": 0.18,  # 최적화된 임계값
                "text_threshold": 0.18   # 최적화된 임계값
            }
        )
        
        if not person_detection or not person_detection.get("detections"):
            raise ValueError("사람을 찾을 수 없습니다. 다른 사진을 시도해주세요.")
        
        # 가장 큰 사람 바운딩 박스 선택
        person_detections = person_detection["detections"]
        person_detection = max(person_detections, key=lambda d: 
                              (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        person_bbox = person_detection["bbox"]
        
        # 4. GroundingDINO로 반려동물 객체 정밀 감지
        update_prediction(prediction_id, "detecting_pet", 30)
        pet_detection = await run_replicate_async(
            GROUNDING_DINO_MODEL,
            {
                "image": pet_data_uri,
                "query": "pet, dog, cat, animal",
                "box_threshold": 0.12,  # 최적화된 임계값
                "text_threshold": 0.12   # 최적화된 임계값
            }
        )
        
        if not pet_detection or not pet_detection.get("detections"):
            raise ValueError("반려동물을 찾을 수 없습니다. 다른 사진을 시도해주세요.")
        
        # 가장 큰 반려동물 바운딩 박스 선택
        pet_detections = pet_detection["detections"]
        pet_detection = max(pet_detections, key=lambda d: 
                           (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        pet_bbox = pet_detection["bbox"]
        
        # 5. SAM으로 사람 마스크 생성 - 정밀도 향상
        update_prediction(prediction_id, "masking_person", 40)
        person_segmentation = await run_replicate_async(
            SAM_MODEL,
            {
                "image": person_data_uri,
                "detection_type": "custom",
                "bbox": person_bbox,
                "mask_color": "#FFFFFF",
                "sam_args": {
                    "points_per_side": 32,  # 더 높은 정밀도
                    "pred_iou_thresh": 0.88, # 더 엄격한 기준
                    "stability_score_thresh": 0.92,
                }
            }
        )
        
        if not person_segmentation or not person_segmentation.get("combined_mask"):
            raise ValueError("사람 마스크 생성에 실패했습니다.")
        
        # 6. SAM으로 반려동물 마스크 생성 - 정밀도 향상
        update_prediction(prediction_id, "masking_pet", 50)
        pet_segmentation = await run_replicate_async(
            SAM_MODEL,
            {
                "image": pet_data_uri,
                "detection_type": "custom",
                "bbox": pet_bbox,
                "mask_color": "#FFFFFF",
                "sam_args": {
                    "points_per_side": 32,  # 더 높은 정밀도
                    "pred_iou_thresh": 0.88, # 더 엄격한 기준
                    "stability_score_thresh": 0.92,
                }
            }
        )
        
        if not pet_segmentation or not pet_segmentation.get("combined_mask"):
            raise ValueError("반려동물 마스크 생성에 실패했습니다.")
        
        # 7. 마스크 이미지 다운로드
        person_mask_path = await http_file_to_url(
            person_segmentation["combined_mask"],
            "person_mask",
            prediction_id
        )
        
        pet_mask_path = await http_file_to_url(
            pet_segmentation["combined_mask"],
            "pet_mask",
            prediction_id
        )
        
        # 8. 합성 이미지 준비 - 원본 객체 뚜렷하게 유지
        update_prediction(prediction_id, "preparing_composition", 60)
        
        # 사람/반려동물 이미지 로드
        person_img = Image.open(person_path).convert('RGBA')
        person_mask = Image.open(person_mask_path).convert('L')
        pet_img = Image.open(pet_path).convert('RGBA')
        pet_mask = Image.open(pet_mask_path).convert('L')
        
        # 이미지 크기 분석
        person_width, person_height = person_img.size
        pet_width, pet_height = pet_img.size
        
        # 캔버스 크기 결정 - SD 최적 크기 (768x512)
        canvas_width, canvas_height = 768, 512
        
        # 장면 유형에 따른 배치 조정
        if scene_type.lower() == "sofa":
            # 소파에 나란히 앉은 구도
            canvas_bg_color = (240, 240, 240, 0)  # 투명 배경
            canvas = Image.new('RGBA', (canvas_width, canvas_height), canvas_bg_color)
            composite_mask = Image.new('L', (canvas_width, canvas_height), 0)
            
            # 사람 크기 조절 - 캔버스의 45-50% 차지
            person_ratio = min(canvas_height * 0.85 / person_height, 
                             canvas_width * 0.45 / person_width)
            person_new_size = (
                int(person_width * person_ratio),
                int(person_height * person_ratio)
            )
            
            # 반려동물 크기 조절 - 사람 크기의 60-80%
            pet_ratio = min(person_new_size[1] * 0.8 / pet_height,
                          person_new_size[0] * 0.8 / pet_width)
            pet_new_size = (
                int(pet_width * pet_ratio),
                int(pet_height * pet_ratio)
            )
            
            # 위치 계산 - 사람은 왼쪽, 반려동물은 오른쪽 (나란히 배치)
            person_pos = (
                canvas_width // 4 - person_new_size[0] // 2,
                canvas_height // 2 - person_new_size[1] // 2 + 20
            )
            
            pet_pos = (
                canvas_width * 3 // 4 - pet_new_size[0] // 2,
                canvas_height // 2 - pet_new_size[1] // 3  # 조금 더 위쪽에 배치
            )
            
        else:  # 기본: 공원에서 공놀이
            # 공원에서 상호작용하는 구도
            canvas_bg_color = (240, 240, 240, 0)  # 투명 배경
            canvas = Image.new('RGBA', (canvas_width, canvas_height), canvas_bg_color)
            composite_mask = Image.new('L', (canvas_width, canvas_height), 0)
            
            # 사람 크기 조절 - 캔버스의 45-55% 차지
            person_ratio = min(canvas_height * 0.9 / person_height, 
                             canvas_width * 0.45 / person_width)
            person_new_size = (
                int(person_width * person_ratio),
                int(person_height * person_ratio)
            )
            
            # 반려동물 크기 조절 - 더 자연스러운 비율
            pet_ratio = min(person_new_size[1] * 0.7 / pet_height,
                          person_new_size[0] * 0.7 / pet_width)
            pet_new_size = (
                int(pet_width * pet_ratio),
                int(pet_height * pet_ratio)
            )
            
            # 위치 계산 - 사람과 반려동물이 상호작용하는 구도
            person_pos = (
                canvas_width // 3 - person_new_size[0] // 2,
                canvas_height // 2 - person_new_size[1] // 2 + 20
            )
            
            pet_pos = (
                canvas_width * 2 // 3 - pet_new_size[0] // 2,
                canvas_height // 2 + person_new_size[1] // 8  # 사람보다 약간 아래에 배치
            )
        
        # 크기 조절 - 고품질 리사이징
        person_img_resized = person_img.resize(person_new_size, Image.LANCZOS)
        person_mask_resized = person_mask.resize(person_new_size, Image.LANCZOS)
        pet_img_resized = pet_img.resize(pet_new_size, Image.LANCZOS)
        pet_mask_resized = pet_mask.resize(pet_new_size, Image.LANCZOS)
        
        # 마스크 품질 향상 - 경계선 개선
        # 1. 더 선명한 마스크 경계를 위한 처리
        # 마스크 경계 향상 - 더 선명하게
        person_mask_enhanced = person_mask_resized.point(lambda p: min(255, int(p * 1.2)))
        pet_mask_enhanced = pet_mask_resized.point(lambda p: min(255, int(p * 1.2)))
        
        # 약간의 블러만 적용 (너무 많은 블러는 객체 투명도 문제 유발)
        person_mask_blurred = person_mask_enhanced.filter(ImageFilter.GaussianBlur(radius=0.7))
        pet_mask_blurred = pet_mask_enhanced.filter(ImageFilter.GaussianBlur(radius=0.7))
        
        # 마스크 경계 개선
        person_mask_final = person_mask_blurred.point(lambda p: 255 if p > 200 else p if p > 50 else 0)
        pet_mask_final = pet_mask_blurred.point(lambda p: 255 if p > 200 else p if p > 50 else 0)
        
        # 알파 채널을 사용하여 객체 분리
        # - 원본 이미지 색상 보존을 위해 알파 채널만 조정
        person_rgba = Image.new('RGBA', person_new_size)
        for y in range(person_new_size[1]):
            for x in range(person_new_size[0]):
                r, g, b, a = person_img_resized.getpixel((x, y))
                alpha = person_mask_final.getpixel((x, y))
                person_rgba.putpixel((x, y), (r, g, b, alpha))
        
        pet_rgba = Image.new('RGBA', pet_new_size)
        for y in range(pet_new_size[1]):
            for x in range(pet_new_size[0]):
                r, g, b, a = pet_img_resized.getpixel((x, y))
                alpha = pet_mask_final.getpixel((x, y))
                pet_rgba.putpixel((x, y), (r, g, b, alpha))
        
        # 합성 이미지에 객체 배치 - RGBA 모드에서 작업
        temp_canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        temp_canvas.paste(person_rgba, person_pos, person_rgba)
        temp_canvas.paste(pet_rgba, pet_pos, pet_rgba)
        
        # 최종 RGB 이미지로 변환 (합성용)
        composite = Image.new('RGB', (canvas_width, canvas_height), (240, 240, 240))
        composite.paste(temp_canvas, (0, 0), temp_canvas)
        
        # SD 인페인팅용 마스크 생성 - 객체 위치만 마스킹
        for y in range(person_new_size[1]):
            for x in range(person_new_size[0]):
                px, py = x + person_pos[0], y + person_pos[1]
                if 0 <= px < canvas_width and 0 <= py < canvas_height:
                    alpha = person_mask_final.getpixel((x, y))
                    if alpha > 30:  # 임계값 설정
                        composite_mask.putpixel((px, py), 255)
        
        for y in range(pet_new_size[1]):
            for x in range(pet_new_size[0]):
                px, py = x + pet_pos[0], y + pet_pos[1]
                if 0 <= px < canvas_width and 0 <= py < canvas_height:
                    alpha = pet_mask_final.getpixel((x, y))
                    if alpha > 30:  # 임계값 설정
                        composite_mask.putpixel((px, py), 255)
        
        # 마스크 최종 처리 - 경계선 향상
        composite_mask = composite_mask.filter(ImageFilter.GaussianBlur(radius=1))
        composite_mask = composite_mask.point(lambda p: 255 if p > 30 else 0)  # 선명한 경계
        
        # 합성 이미지 및 마스크 저장
        composite_path = f"temp/composite_{prediction_id}.png"
        composite_mask_path = f"temp/composite_mask_{prediction_id}.png"
        composite.save(composite_path)
        composite_mask.save(composite_mask_path)
        
        # 9. SD Inpainting으로 배경 생성 - 장면에 맞는 프롬프트
        update_prediction(prediction_id, "generating_background", 80)
        
        # 마스크 반전 - 배경만 생성하도록
        inverted_mask = Image.eval(composite_mask, lambda p: 255 - p)
        inverted_mask_path = f"temp/inverted_mask_{prediction_id}.png"
        inverted_mask.save(inverted_mask_path)
        
        # 이미지와 마스크를 데이터 URI로 변환
        composite_data_uri = file_to_data_uri(composite_path, "image/png")
        inverted_mask_data_uri = file_to_data_uri(inverted_mask_path, "image/png")
        
        # 장면별 프롬프트 설정
        if scene_type.lower() == "sofa":
            prompt = "A cozy living room setting with a comfortable sofa, natural indoor lighting, warm home environment, high quality, photorealistic"
            negative_prompt = "distorted, blurry, low quality, unnatural poses, additional people, additional animals, text, watermarks"
        else:  # 공원 장면
            prompt = "A beautiful park setting with green grass, trees in background, natural sunlight, person and pet playing in park, high quality, photorealistic"
            negative_prompt = "distorted, blurry, low quality, unnatural poses, additional people, additional animals, text, watermarks"
        
        # SD Inpainting으로 배경 생성
        inpainting_result = await run_replicate_async(
            SD_INPAINT_MODEL,
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": composite_data_uri,
                "mask": inverted_mask_data_uri,
                "num_inference_steps": 25,  # 품질 균형
                "guidance_scale": 7.5,    # 품질 향상
                "scheduler": "DPMSolverMultistep",
                "strength": 0.65         # 배경을 뚜렷하게
            }
        )
        
        if not inpainting_result or not isinstance(inpainting_result, list) or len(inpainting_result) == 0:
            raise ValueError("배경 생성에 실패했습니다")
        
        # 10. 결과 이미지 저장 및 반환
        result_image_url = inpainting_result[0]
        
        # 결과 이미지 다운로드
        result_path = await http_file_to_url(result_image_url, f"result_{prediction_id}", prediction_id)

        # base64로 인코딩하여 반환
        with open(result_path, "rb") as f:
            result_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # 성공 상태 업데이트
        update_prediction(prediction_id, "succeeded", 100, [f"data:image/png;base64,{result_base64}"])
        
    except Exception as e:
        logger.error(f"이미지 처리 중 오류: {str(e)}")
        update_prediction(prediction_id, "failed", 0, None, str(e))
    finally:
        # 임시 파일 정리
        try:
            for prefix in ["person", "pet", "person_mask", "pet_mask", 
                          "composite", "composite_mask", "inverted_mask", f"result_{prediction_id}"]:
                temp_path = f"temp/{prefix}_{prediction_id}.{'png' if 'mask' in prefix or 'composite' in prefix else 'jpg'}"
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"임시 파일 정리 중 오류: {str(e)}")

@app.post("/predictions", response_model=PredictionResponse)
async def create_prediction(
    background_tasks: BackgroundTasks,
    request: PredictionRequest = Body(...)
):
    """Base64 이미지를 사용하여 합성 작업 시작"""
    try:
        # 새 예측 ID 생성
        prediction_id = str(uuid.uuid4())
        
        # Base64 이미지 추출
        person_base64 = request.input.get("person_image")
        pet_base64 = request.input.get("pet_image")
        
        if not person_base64 or not pet_base64:
            raise HTTPException(status_code=400, detail="person_image와 pet_image가 필요합니다")
        
        # 장면 유형 - 기본값: 공원, 옵션: 소파
        scene_type = request.scene_type
        if scene_type not in ["park", "sofa"]:
            scene_type = "park"  # 기본값
        
        # 초기 상태 설정
        update_prediction(prediction_id, "starting", 0)
        
        # 백그라운드에서 처리 시작
        background_tasks.add_task(
            process_images,
            prediction_id,
            person_base64,
            pet_base64,
            scene_type
        )
        
        return PredictionResponse(id=prediction_id, status="starting")
        
    except Exception as e:
        logger.error(f"create_prediction 에러: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """예측 상태 조회"""
    if prediction_id not in prediction_data:
        raise HTTPException(status_code=404, detail="예측을 찾을 수 없습니다")
    
    data = prediction_data[prediction_id]
    status_map = {
        "starting": "starting",
        "detecting_person": "processing",
        "detecting_pet": "processing",
        "masking_person": "processing",
        "masking_pet": "processing", 
        "preparing_composition": "processing",
        "generating_background": "processing",
        "succeeded": "succeeded",
        "failed": "failed"
    }
    
    replicate_status = status_map.get(data.get("status", ""), "processing")
    
    response = {
        "id": prediction_id,
        "status": replicate_status
    }
    
    # 진행률 추가
    if "progress" in data:
        response["progress"] = data["progress"]
    
    # 완료된 작업인 경우 결과 포함
    if data.get("status") == "succeeded" and "output" in data:
        response["output"] = data["output"]
    
    # 실패한 작업인 경우 오류 메시지 포함
    if data.get("status") == "failed" and "error" in data:
        response["error"] = data["error"]
    
    return response

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        # 모델 캐시 상태 확인
        cache_status = {}
        current_time = asyncio.get_event_loop().time()
        
        for model, last_time in last_prediction_time.items():
            time_diff = current_time - last_time
            cache_status[model] = {
                "last_used": f"{time_diff:.1f} seconds ago",
                "status": "warm" if time_diff < 900 else "cold"
            }
        
        return {
            "status": "healthy",
            "replicate_configured": bool(REPLICATE_API_TOKEN),
            "active_predictions": len(prediction_data),
            "model_status": cache_status
        }
    except Exception as e:
        logger.error(f"헬스 체크 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="헬스 체크 실패")

# 모델 웜업 시스템 - 주기적으로 모델이 cold되는 것 방지
async def keep_models_warm():
    """주기적으로 모델 재워밍업"""
    while True:
        try:
            current_time = asyncio.get_event_loop().time()
            # 모든 모델 확인
            for model_name, last_call_time in list(last_prediction_time.items()):
                time_since_last_call = current_time - last_call_time
                # 10분 이상 지난 모델만 워밍업 (너무 자주 호출하지 않도록)
                if time_since_last_call > 600 and time_since_last_call < 850:  # 10-14분 사이
                    logger.info(f"모델 {model_name} 워밍업 실행 중 (마지막 호출 후 {time_since_last_call:.1f}초)")
                    # 워밍업 실행 - 가벼운 입력으로
                    try:
                        if "sam" in model_name.lower():
                            await warm_up_sam()
                        elif "dino" in model_name.lower():
                            await warm_up_dino()
                        elif "inpaint" in model_name.lower():
                            await warm_up_inpaint()
                    except Exception as e:
                        logger.error(f"모델 {model_name} 워밍업 실패: {str(e)}")
            
            # 12분마다 체크 (10분보다 좀 더 길게)
            await asyncio.sleep(720)
        except Exception as e:
            logger.error(f"모델 워밍업 프로세스 오류: {str(e)}")
            await asyncio.sleep(300)  # 오류 발생 시 5분 후 재시도

async def warm_up_sam():
    """SAM 모델 워밍업"""
    # 더미 이미지 생성 (작은 크기로)
    dummy_img = Image.new('RGB', (64, 64), color='black')
    buffer = io.BytesIO()
    dummy_img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    img_data_uri = f"data:image/jpeg;base64,{img_base64}"
    
    await run_replicate_async(
        SAM_MODEL,
        {
            "image": img_data_uri,
            "detection_type": "automatic",
            "sam_args": {
                "points_per_side": 8  # 가벼운 설정
            }
        }
    )

async def warm_up_dino():
    """GroundingDINO 모델 워밍업"""
    # 더미 이미지 생성 (작은 크기로)
    dummy_img = Image.new('RGB', (64, 64), color='black')
    buffer = io.BytesIO()
    dummy_img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    img_data_uri = f"data:image/jpeg;base64,{img_base64}"
    
    await run_replicate_async(
        GROUNDING_DINO_MODEL,
        {
            "image": img_data_uri,
            "query": "object",
            "box_threshold": 0.25,
            "text_threshold": 0.25
        }
    )

async def warm_up_inpaint():
    """SD Inpainting 모델 워밍업"""
    # 더미 이미지와 마스크 생성 (작은 크기로)
    dummy_img = Image.new('RGB', (64, 64), color='white')
    mask_img = Image.new('L', (64, 64), color=128)
    
    img_buffer = io.BytesIO()
    dummy_img.save(img_buffer, format='JPEG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    img_data_uri = f"data:image/jpeg;base64,{img_base64}"
    
    mask_buffer = io.BytesIO()
    mask_img.save(mask_buffer, format='PNG')
    mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
    mask_data_uri = f"data:image/png;base64,{mask_base64}"
    
    await run_replicate_async(
        SD_INPAINT_MODEL,
        {
            "prompt": "simple background",
            "image": img_data_uri,
            "mask": mask_data_uri,
            "num_inference_steps": 10,  # 워밍업용으로 적은 스텝 사용
            "scheduler": "K_EULER"      # 지원되는 스케줄러로 변경
        }
    )

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 워밍업 및 타이머 시작"""
    try:
        # 백그라운드 작업으로 모델 워밍업 시작
        asyncio.create_task(keep_models_warm())
        
        logger.info("초기 모델 워밍업 시작...")
        
        # 더미 이미지 생성 (작은 크기)
        dummy_img = Image.new('RGB', (64, 64), color='black')
        buffer = io.BytesIO()
        dummy_img.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_data_uri = f"data:image/jpeg;base64,{img_base64}"
        
        # 마스크 이미지 생성 (작은 크기)
        mask_img = Image.new('L', (64, 64), color=0)
        draw = ImageDraw.Draw(mask_img)
        draw.rectangle((16, 16, 48, 48), fill=255)
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        mask_data_uri = f"data:image/png;base64,{mask_base64}"
        
        # 병렬로 모델 초기화 (3개 모델 동시에)
        await asyncio.gather(
            warm_up_sam(),
            warm_up_dino(),
            warm_up_inpaint()
        )
        
        logger.info("모든 모델 워밍업 완료")
        
    except Exception as e:
        logger.error(f"모델 워밍업 실패: {str(e)}")
        logger.warning("일부 모델 워밍업이 실패했지만 서버는 계속 실행됩니다.")

# 서버 실행코드 (로컬 테스트용)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5050))
    uvicorn.run(app, host="0.0.0.0", port=port)