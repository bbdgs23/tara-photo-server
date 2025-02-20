from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import replicate
import os
import io
import base64
from PIL import Image, ImageOps
import requests
from typing import List, Optional, Dict
import uuid
import tempfile
import asyncio
import datetime
import logging
import random
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Input model definition
class PredictionInput(BaseModel):
    input: Dict[str, str]  # person_image and pet_image as base64 strings
    scene_type: str = "park"  # Default value "park"

# Response models
class PredictionResponse(BaseModel):
    id: str
    status: str = "starting"

class PredictionStatusResponse(BaseModel):
    id: str
    status: str
    output: Optional[List[str]] = None
    progress: int = 0
    error: Optional[str] = None

app = FastAPI(title="Image Composition API")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task storage (in production, use Redis or a database)
removal_tasks = {}

# Environment variables
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN environment variable is not set")

# Model IDs
BACKGROUND_REMOVER_MODEL = "851-labs/background-remover:a029dff38972b5fda4ec5d75d7d1cd25aeff621d2cf4946a41055d7db66b80bc"
STABLE_DIFFUSION_MODEL = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"

# Background scene prompts
SCENE_PROMPTS = {
    "park": "A serene natural park landscape with lush green trees, soft grass, and gentle sunlight, photorealistic",
    "forest": "Dense forest with tall trees, dappled sunlight, mossy ground, no people, no animals, ultra-detailed",
    "meadow": "Peaceful meadow with wildflowers, rolling hills, soft natural lighting, no artificial objects",
    "mountain": "Scenic mountain landscape with distant peaks, green valleys, clear sky, no human-made structures",
    "beach": "Tranquil beach scene with soft sand, calm ocean, distant horizon, no people or buildings"
}

def download_image(url: str) -> Image.Image:
    """Download image from URL"""
    response = requests.get(url, timeout=30)
    return Image.open(io.BytesIO(response.content))

import random

def compose_images_with_background(
    foreground_images: List[Image.Image], 
    background_image: Image.Image
) -> Image.Image:
    """
    Compose images with a background
    - Resize and position foreground images closer together
    - Align images more towards bottom of the frame
    - Overlay foreground images on background
    """
    # Set a standard height for foreground images (now square)
    target_height = 1024  # 정사각형 크기
    
    # Resize and maintain aspect ratio for foreground images
    def resize_image(img: Image.Image) -> Image.Image:
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        return img.resize((new_width, target_height), Image.LANCZOS)
    
    # Resize images
    resized_foregrounds = [resize_image(img) for img in foreground_images]
    
    # Randomize image order to mix up positioning
    random.shuffle(resized_foregrounds)
    
    # Calculate total width with tighter spacing
    total_foreground_width = sum(img.width for img in resized_foregrounds)
    spacing = 10  # 이미지 사이 간격 줄임
    padding = 40  # 이미지 주변 여백
    total_width = total_foreground_width + spacing + (2 * padding)
    
    # Resize background to match total width and height
    background_resized = ImageOps.fit(
        background_image, 
        (total_width, total_width),  # 정사각형으로 맞춤 
        Image.LANCZOS
    )
    
    # Create a copy of the background to paste on
    composite = background_resized.copy()
    
    # Calculate vertical positioning to align more towards bottom
    # Move images up from the very bottom, leaving some space
    vertical_offset = int(total_width * 0.15)  # 바닥에서 15% 위에 배치
    
    # Paste foreground images closer together and lower
    current_x = (total_width - total_foreground_width - spacing) // 2  # 가운데 정렬
    for img in resized_foregrounds:
        # 수직 위치를 아래쪽으로 조정
        vertical_position = total_width - img.height - vertical_offset
        
        composite.paste(
            img, 
            (current_x, vertical_position), 
            img if img.mode == 'RGBA' else None
        )
        current_x += img.width + spacing
    
    return composite

# 다양하고 사실적인 배경 프롬프트 확장
SCENE_PROMPTS = {
    # 실내 배경
    "living_room": "Photorealistic modern living room interior, soft natural light from large windows, clean minimalist design, detailed mid-century furniture, hardwood floors, warm afternoon sunlight, high-resolution professional interior photography, 8k resolution",
    "kitchen": "Ultra-realistic contemporary kitchen, bright natural light, marble countertops, stainless steel appliances, detailed textures, professional interior design photography, clean lines, 8k resolution",
    "home_office": "Highly detailed home office with large panoramic windows, wooden desk, ergonomic chair, indoor plants, soft natural light, professional workspace photography, warm neutral tones, 8k resolution",
    "bedroom": "Luxurious bedroom interior, soft diffused daylight, high-end furniture, crisp bedding, detailed window treatments, professional real estate photography, clean and elegant design, 8k resolution",
    
    # 실외 배경
    "park": "Photorealistic urban park scene, well-maintained grass, wooden bench, mature trees, soft golden hour lighting, detailed landscape, professional photography, 8k resolution",
    "garden": "Meticulously landscaped backyard garden, stone pathway, blooming flowers, neatly trimmed hedges, soft natural lighting, high-definition texture, professional landscape photography, 8k resolution",
    "patio": "Elegant outdoor patio with modern furniture, stone tile flooring, potted plants, soft afternoon sunlight, professional architectural photography, clean design, 8k resolution",
    "terrace": "Rooftop terrace with city skyline view, modern furniture, potted plants, golden hour lighting, professional urban photography, detailed architectural elements, 8k resolution",
    
    # 특별한 장소들
    "cafe_interior": "Cozy cafe interior, large windows, wooden tables, soft natural light, detailed coffee bar, professional interior design photography, warm atmosphere, 8k resolution",
    "art_gallery": "Modern art gallery interior, minimalist design, white walls, soft diffused lighting, professional exhibition photography, clean lines, 8k resolution"
}

def save_image_to_base64(image: Image.Image) -> str:
    """Save PIL Image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

async def remove_background(image_base64: str) -> str:
    """Process a single image and remove background"""
    logger.info(f"Starting background removal")
    
    # Temporary file
    temp_file = None
    
    try:
        # Validate base64 input - remove header if present
        if image_base64.startswith('data:image'):
            image_base64 = image_base64.split(',')[1]
        
        # 1. Convert base64 to temporary file
        try:
            # Decode base64 to image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
            image.save(temp_file, format='PNG')
            logger.info(f"Created temporary image file: {temp_file}")
        except Exception as e:
            logger.error(f"Failed to convert base64 image: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")
        
        # 2. Call background remover API with retries
        max_retries = 3
        retry_delay = 2
        result_url = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Background removal attempt {attempt+1}/{max_retries}")
                
                # 파일 스트림을 연 후 바로 닫아 파일 잠금 방지
                file_content = None
                with open(temp_file, 'rb') as f:
                    file_content = f.read()
                
                # Call Replicate API with file content instead of open file
                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        replicate.run,
                        BACKGROUND_REMOVER_MODEL,
                        input={"image": io.BytesIO(file_content)}
                    ),
                    timeout=60  # 60 second timeout
                )
                
                # Validate output
                if not output:
                    raise ValueError("Background remover returned empty result")
                
                # Handle different output formats
                if isinstance(output, str):
                    result_url = output
                elif isinstance(output, list) and len(output) > 0:
                    result_url = output[0]
                elif hasattr(output, 'url'):
                    result_url = output.url
                else:
                    logger.warning(f"Unexpected output format: {type(output)}")
                    result_url = str(output)
                
                logger.info(f"Background removal completed: {result_url[:30]}...")
                return result_url
                
            except asyncio.TimeoutError:
                logger.warning(f"Background removal timed out (attempt {attempt+1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise TimeoutError("Background removal operation timed out")
                await asyncio.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Background removal failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay)
        
        # Shouldn't reach here, but just in case
        raise Exception("Failed to remove background after all attempts")
    
    finally:
        # Clean up temporary file with retry to handle file locking issues
        if temp_file and os.path.exists(temp_file):
            for i in range(3):  # 3번 시도
                try:
                    os.close(os.open(temp_file, os.O_RDONLY))  # 열린 파일 핸들을 닫기
                    os.remove(temp_file)
                    logger.info(f"Deleted temporary file: {temp_file}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file (attempt {i+1}/3): {str(e)}")
                    await asyncio.sleep(0.5)  # 잠시 대기 후 재시도

async def generate_background(scene_type: str = None) -> str:
    """Generate background using Stable Diffusion with enhanced realism and randomization"""
    # 랜덤으로 장면 선택 (scene_type이 None이거나 제공되지 않은 경우)
    if not scene_type:
        scene_type = random.choice(list(SCENE_PROMPTS.keys()))
    
    logger.info(f"Generating background for scene type: {scene_type}")
    
    # Get prompt for the scene type, fallback to a random scene if not found
    prompt = SCENE_PROMPTS.get(scene_type.lower(), random.choice(list(SCENE_PROMPTS.values())))
    
    try:
        # Call Replicate API for background generation with increased realism
        output = await asyncio.wait_for(
            asyncio.to_thread(
                replicate.run,
                STABLE_DIFFUSION_MODEL,
                input={
                    "prompt": prompt,
                    "width": 1024,  # 정사각형 크기 
                    "height": 1024,
                    "num_outputs": 1,
                    "num_inference_steps": 85,  # 더 디테일한 이미지 생성을 위해 스텝 증가
                    "guidance_scale": 10.0,  # 프롬프트 따르는 정도 최대화
                    "negative_prompt": "low quality, blurry, sketch, drawing, cartoon, unrealistic, poor lighting, amateur photography, soft focus"  # 사실적이지 않은 이미지 방지
                }
            ),
            timeout=120  # 2 minute timeout for background generation
        )
        
        # Validate and return output
        if not output:
            raise ValueError("Stable Diffusion returned empty result")
        
        # Handle different output formats
        if isinstance(output, str):
            return output
        elif isinstance(output, list) and len(output) > 0:
            return output[0]
        else:
            logger.warning(f"Unexpected output format: {type(output)}")
            return str(output)
        
    except Exception as e:
        logger.error(f"Background generation failed: {str(e)}")
        raise

async def process_both_images(task_id: str):
    """Process both person and pet images with background generation"""
    logger.info(f"Starting processing for both images: task_id={task_id}")
    
    try:
        task = removal_tasks[task_id]
        person_image = task["input"].get("person_image")
        pet_image = task["input"].get("pet_image")
        scene_type = task.get("scene_type")  # scene_type을 optional로 변경
        
        # 1. Process person image
        removal_tasks[task_id]["status"] = "detecting_person"
        try:
            person_result_url = await remove_background(person_image)
            removal_tasks[task_id]["person_result_url"] = person_result_url
            logger.info(f"Person image processed successfully")
        except Exception as e:
            logger.error(f"Person image processing failed: {str(e)}")
            raise
        
        # 2. Process pet image
        removal_tasks[task_id]["status"] = "detecting_pet"
        try:
            pet_result_url = await remove_background(pet_image)
            removal_tasks[task_id]["pet_result_url"] = pet_result_url
            logger.info(f"Pet image processed successfully")
        except Exception as e:
            logger.error(f"Pet image processing failed: {str(e)}")
            raise
        
        # 3. Generate background
        removal_tasks[task_id]["status"] = "generating_background"
        try:
            background_url = await generate_background(scene_type)
            removal_tasks[task_id]["background_url"] = background_url
            # 실제 사용된 scene_type 저장
            removal_tasks[task_id]["actual_scene_type"] = scene_type or "random"
            logger.info(f"Background generated successfully")
        except Exception as e:
            logger.error(f"Background generation failed: {str(e)}")
            raise
        
        # 4. Composition preparation
        removal_tasks[task_id]["status"] = "compositing"
        
        # 다운로드 및 합성
        person_image_pil = download_image(person_result_url)
        pet_image_pil = download_image(pet_result_url)
        background_image_pil = download_image(background_url)
        
        # 배경 위에 사람과 동물 이미지 합성
        composed_image = compose_images_with_background(
            [person_image_pil, pet_image_pil], 
            background_image_pil
        )
        
        # 합성된 이미지를 base64로 변환
        composed_base64 = save_image_to_base64(composed_image)
        
        # 5. Update task with results
        removal_tasks[task_id]["status"] = "succeeded"
        
        output = [
            composed_base64,     # 최종 합성된 이미지 (첫 번째)
            str(person_result_url),   # URL을 문자열로 변환
            str(pet_result_url),      # URL을 문자열로 변환
            str(background_url)       # URL을 문자열로 변환
        ]
        
        removal_tasks[task_id]["output"] = output
        removal_tasks[task_id]["completed_at"] = datetime.datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        removal_tasks[task_id]["status"] = "failed"
        removal_tasks[task_id]["error"] = str(e)
        removal_tasks[task_id]["failed_at"] = datetime.datetime.now().isoformat()

@app.post("/predictions", response_model=PredictionResponse)
async def create_prediction(prediction_data: PredictionInput, background_tasks: BackgroundTasks):
    """
    Start image composition task (compatibility with original API)
    - input.person_image: Base64 encoded person image
    - input.pet_image: Base64 encoded pet image
    - scene_type: Background scene type (default: "park")
    """
    try:
        # Generate new task ID
        task_id = str(uuid.uuid4())
        
        # Validate inputs
        person_image = prediction_data.input.get("person_image")
        pet_image = prediction_data.input.get("pet_image")
        
        if not person_image:
            raise HTTPException(status_code=400, detail="person_image is required")
        if not pet_image:
            raise HTTPException(status_code=400, detail="pet_image is required")
        
        # Remove data URI prefix if present
        if person_image.startswith('data:image'):
            person_image = person_image.split(',')[1]
        if pet_image.startswith('data:image'):
            pet_image = pet_image.split(',')[1]
            
        # Initial task state
        removal_tasks[task_id] = {
            "id": str(task_id),  # 문자열로 명시적 변환
            "status": "detecting_person",
            "input": {
                "person_image": person_image,
                "pet_image": pet_image
            },
            "scene_type": str(prediction_data.scene_type),
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Start background processing for both images
        background_tasks.add_task(process_both_images, task_id)
        
        return {"id": task_id, "status": "starting"}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to create prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create prediction: {str(e)}")

@app.get("/predictions/{prediction_id}", response_model=PredictionStatusResponse)
async def get_prediction_status(prediction_id: str):
    """
    Check status of a prediction
    - prediction_id: Prediction ID
    """
    if prediction_id not in removal_tasks:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    task = removal_tasks[prediction_id]
    status = task["status"]
    
    # Status mapping for progress
    status_map = {
        "detecting_person": 30,
        "detecting_pet": 60,
        "generating_background": 75,
        "compositing": 90,
        "succeeded": 100,
        "failed": 0
    }
    
    # Determine progress
    progress = status_map.get(status, 0)
    
    # Response structure
    if status == "succeeded":
        return {
            "id": prediction_id,
            "status": status,
            "output": task.get("output", []),
            "progress": progress
        }
    elif status == "failed":
        return {
            "id": prediction_id,
            "status": status,
            "error": task.get("error"),
            "progress": progress
        }
    else:
        return {
            "id": prediction_id,
            "status": status,
            "progress": progress
        }

@app.get("/proxy-image")
async def proxy_image(url: str):
    """
    Proxy external image to solve CORS issues
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch image")

        return Response(
            content=response.content,
            media_type=response.headers.get('content-type', 'image/png')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image proxy error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Image Composition API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/upload-and-remove", response_model=PredictionResponse)
async def upload_and_remove(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload file and remove background in one step
    """
    try:
        # Read file content
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')
        
        # Create prediction request
        prediction_data = PredictionInput(
            input={
                "person_image": base64_image,
                "pet_image": base64_image  # 동일한 이미지로 테스트
            }
        )
        
        # Process normally
        return await create_prediction(prediction_data, background_tasks)
        
    except Exception as e:
        logger.error(f"Upload and remove failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload and remove failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)