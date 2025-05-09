# Приложение для создания эталонных значений распознаваемых полей

import os
import json
import cv2
import base64
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from pathlib import Path
from typing import Dict, List

app = FastAPI()

# Конфигурация
IMAGES_DIR = Path("/home/lastinm/PROJECTS/credit_cards_detection/dataset/ocr val")
ANNOTATION_FILE = Path("/home/lastinm/PROJECTS/credit_cards_detection/dataset/merged_annotations.coco_cleaned.json")
CSV_FILE = Path("image_data.csv")
FIELDS = ["CardHolder", "CardNumber", "DateExpired"]

# Загружаем COCO аннотации
with open(ANNOTATION_FILE) as f:
    coco_data = json.load(f)

# Создаем структуры для быстрого доступа
images_info = {img['id']: img for img in coco_data['images']}
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
annotations: Dict[int, List[dict]] = {}

for ann in coco_data['annotations']:
    if ann['image_id'] not in annotations:
        annotations[ann['image_id']] = []
    annotations[ann['image_id']].append(ann)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory=IMAGES_DIR), name="static")
templates = Jinja2Templates(directory="templates")

def draw_annotations(image_path: Path, image_id: int) -> str:
    """Рисует bounding boxes и названия классов на изображении и возвращает base64 строку"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if image_id in annotations:
        for ann in annotations[image_id]:
            x, y, w, h = map(int, ann['bbox'])
            category_name = categories.get(ann['category_id'], 'unknown')
            
            # Рисуем прямоугольник (красный)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Добавляем текст с названием класса
            cv2.putText(img, category_name, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Конвертируем изображение в base64 для отображения в HTML
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def get_image_list() -> List[dict]:
    """Получаем список всех изображений с аннотациями"""
    return sorted([
        {"id": img_id, "file_name": img_info['file_name']} 
        for img_id, img_info in images_info.items()
    ], key=lambda x: x['file_name'])

def get_processed_images() -> set:
    """Получаем список уже обработанных изображений"""
    if CSV_FILE.exists():
        try:
            df = pd.read_csv(CSV_FILE)
            return set(df["image"].unique())
        except Exception:
            return set()
    return set()

def save_to_csv(data: dict) -> bool:
    """Надежное сохранение данных в CSV"""
    try:
        df = pd.DataFrame([data])
        if CSV_FILE.exists():
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        print(f"Ошибка сохранения: {e}")
        return False

@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    try:
        all_images = get_image_list()
        processed = get_processed_images()
        
        # Находим следующее необработанное изображение
        next_image = next(
            (img for img in all_images if img['file_name'] not in processed), 
            None
        )
        
        if next_image:
            image_path = IMAGES_DIR / next_image['file_name']
            image_data = draw_annotations(image_path, next_image['id'])
            
            return templates.TemplateResponse("form.html", {
                "request": request,
                "image_file": next_image['file_name'],
                "image_data": image_data,
                "fields": FIELDS
            })
        return templates.TemplateResponse("completed.html", {"request": request})
    except Exception as e:
        print(f"Ошибка: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.post("/process")
async def process_form(
    request: Request,
    image_file: str = Form(...),
    field1: str = Form(default=""),
    field2: str = Form(...),
    field3: str = Form(default="")
):
    try:
        if save_to_csv({
            "image": image_file,
            "field1": field1,
            "field2": field2,
            "field3": field3
        }):
            return RedirectResponse(url="/", status_code=303)
        raise Exception("Не удалось сохранить данные")
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

def run_app():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run_app()