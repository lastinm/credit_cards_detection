# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from pathlib import Path

app = FastAPI()

# Конфигурация
IMAGES_DIR = Path("/home/lastinm/PROJECTS/credit_cards_detection/dataset/coco/valid/images")  # Папка с изображениями
CSV_FILE = Path("image_data.csv")  # Файл для хранения данных
FIELDS = ["CardHolder", "CardNumber", "DateExpired"]  # Названия полей для ввода

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory=IMAGES_DIR), name="static")
templates = Jinja2Templates(directory="templates")

def get_image_list():
    """Получаем список всех изображений"""
    return sorted([
        f for f in os.listdir(IMAGES_DIR) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
    ])

def get_processed_images():
    """Получаем список уже обработанных изображений"""
    if CSV_FILE.exists():
        try:
            df = pd.read_csv(CSV_FILE)
            return set(df["image"].unique())
        except:
            return set()
    return set()

def save_to_csv(data):
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
        next_image = next((img for img in all_images if img not in processed), None)
        
        if next_image:
            return templates.TemplateResponse("form.html", {
                "request": request,
                "image_file": next_image,
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
    field1: str = Form(default=""), # Поле может быть пустым
    field2: str = Form(...),
    field3: str = Form(default="")  # Поле может быть пустым

):
    try:
        if save_to_csv({
            "image": image_file,
            "field1": field1,
            "field2": field2,  # Сохраняем даже если пустое
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