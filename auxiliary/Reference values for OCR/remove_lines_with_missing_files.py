# Пробежаться по файлу image_data.csv с содержимым:
#  image,CardHolder,CardNumber,DateExpired
#  и удалить строки относящиеся к файлу, которого нет в каталоге: 
# /home/lastinm/PROJECTS/credit_cards_detection/dataset/ocr val

import pandas as pd
import os
from pathlib import Path

# Пути к файлам и директориям
csv_path = "image_data.csv"
images_dir = Path("/home/lastinm/PROJECTS/credit_cards_detection/dataset/ocr val")

def clean_missing_images(csv_path, images_dir):
    # Загружаем CSV файл
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Файл {csv_path} не найден!")
        return
    
    # Получаем список существующих изображений
    existing_images = set(f.name for f in images_dir.glob("*") if f.is_file())
    
    # Фильтруем DataFrame
    initial_count = len(df)
    df_clean = df[df["image"].apply(lambda x: x in existing_images)]
    removed_count = initial_count - len(df_clean)
    
    # Сохраняем очищенный файл
    df_clean.to_csv(csv_path, index=False)
    
    print(f"Обработка завершена. Удалено {removed_count} строк.")
    print(f"Сохранено {len(df_clean)} записей в {csv_path}")

if __name__ == "__main__":
    clean_missing_images(csv_path, images_dir)
    