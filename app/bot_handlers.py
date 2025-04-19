from aiogram import Bot, F, types, Router
from aiogram.types import FSInputFile
from aiogram.filters import CommandStart, Command
import numpy as np
import imghdr       # Для проверки формата изображения
#import re
from pathlib import Path
import logging, os                                         

import bot_keyboards as kb
import common_utils as utils
import FasterRCNN as faster
import YOLOv12 as yolo
#from app.constants import ARTEFACTS_DIR, CLASS_NAMES
import EasyOCR as easyocr


router = Router()

@router.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer("Привет! Загрузи изображение банковской карты.")   #, reply_markup=kb.main


# Обработка изображений
@router.message(F.photo)
async def handle_photo(message: types.Message, bot: Bot):
    try:
        # Скачиваем изображение
        file_id = message.photo[-1].file_id  # Берем фото с максимальным качеством
        file = await bot.get_file(file_id)
        file_path = file.file_path
        
        # Загружаем файл в память
        downloaded_file = await bot.download_file(file_path)
        file_bytes = downloaded_file.read()

        # 1. Проверяем реальный формат изображения
        image_format = imghdr.what(None, h=file_bytes)
        if image_format not in ['jpeg', 'png']:
            await message.answer("Пожалуйста, отправьте изображение в формате JPG или PNG")
            return

        # 2. Сохраняем с правильным расширением
        utils.clean_artefact_dir()
        original_filename = f"artefacts/{file_id}.{image_format}"
        with open(original_filename, "wb") as f:
            f.write(file_bytes)
    
        await message.answer("Фотография успешно загружена. Выбери детектор полей реквизитов.", reply_markup=kb.detect)

    except Exception as e:
        await message.answer(f"Ошибка: {e}")


@router.message(F.text == 'YOLOv12')
async def detect_yolo_v12(message: types.Message, bot: Bot):
    try:
        utils.delete_old_detections()
        img_path = utils.get_image_from_artefacts()
        yolo.main(img_path)
        #await message.answer(f"Файл сохранен по пути: {img_path}")
        await message.answer("Производится детекция объектов...")

        image_files = utils.get_list_of_images()

        if not image_files:
            logging.error(f"Нет изображений с корректным форматом имени.")
            return

        sent_count = 0

        for image_file, class_name, confidence in image_files:
            try:
                caption = (
                    f"🏷 Класс: {class_name}\n"
                    f"🟢 Уверенность: {confidence}"
                )
                
                await message.answer_photo(
                    FSInputFile(image_file),
                    caption=caption
                )
                sent_count += 1
            except Exception as e:
                logging.error(f"Error sending {image_file.name}: {e}")

        await message.answer(f"✅ Отправлено {sent_count} результатов (из {len(image_files)})", reply_markup=kb.ocr)
        
    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)}")
        logging.error(f"Detect error: {e}")


@router.message(F.text == 'Faster R-CNN')
async def detect_faster_rcnn(message: types.Message, bot: Bot):
    try:
        utils.delete_old_detections()
        img_path = utils.get_image_from_artefacts()
        faster.main(img_path)
        #await message.answer(f"Файл сохранен по пути: {img_path}")
        await message.answer("Производится детекция объектов...")

        # # Регулярное выражение для проверки формата имени файла
        # pattern = re.compile(r'^(\d+)_(\d+\.\d+)_')

        # # Получаем и фильтруем изображения
        # image_files = []
        # for file in ARTEFACTS_DIR.glob('*.*'):
        #     logging.info(f"Имя файла: {file.name}")
        #     if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        #         logging.info(f"Файл существует и явл. изображением: {file.name}")
        #         match = pattern.match(file.stem)
        #         if match:
        #             #class_id = match.group(1)
        #             class_name = CLASS_NAMES[int(match.group(1))]
        #             logging.info(f"Класс изображениея: {class_name}")
        #             confidence = match.group(2) #.replace('_', '.')
        #             #image_files.append((file, class_id, confidence))
        #             image_files.append((file, class_name, confidence))

        # if not image_files:
        #     await message.answer("Нет изображений с корректным форматом имени")
        #     return
                
        # Сортируем по уверенности (от высокой к низкой)
        #image_files.sort(key=lambda x: float(x[2]), reverse=True)

        image_files = utils.get_list_of_images()

        if not image_files:
            logging.error(f"Нет изображений с корректным форматом имени.")
            return

        sent_count = 0

        for image_file, class_name, confidence in image_files:
            try:
                caption = (
                    f"🏷 Класс: {class_name}\n"
                    f"🟢 Уверенность: {confidence}"
                )
                
                await message.answer_photo(
                    FSInputFile(image_file),
                    caption=caption
                )
                sent_count += 1
            except Exception as e:
                logging.error(f"Error sending {image_file.name}: {e}")

        await message.answer(f"✅ Отправлено {sent_count} результатов (из {len(image_files)})", reply_markup=kb.ocr)
        
    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)}")
        logging.error(f"Detect error: {e}")


@router.message(F.text == 'EasyOCR')
async def recognition_EasyOCR(message: types.Message, bot: Bot):
    #await message.answer(f"Здесь будет результат распознавания...")
    image_files = utils.get_list_of_images()

    if not image_files:
        logging.error(f"Нет изображений с корректным форматом имени.")
        return

    sent_count = 0
    # try:
    for image_file, class_name, confidence in image_files:
        #logging.INFO(f"Передаем в EasyOCR файл: {image_file.name}.")
        try:
            img_path, results, class_id, processed_img = easyocr.recognize_images_in_directory(image_file, languages=['en', 'ru'], gpu=False)

            print("Подготавливаем результаты")
            orig_temp_path, processed_temp_path, recognized_texts = utils.prepare_enhanced_results(img_path, results, class_id, processed_img)

            # Отправляем пользователю
            await message.answer_photo(FSInputFile(orig_temp_path, filename="detected region.jpg"))
            await message.answer_photo(FSInputFile(processed_temp_path, filename="OCR processed.jpg"))
            await message.answer(f"{recognized_texts}", reply_markup=kb.ocr)
            # Отправляем распознанный текст
            #text_message = "Распознанный текст:\n" + "\n".join(texts)
            #await message.answer(text_message)
            os.unlink(orig_temp_path)
            os.unlink(processed_temp_path)

            sent_count += 1
        except Exception as e:
            logging.error(f"Error sending {image_file.name}: {e}")

    await message.answer(f"✅ Отправлено {sent_count} результатов (из {len(image_files)})", reply_markup=kb.ocr)
        
    # except Exception as e:
    #     await message.answer(f"❌ Ошибка: {str(e)}")
    #     logging.error(f"Detect error: {e}")