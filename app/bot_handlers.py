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
import KerasOCR as kerasocr
import EasyOCR as easyocr
import TrOCR as trocr
import PaddleOCR as paddleocr

router = Router()


async def output_detect_result(message):
    image_files = utils.get_list_of_images()

    if not image_files:
        logging.error(f"Нет изображений с корректным форматом имени.")
        return

    sent_count = 0
    for image_file, class_name, confidence in image_files:
        try:
            if isinstance(confidence, str):
                confidence = float(confidence)
            elif not isinstance(confidence, (float, int)):
                confidence = 0.0  # или другое значение по умолчанию

            if confidence>0.85:
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
    #await message.answer(f"Файл сохранен по пути: {img_path}")
    await message.answer("Производится детекция объектов...")

    try:
        utils.delete_old_detections()
        img_path = utils.get_image_from_artefacts()
        yolo.main(img_path)        
    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)}")
        logging.error(f"Detect error: {e}")
    
    await output_detect_result(message)


@router.message(F.text == 'Faster R-CNN')
async def detect_faster_rcnn(message: types.Message, bot: Bot):
    #await message.answer(f"Файл сохранен по пути: {img_path}")
    await message.answer("Производится детекция объектов...")
    
    try:
        utils.delete_old_detections()
        img_path = utils.get_image_from_artefacts()
        faster.main(img_path)        
    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)}")
        logging.error(f"Detect error: {e}")

    await output_detect_result(message)


@router.message(F.text == 'KerasOCR')
async def recognition_KerasOCR(message: types.Message, bot: Bot):
    #await message.answer(f"Здесь будет результат распознавания...")
    image_files = utils.get_list_of_images()

    if not image_files:
        logging.error(f"Нет изображений с корректным форматом имени.")
        return

    sent_count = 0
    # try:
    for image_file, class_name, confidence in image_files:
        #logging.INFO(f"Передаем в KerasOCR файл: {image_file.name}.")
        try:
            full_text, predictions = kerasocr.recognize_with_confidence(image_file)
            print(f"Рапознан текст KerasOCR: {full_text}.")
            processed_temp_path = kerasocr.get_tmp_image_file(image_file, predictions)

            # Отправляем пользователю
            await message.answer_photo(FSInputFile(processed_temp_path, filename="OCR processed.jpg"))
            #await message.answer(f"{full_text}", reply_markup=kb.ocr)
            # Отправляем распознанный текст
            #text_message = "Распознанный текст:\n" + "\n".join(texts)
            #await message.answer(text_message)
            os.unlink(processed_temp_path)

            sent_count += 1
        except Exception as e:
            logging.error(f"Error sending {image_file.name}: {e}")

    await message.answer(f"✅ Отправлено {sent_count} результатов (из {len(image_files)})", reply_markup=kb.ocr)
        
    # except Exception as e:
    #     await message.answer(f"❌ Ошибка: {str(e)}")
    #     logging.error(f"Detect error: {e}")


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


@router.message(F.text == 'TrOCR')
async def recognition_TrOCR(message: types.Message, bot: Bot):

    #await message.answer(f"Здесь будет результат распознавания...")
    image_files = utils.get_list_of_images()

    if not image_files:
        logging.error(f"Нет изображений с корректным форматом имени.")
        return

    sent_count = 0
    # try:
    for image_file, class_name, confidence in image_files:
        #logging.INFO(f"Передаем в KerasOCR файл: {image_file.name}.")
        try:
            #img_path, results, class_id, processed_img = trocr.recognize_images_in_directory(image_file, languages=['en', 'ru'], gpu=False)
            outputs, processor = trocr.recognize_images_in_directory(image_file)
            full_text, confidences = trocr.get_text_with_confidence(outputs, processor)

            print("Подготавливаем результаты")
            await message.answer(f"\'{full_text}\' (уверенность: {confidences:.3f})")
            # orig_temp_path, processed_temp_path, recognized_texts = utils.prepare_enhanced_results(img_path, results, class_id, processed_img)

            # # Отправляем пользователю
            # await message.answer_photo(FSInputFile(orig_temp_path, filename="detected region.jpg"))
            # await message.answer_photo(FSInputFile(processed_temp_path, filename="OCR processed.jpg"))
            # await message.answer(f"{recognized_texts}", reply_markup=kb.ocr)
            # # Отправляем распознанный текст
            # #text_message = "Распознанный текст:\n" + "\n".join(texts)
            # #await message.answer(text_message)
            # os.unlink(orig_temp_path)
            # os.unlink(processed_temp_path)

            sent_count += 1
        except Exception as e:
            logging.error(f"Error sending {image_file.name}: {e}")

    await message.answer(f"✅ Отправлено {sent_count} результатов (из {len(image_files)})", reply_markup=kb.ocr)
        
    # except Exception as e:
    #     await message.answer(f"❌ Ошибка: {str(e)}")
    #     logging.error(f"Detect error: {e}")

@router.message(F.text == 'PaddleOCR')
async def recognition_PaddleOCR(message: types.Message, bot: Bot):

    #await message.answer(f"Здесь будет результат распознавания...")
    image_files = utils.get_list_of_images()

    if not image_files:
        logging.error(f"Нет изображений с корректным форматом имени.")
        return

    sent_count = 0
    # try:
    for image_file, class_name, confidence in image_files:
        #logging.INFO(f"Передаем в KerasOCR файл: {image_file.name}.")
        try:
            #img_path, results, class_id, processed_img = trocr.recognize_images_in_directory(image_file, languages=['en', 'ru'], gpu=False)
            full_text, confidences = paddleocr.recognize_images_in_directory(image_file)

            print("Подготавливаем результаты")
            await message.answer(f"\'{full_text}\' (уверенность: {confidences:.3f})")
            # orig_temp_path, processed_temp_path, recognized_texts = utils.prepare_enhanced_results(img_path, results, class_id, processed_img)

            # # Отправляем пользователю
            # await message.answer_photo(FSInputFile(orig_temp_path, filename="detected region.jpg"))
            # await message.answer_photo(FSInputFile(processed_temp_path, filename="OCR processed.jpg"))
            # await message.answer(f"{recognized_texts}", reply_markup=kb.ocr)
            # # Отправляем распознанный текст
            # #text_message = "Распознанный текст:\n" + "\n".join(texts)
            # #await message.answer(text_message)
            # os.unlink(orig_temp_path)
            # os.unlink(processed_temp_path)

            sent_count += 1
        except Exception as e:
            logging.error(f"Error sending {image_file.name}: {e}")

    await message.answer(f"✅ Отправлено {sent_count} результатов (из {len(image_files)})", reply_markup=kb.ocr)
        
    # except Exception as e:
    #     await message.answer(f"❌ Ошибка: {str(e)}")
    #     logging.error(f"Detect error: {e}")


@router.message(F.text == 'Сравнить OCR')
async def compare_OCR(message: types.Message, bot: Bot):
    pass

@router.message(F.text == 'TrOCR + PaddleOCR')
async def ensemble_OCR(message: types.Message, bot: Bot):
    image_files = utils.get_list_of_images()

    if not image_files:
        logging.error(f"Нет изображений с корректным форматом имени.")
        return

    sent_count = 0
    # try:
    for image_file, class_name, confidence in image_files:
        #logging.INFO(f"Передаем в KerasOCR файл: {image_file.name}.")
        try:
            if class_name == 'CardNumber':
                full_text, confidences = paddleocr.recognize_images_in_directory(image_file)

                print("Подготавливаем результаты")
                await message.answer(f"\'{full_text}\' (уверенность: {confidences:.3f})")
            else:
                outputs, processor = trocr.recognize_images_in_directory(image_file)
                full_text, confidences = trocr.get_text_with_confidence(outputs, processor)

                print("Подготавливаем результаты")
                await message.answer(f"\'{full_text}\' (уверенность: {confidences:.3f})")

            sent_count += 1

        except Exception as e:
            logging.error(f"Error sending {image_file.name}: {e}")

    await message.answer(f"✅ Отправлено {sent_count} результатов (из {len(image_files)})", reply_markup=kb.ocr)