from aiogram import Bot, F, types, Router
from aiogram.types import FSInputFile
from aiogram.filters import CommandStart, Command
import numpy as np
import imghdr       # Для проверки формата изображения
#import re
from pathlib import Path
import logging, os 
import pandas as pd
import dataframe_image as dfi
from tabulate import tabulate
from PIL import Image
                                 

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

            if confidence>0.50:
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

# Функция для добавления результатов распознавания
def add_ocr_result(df, ocr_system, class_name, recognized_text):
    # Создаем временный столбец если его нет
    col_name = f'text_{ocr_system}'
    if col_name not in df.columns:
        df[col_name] = None
    
    # Находим индекс для добавления данных
    mask = (df['Class'] == class_name)
    if mask.any():
        df.loc[mask, col_name] = recognized_text
    else:
        # Если нет подходящей строки, добавляем новую
        new_row = {'Class': class_name, col_name: recognized_text}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    return df

# Собрать несколько изображений в одно
def combine_images_with_table(table_path, image_data, output_path, padding=10):
    """
    Объединяет таблицу и изображения в одно вертикальное изображение
    
    :param table_path: путь к изображению таблицы (str/Path)
    :param image_data: список кортежей (PosixPath, class_name, confidence)
    :param output_path: путь для сохранения (str/Path)
    :param padding: отступ между изображениями (пиксели)
    """
    # 1. Преобразуем пути и извлекаем данные
    def process_image_data(item):
        path, class_name, confidence = item
        return {
            'path': str(Path(path).absolute()),
            'class': class_name,
            'confidence': float(confidence)
        }
    
    try:
        # Обработка входных данных
        table_path = str(Path(table_path).absolute())
        output_path = str(Path(output_path).absolute())
        processed_images = [process_image_data(item) for item in image_data]
        
        # 2. Проверка существования файлов
        missing = [img['path'] for img in processed_images if not os.path.exists(img['path'])]
        if not os.path.exists(table_path):
            missing.append(table_path)
            
        if missing:
            raise FileNotFoundError(f"Отсутствуют файлы: {missing}")
        
        # 3. Загрузка изображений с подписями
        images = []
        
        # Сначала добавляем таблицу
        with Image.open(table_path) as img:
            images.append({
                'image': img.copy(),
                'label': "OCR Results Table"
            })
        
        # Затем добавляем обработанные изображения
        for img_data in processed_images:
            try:
                with Image.open(img_data['path']) as img:
                    label = f"{img_data['class']} (Confidence: {img_data['confidence']:.2f})"
                    images.append({
                        'image': img.copy(),
                        'label': label
                    })
            except Exception as e:
                print(f"Ошибка загрузки {img_data['path']}: {str(e)}")
                continue
        
        if len(images) <= 1:  # Только таблица
            raise ValueError("Нет изображений для обработки")
        
        # 4. Обработка изображений
        max_width = max(img['image'].width for img in images)
        
        # Функция для добавления подписи
        def add_caption(base_img, text):
            from PIL import ImageDraw, ImageFont
            font = ImageFont.load_default()
            caption_height = 30
            new_img = Image.new('RGB', 
                              (base_img.width, base_img.height + caption_height),
                              (255, 255, 255))
            new_img.paste(base_img, (0, 0))
            
            draw = ImageDraw.Draw(new_img)
            text_width = draw.textlength(text, font=font)
            draw.text(((base_img.width - text_width) // 2, base_img.height + 5),
                     text, fill="black", font=font)
            return new_img
        
        # Подготовка изображений с подписями
        processed = []
        for img_data in images:
            img = img_data['image']
            if img.width != max_width:
                new_height = int(img.height * (max_width / img.width))
                img = img.resize((max_width, new_height), Image.LANCZOS)
            
            # Добавляем подпись
            labeled_img = add_caption(img, img_data['label'])
            processed.append(labeled_img)
        
        # 5. Создание итогового изображения
        total_height = sum(img.height for img in processed) + padding * (len(processed)-1)
        combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for img in processed:
            combined.paste(img, (0, y_offset))
            y_offset += img.height + padding
        
        # 6. Сохранение результата
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        combined.save(output_path)
        print(f"Результат сохранен в {output_path}")
        return True
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False


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


@router.message(F.text == 'TrOCR + PaddleOCR')
async def ensemble_OCR(message: types.Message, bot: Bot):
    image_files = utils.get_list_of_images()

    if not image_files:
        logging.error(f"Нет изображений с корректным форматом имени.")
        return

    sent_count = 0
    # try:
    for image_file, class_name, confidence in image_files:
        try:
            if class_name == 'DateExpired':
                outputs, processor = trocr.recognize_images_in_directory(image_file)
                full_text, confidences = trocr.get_text_with_confidence(outputs, processor)

                print("Подготавливаем результаты")
                await message.answer(f"\'{full_text}\' (уверенность: {confidences:.3f})")
            else:
                full_text, confidences = paddleocr.recognize_images_in_directory(image_file)

                print("Подготавливаем результаты")
                await message.answer(f"\'{full_text}\' (уверенность: {confidences:.3f})")

            sent_count += 1

        except Exception as e:
            logging.error(f"Error sending {image_file.name}: {e}")

    await message.answer(f"✅ Отправлено {sent_count} результатов (из {len(image_files)})", reply_markup=kb.ocr)


@router.message(F.text == 'Сравнить OCR')
async def compare_OCR(message: types.Message, bot: Bot):
    # Создаем DataFrame для результатов
    columns = [
        'Class'      # Распознаваемое поле
    ]
    df = pd.DataFrame(columns=columns)

    image_files = utils.get_list_of_images()

    if not image_files:
        logging.error(f"Нет изображений с корректным форматом имени.")
        return
    
    for image_file, class_name, confidence in image_files:
        #logging.INFO(f"Передаем в KerasOCR файл: {image_file}.")
        await message.answer(f"Подождите идет распознавание KerasOCR класса: {class_name}", reply_markup=kb.ocr)
        try:
            full_text, predictions = kerasocr.recognize_with_confidence(image_file)
            # Добавляем результат
            df = add_ocr_result(df, 'KerasOCR', class_name, full_text)
        except Exception as e:
            logging.error(f"Ошибка распознавания: {image_file}: {e}")

        # logging.INFO(f"Передаем в EasyOCR файл: {image_file.name}.")
        await message.answer(f"Подождите идет распознавание EasyOCR класса: {class_name}", reply_markup=kb.ocr)
        try:
            img_path, results, class_id, processed_img = easyocr.recognize_images_in_directory(image_file, languages=['en', 'ru'], gpu=False)
            # Собираем результаты
            result_text = []
            for i, (_, text, prob) in enumerate(results):
                #print(f"{i+1}. {text} (точность: {prob:.2f})")
                result_text.append(text)
            full_text = ''.join(result_text)
            # Добавляем результат
            df = add_ocr_result(df, 'EasyOCR', class_name, full_text)
        except Exception as e:
            logging.error(f"Ошибка распознавания: {image_file.name}: {e}")

        # logging.INFO(f"Передаем в TrOCR файл: {image_file.name}.")
        await message.answer(f"Подождите идет распознавание TrOCR класса: {class_name}", reply_markup=kb.ocr)
        try:
            outputs, processor = trocr.recognize_images_in_directory(image_file)
            full_text, confidences = trocr.get_text_with_confidence(outputs, processor)

            # Добавляем результат
            df = add_ocr_result(df, 'TrOCR', class_name, full_text)
        except Exception as e:
            logging.error(f"Ошибка распознавания: {image_file.name}: {e}")

        #logging.INFO(f"Передаем в PaddleOCR файл: {image_file.name}.")
        await message.answer(f"Подождите идет распознавание PaddleOCR класса: {class_name}", reply_markup=kb.ocr)       
        try:
            full_text, confidences = paddleocr.recognize_images_in_directory(image_file)

            # Добавляем результат
            df = add_ocr_result(df, 'PaddleOCR', class_name, full_text)
        except Exception as e:
            logging.error(f"Ошибка распознавания: {image_file.name}: {e}")

     
            # Добавляем результат
            df = add_ocr_result(df, 'TrOCR', class_name, full_text)
        except Exception as e:
            logging.error(f"Ошибка распознавания: {image_file.name}: {e}")   #         # logging.INFO(f"Передаем в TrOCR файл: {image_file.name}.")
        try:
            outputs, processor = trocr.recognize_images_in_directory(image_file)
            full_text, confidences = trocr.get_text_with_confidence(outputs, processor)

            # Добавляем результат
            df = add_ocr_result(df, 'TrOCR', class_name, full_text)
        except Exception as e:
            logging.error(f"Ошибка распознавания: {image_file.name}: {e}")

        #logging.INFO(f"Передаем в ансамбль файл: {image_file.name}.")
        await message.answer(f"Подождите идет распознавание 'PaddleOCR + TrOCR' класса: {class_name}", reply_markup=kb.ocr)    
        try:
            if class_name == 'CardNumber':
                full_text, confidences = paddleocr.recognize_images_in_directory(image_file)
            else:
                outputs, processor = trocr.recognize_images_in_directory(image_file)
                full_text, confidences = trocr.get_text_with_confidence(outputs, processor)

            # Добавляем результат
            df = add_ocr_result(df, 'Ansible', class_name, full_text)
        except Exception as e:
            logging.error(f"Ошибка распознавания: {image_file.name}: {e}")

    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

    table_image = "ocr_results_table.png"
    output_file = "combined_result.png"
    dfi.export(df, table_image, table_conversion='matplotlib')
    combine_images_with_table(table_image, image_files, output_file, padding=20)

    await message.answer_photo(FSInputFile(output_file, filename="OCR processed.jpg"))

