from aiogram.types import ReplyKeyboardMarkup, KeyboardButton


detect = ReplyKeyboardMarkup(keyboard = [[KeyboardButton(text="Faster R-CNN"),
                                        KeyboardButton(text="YOLOv12")]],
                                        #[KeyboardButton(text="KerasOCR"),
                                        #KeyboardButton(text="EasyOCR")]],
                            resize_keyboard=True,
                            input_field_placeholder='Выбери способ детектирования...')

# catalog = [[KeyboardButton(text="Faster R-CNN"),
#                                         KeyboardButton(text="YOLOv12")],
#                                         [KeyboardButton(text="KerasOCR"),
#                                         KeyboardButton(text="EasyOCR")]], 
# 
ocr = ReplyKeyboardMarkup(keyboard = [[KeyboardButton(text="Faster R-CNN"),
                                        KeyboardButton(text="YOLOv12")],
                                        [KeyboardButton(text="KerasOCR"),
                                        KeyboardButton(text="EasyOCR"),
                                        KeyboardButton(text="TrOCR"),
                                        KeyboardButton(text="PaddleOCR")]],

                            resize_keyboard=True,
                            input_field_placeholder='Выбери способ детектирования или распознавания...')