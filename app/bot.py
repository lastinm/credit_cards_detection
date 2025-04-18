import asyncio
from aiogram import Bot,Dispatcher 
import logging

from handlers import router

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)  # или INFO
logger = logging.getLogger(__name__)

bot = Bot(token="7556992823:AAGfUzQIWrp7ZM2Y86C-attTJLSIshdgg88")
dp = Dispatcher()


async def main():
    dp.include_router(router)
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Ошибка: {e}")
    finally:
        await bot.session.close()
 
 
if __name__ == '__main__':
    asyncio.run(main())

 