import pandas as pd

# Загрузка данных
results_df = pd.read_csv('results.csv')

# Создаем список для хранения данных
data_records = []

# Обрабатываем каждое уникальное изображение
for image_name in results_df['image'].unique():
    # Фильтруем записи для текущего изображения
    image_records = results_df[results_df['image'] == image_name]
    
    # Создаем словарь для данных текущего изображения
    record = {'image': image_name}
    
    # Заполняем данные по каждому полю
    for field in ['CardHolder', 'CardNumber', 'DateExpired']:
        field_data = image_records[image_records['field_type'] == field]
        record[field] = field_data['true_text'].values[0] if not field_data.empty else ''
    
    # Добавляем словарь в список
    data_records.append(record)

# Создаем DataFrame из списка словарей
image_data_df = pd.DataFrame(data_records)

# Сохраняем результат
image_data_df.to_csv('image_data.csv', index=False)

print(f"Файл image_data.csv успешно создан. Записей: {len(image_data_df)}")
