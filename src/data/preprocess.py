# src/data/preprocess.py

import argparse
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Простая функция для очистки текста."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Приводим к нижнему регистру
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Удаляем все, кроме букв и пробелов
    return text

def main(args):
    """Главная функция для выполнения предобработки."""
    print("Loading raw data...")
    # Загружаем исходный CSV-файл
    df = pd.read_csv(args.input_path, header=None, names=['sentiment', 'title', 'text'])

    print("Combining title and text...")
    # Объединяем заголовок и текст отзыва в одну колонку
    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    print("Cleaning text data...")
    # Применяем функцию очистки
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Конвертируем метки: 1 -> 0 (negative), 2 -> 1 (positive)
    # Модели обычно лучше работают с метками, начинающимися с 0
    df['sentiment'] = df['sentiment'].apply(lambda x: 0 if x == 1 else 1)

    # Выбираем только нужные колонки
    processed_df = df[['sentiment', 'cleaned_text']]
    
    print("Splitting data into train and validation sets...")
    # Разделяем данные на обучающую (80%) и валидационную (20%) выборки
    # stratify=processed_df['sentiment'] гарантирует, что в обеих выборках
    # будет одинаковое соотношение позитивных и негативных отзывов.
    train_df, val_df = train_test_split(
        processed_df, 
        test_size=0.2, 
        random_state=42, # random_state для воспроизводимости
        stratify=processed_df['sentiment']
    )

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Создаем выходную директорию, если она не существует
    os.makedirs(args.output_path, exist_ok=True)
    
    # Сохраняем обработанные данные
    train_df.to_csv(os.path.join(args.output_path, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_path, 'validation.csv'), index=False)
    
    print(f"Processed data saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess raw review data.")
    parser.add_argument('--input-path', type=str, required=True, help='Path to the raw train.csv file.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the directory to save processed data.')
    
    args = parser.parse_args()
    main(args)