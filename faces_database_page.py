import os
import cv2
import numpy as np
import streamlit as st
import face_recognition
from PIL import Image, ImageEnhance
import psycopg2
import uuid

# Предполагается, что эти константы и функции совпадают с теми, что в основном файле.
PG_HOST = "localhost"
PG_DB = "facesdb"
PG_USER = "postgres"
PG_PASSWORD = "0707"
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def get_connection():
    return psycopg2.connect(
        host=PG_HOST,
        database=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD
    )

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id SERIAL PRIMARY KEY,
        full_name TEXT NOT NULL,
        filename TEXT NOT NULL,
        img_path TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def add_user_to_db(full_name, filename, img_path):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO faces (full_name, filename, img_path) VALUES (%s, %s, %s)", (full_name, filename, img_path))
    conn.commit()
    conn.close()

def generate_variations(image: Image.Image, num_variations=5):
    """
    Простая функция генерации вариаций изображения:
    - Повороты на разные углы
    - Отражение по горизонтали
    - Изменение яркости
    Можно расширять по необходимости.
    """
    variations = []

    # Базовое изображение
    variations.append(image)

    # Повороты
    angles = [15, -15, 30, -30]
    for angle in angles:
        rotated = image.rotate(angle)
        variations.append(rotated)

    # Отражение по горизонтали
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    variations.append(flipped)

    # Изменение яркости
    enhancer = ImageEnhance.Brightness(image)
    brighter = enhancer.enhance(1.2)
    darker = enhancer.enhance(0.8)
    variations.append(brighter)
    variations.append(darker)

    # Ограничим общее число (например 5-10)
    # Если нужно строго num_variations, можно рандомизировать выбор.
    # Здесь просто возвращаем все сгенерированные.
    return variations[:num_variations]

def faces_database_page():
    st.title("База лиц")
    st.markdown("""
        Здесь вы можете загрузить фотографию лица и автоматически сгенерировать несколько вариаций
        (разные ракурсы, слегка изменённая яркость или отражения), чтобы улучшить точность модели.
    """)

    init_db()

    full_name = st.text_input("Введите ФИО человека:")
    uploaded_file = st.file_uploader("Загрузите изображение лица", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        # Проверка, что на изображении есть хотя бы одно лицо
        img_array = np.array(image)
        encodings = face_recognition.face_encodings(img_array)
        if len(encodings) == 0:
            st.warning("На изображении не найдено лицо. Загрузите другое фото.")
        else:
            st.success("Лицо обнаружено!")

            if st.button("Сгенерировать вариации и сохранить в базу"):
                if not full_name.strip():
                    st.error("ФИО не может быть пустым!")
                else:
                    base_name = full_name.strip()

                    variations = generate_variations(image, num_variations=5)
                    saved_count = 0
                    for i, var_img in enumerate(variations, start=1):
                        var_img_array = np.array(var_img.convert("RGB"))
                        # Ещё раз убедимся, что лицо есть
                        var_enc = face_recognition.face_encodings(var_img_array)
                        if len(var_enc) > 0:
                            # Сохраняем вариацию в файл
                            filename = f"{base_name}_{uuid.uuid4().hex}_{i}.jpg"
                            img_path = os.path.join(KNOWN_FACES_DIR, filename)
                            cv2.imwrite(img_path, cv2.cvtColor(var_img_array, cv2.COLOR_RGB2BGR))
                            add_user_to_db(base_name, filename, img_path)
                            saved_count += 1
                        else:
                            st.info(f"На вариации #{i} лицо не распознано, пропускаем.")

                    st.success(f"Добавлено {saved_count} вариаций лица для пользователя {base_name} в базу!")

# В отдельном основном файле (например main.py), вы можете импортировать и вызвать эту страницу при навигации:
# from faces_database_page import faces_database_page
#
# Затем добавить в навигацию:
# if page == "База лиц":
#     faces_database_page()
