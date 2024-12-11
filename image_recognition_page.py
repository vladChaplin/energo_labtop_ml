import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import face_recognition

# Предполагается, что эти константы и функции совпадают с теми, что в основном файле
PG_HOST = "localhost"
PG_DB = "facesdb"
PG_USER = "postgres"
PG_PASSWORD = "0707"
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
FONT_PATH = "DejaVuSans.ttf"  # Убедитесь, что файл существует


def get_connection():
    import psycopg2
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


def get_all_users():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT full_name, filename, img_path FROM faces")
    results = c.fetchall()
    conn.close()
    return results


def load_known_faces():
    known_encodings = []
    known_names = []
    users = get_all_users()
    for full_name, filename, img_path in users:
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(full_name)
    return known_encodings, known_names


def image_recognition_page():
    st.title("Распознавание лица по загруженному изображению")
    st.markdown("""
        Загрузите изображение с лицами.  
        Программа попытается распознать лица:  
        - Известные: зелёная рамка и ФИО + точность  
        - Неизвестные: красная рамка и надпись "Неизвестное".
    """)

    init_db()
    if "known_encodings" not in st.session_state or "known_names" not in st.session_state:
        st.session_state.known_encodings, st.session_state.known_names = load_known_faces()

    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Исходное изображение", use_column_width=True)

        # Переведём в numpy для face_recognition
        img_array = np.array(image)

        face_locations = face_recognition.face_locations(img_array)
        face_encodings = face_recognition.face_encodings(img_array, face_locations)

        if len(face_locations) == 0:
            st.warning("Лицо не обнаружено на изображении.")
        else:
            pil_image = image.copy()
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(FONT_PATH, 20)

            known_encodings = st.session_state.known_encodings
            known_names = st.session_state.known_names

            tolerance = 0.6
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Считаем расстояния до всех известных лиц
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    best_match_index = np.argmin(distances)
                    if min_dist < tolerance:
                        name = known_names[best_match_index]
                        # Рассчитаем "точность" как (1 - dist)*100
                        confidence = (1.0 - min_dist) * 100
                        rect_color = (0, 255, 0)  # Зеленый
                        label = f"{name} ({confidence:.2f}%)"
                    else:
                        rect_color = (255, 0, 0)  # Красный
                        label = "Неизвестное"
                else:
                    # Нет известных лиц в базе
                    rect_color = (255, 0, 0)
                    label = "Неизвестное"

                # Рисуем прямоугольник
                draw.rectangle([(left, top), (right, bottom)], outline=rect_color, width=2)
                # Пишем текст над лицом
                text_x, text_y = left, top - 30
                if text_y < 0:
                    text_y = bottom + 5
                draw.text((text_x, text_y), label, font=font, fill=rect_color)

            st.image(pil_image, caption="Результат распознавания", use_column_width=True)
