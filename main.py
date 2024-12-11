import os
import cv2
import numpy as np
import psycopg2
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import time
import face_recognition

from faces_database_page import faces_database_page
from image_recognition_page import image_recognition_page
from dish_model_page import dish_model_page

# Константы для подключения к PostgreSQL
PG_HOST = "localhost"
PG_DB = "facesdb"
PG_USER = "postgres"
PG_PASSWORD = "0707"

# Путь для сохранения известных лиц
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Путь к шрифту с поддержкой кириллицы (убедитесь, что файл DejaVuSans.ttf есть)
FONT_PATH = "DejaVuSans.ttf"

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
        img = face_recognition.load_image_file(img_path)  # Возвращает RGB-массив (H,W,3)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(full_name)
    return known_encodings, known_names

def recognize_face_in_frame(frame, known_encodings, known_names, tolerance=0.6):
    # Проверим, что кадр получен
    if frame is None:
        return []

    # Проверим тип и формат
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Используем cvtColor для преобразования в RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Отладочная информация
    st.write("rgb_frame dtype:", rgb_frame.dtype, "shape:", rgb_frame.shape)

    # Проверим, что форма правильная
    if len(rgb_frame.shape) != 3 or rgb_frame.shape[2] != 3:
        return []

    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) == 0:
        return []
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        name = "Неизвестный"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        results.append((name, (left, top, right, bottom)))
    return results

def add_logo():
    st.sidebar.image("logo_labtop.jpg", use_container_width=True)
    st.image("logo_labtop.jpg", width=150)

def main_page():
    st.title("Технология распознавания лиц для оплаты")
    st.markdown("""
    Добро пожаловать!  
    Перейдите на вкладку "Распознать лицо" или "Онлайн распознавание" чтобы начать.
    """)

def face_recognition_page():
    st.title("Регистрация нового пользователя")
    st.markdown("""
        Для более точного распознавания лица нам нужны несколько кадров с разными ракурсами.  
        Пожалуйста, следуйте инструкциям:
        1. Поверните голову направо.
        2. Поверните голову налево.
        3. Посмотрите прямо в камеру.
        4. Посмотрите вверх.
        5. Посмотрите вниз.

        На каждом шаге нажмите "Сделать кадр". Программа подождёт 5 секунд, после чего сделает снимок.
    """)

    init_db()

    if "known_encodings" not in st.session_state or "known_names" not in st.session_state:
        st.session_state.known_encodings, st.session_state.known_names = load_known_faces()

    head_positions = [
        "Поверните голову направо",
        "Поверните голову налево",
        "Посмотрите прямо в камеру",
        "Посмотрите вверх",
        "Посмотрите вниз"
    ]

    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    if "unrecognized_frames" not in st.session_state:
        st.session_state.unrecognized_frames = []

    if st.session_state.current_step < len(head_positions):
        instruction = head_positions[st.session_state.current_step]
        st.info(f"Шаг {st.session_state.current_step + 1}/{len(head_positions)}: {instruction}")

        if st.button("Сделать кадр"):
            st.info("Ожидаем 2 секунд...")
            time.sleep(2)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Камера не обнаружена.")
                return
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                st.error("Не удалось получить кадр с камеры.")
                return

            results = recognize_face_in_frame(frame, st.session_state.known_encodings, st.session_state.known_names)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(FONT_PATH, 20)

            found_new_face = False
            if len(results) == 0:
                st.warning("Лицо не обнаружено, попробуйте ещё раз.")
            else:
                for (name, (left, top, right, bottom)) in results:
                    if name == "Неизвестный":
                        face_image = frame[top:bottom, left:right]
                        st.session_state.unrecognized_frames.append(face_image)
                        found_new_face = True
                        rect_color = (255, 0, 0)
                    else:
                        rect_color = (0, 255, 0)

                    draw.rectangle([(left, top), (right, bottom)], outline=rect_color, width=2)
                    draw.text((left, top - 30), name, font=font, fill=rect_color)

                final_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                st.image(final_frame, channels="BGR", caption="Результат этого шага")

                if found_new_face:
                    st.success("Кадр сохранён для дальнейшего добавления в базу.")
                    st.session_state.current_step += 1
                else:
                    st.info("Лицо известно, для демонстрации нужен незнакомый пользователь.")
    else:
        # Все шаги выполнены, просим ввести ФИО и сохранить в БД
        st.success("Все ракурсы собраны! Введите ФИО и добавьте в базу.")
        with st.form(key="add_user_form"):
            full_name = st.text_input("Введите ФИО для нового пользователя:")
            submitted = st.form_submit_button("Добавить в базу")
            if submitted:
                if not full_name.strip():
                    st.error("ФИО не может быть пустым!")
                else:
                    base_name = full_name.strip()
                    for i, uf in enumerate(st.session_state.unrecognized_frames, start=1):
                        filename = f"{base_name}_{i}.jpg"
                        img_path = os.path.join(KNOWN_FACES_DIR, filename)
                        res = cv2.imwrite(img_path, uf)
                        if res:
                            add_user_to_db(base_name, filename, img_path)
                        else:
                            st.error(f"Не удалось сохранить кадр #{i}.")

                    st.success(f"Пользователь {full_name} добавлен в базу!")
                    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
                    st.session_state.unrecognized_frames = []
                    st.session_state.current_step = 0

def live_recognition_page():
    st.title("Онлайн распознавание")
    st.markdown("Нажмите 'Обновить кадр' для захвата нового кадра и попытки распознать лицо.")

    init_db()
    if "known_encodings" not in st.session_state or "known_names" not in st.session_state:
        st.session_state.known_encodings, st.session_state.known_names = load_known_faces()

    if st.button("Обновить кадр"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Камера не обнаружена.")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            st.error("Не удалось получить изображение с камеры.")
            return

        results = recognize_face_in_frame(frame, st.session_state.known_encodings, st.session_state.known_names)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(FONT_PATH, 20)

        if len(results) == 0:
            st.warning("Лицо не обнаружено. Попробуйте снова.")
        else:
            recognized_someone = False
            for (name, (left, top, right, bottom)) in results:
                if name != "Неизвестный":
                    rect_color = (0, 255, 0)
                    recognized_someone = True
                else:
                    rect_color = (255, 0, 0)

                draw.rectangle([(left, top), (right, bottom)], outline=rect_color, width=2)
                draw.text((left, top - 30), name, font=font, fill=rect_color)

            final_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            st.image(final_frame, channels="BGR", caption="Результат распознавания")

            if not recognized_someone:
                st.warning("Лицо не распознано! Хотите добавить в базу?")
                if st.button("Перейти к добавлению нового лица"):
                    st.experimental_set_query_params(page="Распознать%20лицо")
                    st.info("Переключитесь на страницу 'Распознать лицо' в меню слева для добавления нового пользователя.")

# Навигация
st.sidebar.title("Навигация")
add_logo()
page = st.sidebar.radio("Выберите страницу", ["Главная", "Распознать лицо", "Онлайн распознавание", "База лиц", "Распознать на изображении", "Добавление блюда", "Модель блюд"])

if page == "Главная":
    main_page()
elif page == "Распознать лицо":
    face_recognition_page()
elif page == "Онлайн распознавание":
    live_recognition_page()
elif page == "База лиц":
    faces_database_page()
elif page == "Распознать на изображении":
    image_recognition_page()
elif page == "Добавление блюда":
    from add_dish_page import add_dish_page
    add_dish_page()
elif page == "Модель блюд":
    dish_model_page()

