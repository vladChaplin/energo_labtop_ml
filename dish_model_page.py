import os
import json
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Пути
IMAGE_DIR = "./dish_images"
ANNOTATION_FILE = "./dish_images/annotations.json"
MODEL_DIR = "./yolov8_training/food_recognition/weights"
MODEL_PATH = "./yolov8_training/food_recognition5/weights/best.pt"

os.makedirs(IMAGE_DIR, exist_ok=True)

# Функция для получения количества категорий из аннотаций
def get_number_of_categories():
    if not os.path.exists(ANNOTATION_FILE):
        return 0, []
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = data.get("categories", [])
    return len(categories), [c["name"] for c in categories]

# Функция для обучения модели
def train_model():
    if not os.path.exists(ANNOTATION_FILE):
        st.error("Файл аннотаций COCO (annotations.json) не найден.")
        return None

    num_categories, category_names = get_number_of_categories()
    if num_categories == 0:
        st.error("Аннотации отсутствуют или неверно отформатированы.")
        return None

    # Абсолютный путь к папке с изображениями
    image_dir_abs_path = os.path.abspath(IMAGE_DIR)

    # Создание YAML-файла конфигурации
    yaml_content = f"""
train: {image_dir_abs_path}
val: {image_dir_abs_path}
nc: {num_categories}
names: {category_names}
"""
    yaml_path = "./dataset.yaml"

    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_content)

    # Обучение модели
    model = YOLO("yolov8s.pt")
    st.info("Начинается обучение модели...")
    model.train(data=yaml_path, epochs=10, imgsz=640, save=True, project="yolov8_training", name="food_recognition")
    st.success("Обучение завершено!")
    model_path = "./yolov8_training/food_recognition/weights/best.pt"
    return model_path

# Функция для предсказания
def predict_image(model_path, image_path):
    model = YOLO(model_path)
    results = model.predict(source=image_path, save=True, save_txt=True)
    return results

# Основная страница работы с моделью блюд
def dish_model_page():
    st.title("Модель распознавания блюд")

    menu = ["Обучение модели", "Распознавание"]
    choice = st.radio("Выберите действие", menu)

    if choice == "Обучение модели":
        st.subheader("Обучение YOLOv8 на ваших данных")
        if st.button("Начать обучение"):
            model_path = train_model()
            if model_path:
                st.success(f"Модель успешно обучена! Сохранена в {model_path}")

    elif choice == "Распознавание":
        st.subheader("Распознавание блюд")
        if not os.path.exists(MODEL_PATH):
            st.warning("Сначала обучите модель в разделе 'Обучение модели'.")
        else:
            uploaded_file = st.file_uploader("Загрузите изображение блюда", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_path = os.path.join(IMAGE_DIR, "uploaded_image.jpg")
                image.save(image_path)

                st.image(image, caption="Загруженное изображение", use_column_width=True)

                if st.button("Распознать"):
                    results = predict_image(MODEL_PATH, image_path)
                    if results:
                        for r in results:
                            st.image(r.plot(), caption="Результат распознавания", use_column_width=True)
