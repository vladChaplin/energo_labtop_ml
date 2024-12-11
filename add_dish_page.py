import streamlit as st
import os
import json
import cv2
from PIL import Image
import numpy as np

annotations = []
image_counter = 1  # Счётчик для ID изображений
annotation_counter = 1  # Счётчик для ID аннотаций


def draw_annotations(img, annotations, preview_bbox=None):
    """
    Рисует аннотации на изображении. Если передан preview_bbox, также рисует предварительную рамку.
    """
    for annotation in annotations:
        x_min, y_min, width, height = annotation["bbox"]
        x_max, y_max = x_min + width, y_min + height
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    if preview_bbox:
        x_min, y_min, width, height = preview_bbox
        x_max, y_max = x_min + width, y_min + height
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Рамка для предварительного просмотра
    return img


def save_coco_annotations(output_path, annotations, categories, images):
    """
    Сохраняет аннотации в формате COCO.
    """
    coco_format = {
        "info": {"description": "Dataset for food recognition", "year": 2024},
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_format, f, indent=4)


def add_dish_page():
    global image_counter, annotation_counter

    st.title("Добавление блюда")

    # Категории блюд
    if "categories" not in st.session_state:
        st.session_state.categories = []

    category_name = st.text_input("Добавить категорию:")
    if st.button("Добавить категорию"):
        if category_name:
            st.session_state.categories.append({"id": len(st.session_state.categories) + 1, "name": category_name})
            st.success(f"Категория '{category_name}' добавлена!")

    st.write("Категории:", [c["name"] for c in st.session_state.categories])

    uploaded_file = st.file_uploader("Загрузите изображение блюда", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image_name = uploaded_file.name
        image_path = os.path.join("./dish_images", image_name)
        os.makedirs("./dish_images", exist_ok=True)

        # Сохраняем загруженное изображение
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Проверяем наличие категорий
        if len(st.session_state.categories) == 0:
            st.warning("Добавьте хотя бы одну категорию для начала разметки.")
            return

        category_options = [c["name"] for c in st.session_state.categories]
        selected_category = st.selectbox("Выберите категорию для объекта", category_options)

        st.write("Инструкция: Укажите параметры рамки для выделения области.")

        # Поля для ввода координат
        x_min = st.slider("X (верхний левый угол):", 0, image.shape[1], 0, step=1)
        y_min = st.slider("Y (верхний левый угол):", 0, image.shape[0], 0, step=1)
        width = st.slider("Ширина рамки:", 1, image.shape[1] - x_min, 50, step=1)
        height = st.slider("Высота рамки:", 1, image.shape[0] - y_min, 50, step=1)

        # Динамическое обновление изображения
        preview_bbox = [x_min, y_min, width, height]
        preview_image = draw_annotations(image.copy(), annotations, preview_bbox)
        st.image(preview_image, caption="Предварительный просмотр", use_column_width=True)

        # Добавление аннотации
        if st.button("Добавить рамку"):
            if selected_category:
                category_id = category_options.index(selected_category) + 1
                area = width * height
                annotations.append({
                    "id": annotation_counter,
                    "image_id": image_counter,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "area": area,
                    "iscrowd": 0
                })
                annotation_counter += 1
                st.success("Рамка добавлена!")
            else:
                st.error("Выберите категорию перед добавлением рамки.")

        # Обновлённое изображение с аннотациями
        final_image = draw_annotations(image.copy(), annotations)
        st.image(final_image, caption="Изображение с аннотациями", use_column_width=True)

        # Сохранение аннотаций
        if st.button("Сохранить аннотации"):
            images = [{
                "id": image_counter,
                "file_name": image_name,
                "width": image.shape[1],
                "height": image.shape[0]
            }]
            save_coco_annotations("dish_images/annotations.json", annotations, st.session_state.categories, images)
            image_counter += 1
            st.success("Аннотации сохранены в файл annotations.json!")
