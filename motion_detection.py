import cv2
import numpy as np
import os
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
# Определяет класс крыса ходит/стоит
class RatBehaviorClassifier:
    def __init__(self, initial_bboxes, frame_width, frame_height):
        self.frame_width = frame_width  # Ширина кадра видео
        self.frame_height = frame_height  # Высота кадра видео
        self.kalman = self.initialize_kalman_with_first_frames(initial_bboxes, frame_width, frame_height)
        self.bbox_history = []  # История ограничивающих рамок для отслеживания движения
        self.classification_history = []  # История классификаций поведения
        self.movement_threshold = 2  # Порог для величины вектора скорости, чтобы определить движение

    def initialize_kalman_filter(self):
        # Инициализация фильтра Калмана
        kalman = cv2.KalmanFilter(6, 4)  # Создание фильтра Калмана с 6 состояниями и 4 измерениями
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], np.float32)  # Матрица измерений
        kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)  # Матрица перехода
        kalman.processNoiseCov = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32) * 0.017  # Ковариация шума процесса
        kalman.measurementNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32) * 10  # Ковариация шума измерений
        return kalman

    def initialize_kalman_with_first_frames(self, initial_bboxes, frame_width, frame_height, num_frames=6):
        # Инициализация фильтра Калмана с использованием начальных кадров
        x, y, width, height = 0, 0, 0, 0  # Начальные координаты и размеры ограничивающей рамки
        for bbox in initial_bboxes:
            x += bbox[0] * frame_width
            y += bbox[1] * frame_height
            width += (bbox[2] - bbox[0]) * frame_width
            height += (bbox[3] - bbox[1]) * frame_height
        x /= num_frames
        y /= num_frames
        width /= num_frames
        height /= num_frames

        kalman = self.initialize_kalman_filter()
        kalman.statePre = np.array([[x], [y], [width], [height], [0], [0]], np.float32)  # Начальное состояние
        kalman.statePost = np.array([[x], [y], [width], [height], [0], [0]], np.float32)  # Постобработанное состояние
        return kalman

    def filter_step(self, bbox):
        # Обновление фильтра Калмана и предсказание новой ограничивающей рамки
        x1, y1, x2, y2 = bbox  # Координаты ограничивающей рамки
        x1, y1, x2, y2 = int(x1 * self.frame_width), int(y1 * self.frame_height), int(x2 * self.frame_width), int(y2 * self.frame_height)
        measurement = np.array([[x1], [y1], [x2 - x1], [y2 - y1]], np.float32)  # Измерение для фильтра Калмана
        self.kalman.correct(measurement)  # Коррекция фильтра Калмана
        prediction = self.kalman.predict()  # Предсказание нового состояния
        x1, y1, width, height = int(prediction[0][0]), int(prediction[1][0]), int(prediction[2][0]), int(prediction[3][0])
        x2, y2 = x1 + width, y1 + height

        vx, vy = prediction[4][0], prediction[5][0]  # Скорости по осям x и y
        velocity_vector = np.array([vx, vy])  # Вектор скорости

        return [x1, y1, x2, y2], velocity_vector

    def clsf_step(self, velocity_vector):
        # Классификация поведения крысы на основе вектора скорости
        if len(self.bbox_history) < 3:
            self.bbox_history.append(velocity_vector)
            return 0

        velocity_magnitude = np.linalg.norm(velocity_vector)  # Величина вектора скорости
        is_walking = velocity_magnitude > self.movement_threshold  # Флаг, указывающий, движется ли крыса

        self.bbox_history.append(velocity_vector)
        if len(self.bbox_history) > 5:
            self.bbox_history.pop(0)  # Удаление старых записей

        self.classification_history.append(1 if is_walking else 0)  # Добавление результата классификации
        if len(self.classification_history) > 10:
            self.classification_history.pop(0)  # Удаление старых записей

        if len(self.classification_history) >= 5:
            self.classification_history[-3] = round((self.classification_history[-1] + self.classification_history[-2] + self.classification_history[-3] + self.classification_history[-4] + self.classification_history[-5]) / 5)
            filtered_history = median_filter(self.classification_history, size=3)  # Фильтрация истории классификаций
            self.classification_history = list(filtered_history)
            return self.classification_history[-3]
        else:
            return self.classification_history[-1]

    def step(self, bbox):
        # Основной шаг обработки кадра
        filtered_bbox, velocity_vector = self.filter_step(bbox)  # Обновление фильтра Калмана
        classification_result = self.clsf_step(velocity_vector)  # Классификация поведения
        return filtered_bbox, classification_result, velocity_vector
# Определяет класс крыса встала на задние лапы
class RatLegsClassifier:
    def __init__(self):
        self.model = joblib.load('random_forest_model.joblib')  # Загрузка модели машинного обучения
        self.scaler = StandardScaler()  # Инициализация нормализатора
        self.classification_history = []  # История классификаций

    def calculate_angles_between_frames(self, points_history):
        # Вычисление углов между векторами в последовательных кадрах
        angles_dict = {
            '0-3': [],
            '3-4': [],
            '4-7': [],
            '0-7': []
        }  # Словарь для хранения углов
        min_length = min(len(v) for v in points_history.values())  # Минимальная длина истории точек

        for i in range(1, min_length):
            if 0 in points_history and 3 in points_history:
                vector_prev = np.array(points_history[3][i-1]) - np.array(points_history[0][i-1])  # Вектор в предыдущем кадре
                vector_curr = np.array(points_history[3][i]) - np.array(points_history[0][i])  # Вектор в текущем кадре
                angle = np.arctan2(np.linalg.det([vector_prev, vector_curr]), np.dot(vector_prev, vector_curr))  # Угол между векторами
                angles_dict['0-3'].append(np.degrees(angle))  # Добавление угла в словарь

            if 3 in points_history and 4 in points_history:
                vector_prev = np.array(points_history[4][i-1]) - np.array(points_history[3][i-1])
                vector_curr = np.array(points_history[4][i]) - np.array(points_history[3][i])
                angle = np.arctan2(np.linalg.det([vector_prev, vector_curr]), np.dot(vector_prev, vector_curr))
                angles_dict['3-4'].append(np.degrees(angle))

            if 4 in points_history and 7 in points_history:
                vector_prev = np.array(points_history[7][i-1]) - np.array(points_history[4][i-1])
                vector_curr = np.array(points_history[7][i]) - np.array(points_history[4][i])
                angle = np.arctan2(np.linalg.det([vector_prev, vector_curr]), np.dot(vector_prev, vector_curr))
                angles_dict['4-7'].append(np.degrees(angle))

            if 0 in points_history and 7 in points_history:
                vector_prev = np.array(points_history[7][i-1]) - np.array(points_history[0][i-1])
                vector_curr = np.array(points_history[7][i]) - np.array(points_history[0][i])
                angle = np.arctan2(np.linalg.det([vector_prev, vector_curr]), np.dot(vector_prev, vector_curr))
                angles_dict['0-7'].append(np.degrees(angle))

        return angles_dict

    def calculate_distances(self, points):
        # Вычисление расстояний между точками в последовательных кадрах
        distances = []  # Список для хранения расстояний
        for i in range(1, len(points)):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))  # Расстояние между точками
            distances.append(distance)
        return distances

    def calculate_specific_distances(self, keypoints_history):
        # Вычисление специфических расстояний между ключевыми точками
        distances_dict = {
            '0-3': [],
            '3-4': [],
            '4-7': [],
            '0-7': [],
            '6-7': [],
            '5-7': []
        }  # Словарь для хранения расстояний
        min_length = min(len(v) for v in keypoints_history.values())  # Минимальная длина истории ключевых точек

        for i in range(min_length):
            if 0 in keypoints_history and 3 in keypoints_history:
                distances_dict['0-3'].append(np.linalg.norm(np.array(keypoints_history[0][i]) - np.array(keypoints_history[3][i])))  # Расстояние между точками
            if 3 in keypoints_history and 4 in keypoints_history:
                distances_dict['3-4'].append(np.linalg.norm(np.array(keypoints_history[3][i]) - np.array(keypoints_history[4][i])))
            if 4 in keypoints_history and 7 in keypoints_history:
                distances_dict['4-7'].append(np.linalg.norm(np.array(keypoints_history[4][i]) - np.array(keypoints_history[7][i])))
            if 0 in keypoints_history and 7 in keypoints_history:
                distances_dict['0-7'].append(np.linalg.norm(np.array(keypoints_history[0][i]) - np.array(keypoints_history[7][i])))
            if 6 in keypoints_history and 7 in keypoints_history:
                distances_dict['6-7'].append(np.linalg.norm(np.array(keypoints_history[6][i]) - np.array(keypoints_history[7][i])))
            if 5 in keypoints_history and 7 in keypoints_history:
                distances_dict['5-7'].append(np.linalg.norm(np.array(keypoints_history[5][i]) - np.array(keypoints_history[7][i])))

        return distances_dict

    def calculate_speeds(self, points, fps):
        # Вычисление скоростей между точками в последовательных кадрах
        speeds = []  # Список для хранения скоростей
        for i in range(1, len(points)):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))  # Расстояние между точками
            time = 1 / fps  # Время между кадрами
            speed = distance / time  # Скорость
            speeds.append(speed)
        return speeds

    def calculate_velocity_changes(self, velocity_vectors):
        # Вычисление изменений в направлении вектора скорости
        velocity_changes = []  # Список для хранения изменений направления
        for i in range(1, len(velocity_vectors)):
            angle_between_vectors = np.arccos(np.clip(np.dot(velocity_vectors[i-1].flatten(), velocity_vectors[i].flatten()) / (np.linalg.norm(velocity_vectors[i-1]) * np.linalg.norm(velocity_vectors[i])), -1.0, 1.0))  # Угол между векторами
            velocity_changes.append(np.degrees(angle_between_vectors))
        return velocity_changes

    def data_step(self, keypoints_dict, fps, velocity_vectors, classification_results):
        # Подготовка данных для модели классификации
        angles_dict = self.calculate_angles_between_frames(keypoints_dict)  # Вычисление углов
        distances_dict = {key: self.calculate_distances(points) for key, points in keypoints_dict.items()}  # Вычисление расстояний
        specific_distances_dict = self.calculate_specific_distances(keypoints_dict)  # Вычисление специфических расстояний
        speeds_dict = {key: self.calculate_speeds(points, fps) for key, points in keypoints_dict.items()}  # Вычисление скоростей
        velocity_changes = self.calculate_velocity_changes(velocity_vectors)  # Вычисление изменений в направлении вектора скорости

        data = {
            'velocity_changes': [velocity_changes[-1]] if velocity_changes else [np.nan],  # Последнее изменение направления
            'classification_results': [classification_results],  # Результат классификации
        }

        for key, angles in angles_dict.items():
            data[f'angle_{key}'] = [angles[-1]] if angles else [np.nan]  # Последний угол

        for key, distances in distances_dict.items():
            data[f'distance_{key}'] = [distances[-1]] if distances else [np.nan]  # Последнее расстояние

        for key, distances in specific_distances_dict.items():
            data[f'specific_distance_{key}'] = [distances[-1]] if distances else [np.nan]  # Последнее специфическое расстояние

        for key, speeds in speeds_dict.items():
            data[f'speed_{key}'] = [speeds[-1]] if speeds else [np.nan]  # Последняя скорость

        data = pd.DataFrame(data)  # Преобразование данных в DataFrame
        data = data.dropna().fillna(0)  # Удаление пустых значений
        return data

    def clsf_step(self, data):
        # Классификация поведения крысы на основе подготовленных данных
        classification_result = self.model.predict(data)  # Предсказание модели

        self.classification_history.append(classification_result[0])  # Добавление результата классификации
        if len(self.classification_history) > 5:
            self.classification_history.pop(0)  # Удаление старых записей
            self.classification_history[-3] = round((self.classification_history[-1] + self.classification_history[-2] + self.classification_history[-3] + self.classification_history[-4] + self.classification_history[-5]) / 5)
            return self.classification_history[-3]
        else:
            return self.classification_history[-1]

    def step(self, keypoints_dict, fps, velocity_vector, classification_results):
        # Основной шаг обработки кадра
        data = self.data_step(keypoints_dict, fps, velocity_vector, classification_results)  # Подготовка данных
        classification_result = self.clsf_step(data)  # Классификация поведения

        return classification_result
# Связывает два класса классификатора, считывает входные данные и визуализирует результаты классификации
class RatBehaviorVisualizer:
    def __init__(self):
        self.i_const = 0  # Константа для индексации
        self.current_video_number = 0  # Переменная для отслеживания текущего номера видео

    def count_files_in_directory(self, directory_path):
        # Подсчет файлов в указанной директории
        try:
            items = os.listdir(directory_path)  # Список элементов в директории
            files = [item for item in items if os.path.isfile(os.path.join(directory_path, item))]  # Список файлов
            return len(files)  # Возвращает количество файлов
        except Exception as e:
            print(f"Ошибка: {e}")
            return 0

    def read_annotations(self, file_path):
        # Чтение аннотаций из файла
        annotations = []  # Список для хранения аннотаций
        with open(file_path, 'r') as file:
            dimensions = list(map(int, file.readline().strip().split()))  # Размеры кадра
            frame_width, frame_height = dimensions[0], dimensions[1]  # Ширина и высота кадра
            for line in file:
                data = list(map(float, line.strip().split()))  # Данные из строки
                frame_number = int(data[0])  # Номер кадра
                bbox = data[1:5]  # Координаты ограничивающей рамки
                skeleton_points = data[5:]  # Ключевые точки скелета
                annotations.append((frame_number, bbox, skeleton_points))  # Добавление аннотации в список
        return annotations, frame_width, frame_height

    def visualize_annotations_on_video(self, video_path, annotations, output_path_template, frame_width, frame_height):
        # Визуализация аннотаций на видео
        cap = cv2.VideoCapture(video_path)  # Захват видео
        self.rat_legs_classifier = RatLegsClassifier()  # Инициализация классификатора
        if not cap.isOpened():
            print("Ошибка: не удалось открыть видеофайл.")
            return

        fps = 30  # Установим fps как константу
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для записи видео
        output_path = output_path_template.format(classification="sitting")  # Шаблон для пути выходного файла
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))  # Запись видео

        frame_index = 0  # Индекс текущего кадра
        keypoints_history = {0: [], 3: [], 4: [], 5: [], 6: [], 7: []}  # История ключевых точек
        velocity_vectors = []  # Векторы скорости

        initial_bboxes = [annotations[i][1] for i in range(6)]  # Начальные ограничивающие рамки
        classifier = RatBehaviorClassifier(initial_bboxes, frame_width, frame_height)  # Инициализация классификатора

        while cap.isOpened() and frame_index < len(annotations):
            ret, frame = cap.read()  # Чтение кадра
            if not ret:
                break

            frame_number, bbox, skeleton_points = annotations[frame_index]  # Данные текущего кадра
            filtered_bbox, classification_walking, velocity_vector = classifier.step(bbox)  # Обработка кадра

            if frame_index >= 3:

                velocity_vectors.append(velocity_vector)  # Добавление вектора скорости
                x1, y1, x2, y2 = filtered_bbox  # Координаты ограничивающей рамки
                classification_text = "walking" if classification_walking == 1 else "sitting"  # Текст классификации
                if frame_index > 4:
                    # Создание нового словаря с последними значениями из keypoints_history
                    last_keypoints_history = {key: values[-2:] for key, values in keypoints_history.items() if values}
                    last_velocity_vectors = velocity_vectors[-2:]  # Последние векторы скорости
                    # Определение значения classification_leg
                    class_hind_legs = self.rat_legs_classifier.step(last_keypoints_history, fps, last_velocity_vectors, classification_walking)

                else:
                    class_hind_legs=0
                classification_leg = ', 2 legs' if class_hind_legs == 1 else ', 4 legs'  # Текст классификации ног

                cv2.putText(frame, classification_text + classification_leg, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Добавление текста на кадр
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Отрисовка ограничивающей рамки

                arrow_start = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))  # Начало стрелки
                arrow_end = (int(arrow_start[0] + velocity_vector[0] * 10), int(arrow_start[1] + velocity_vector[1] * 10))  # Конец стрелки
                cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 0, 255), 2)  # Отрисовка стрелки

                for i in range(0, len(skeleton_points), 3):
                    if i + 2 < len(skeleton_points):
                        x, y = skeleton_points[i], skeleton_points[i+1]  # Координаты точки
                        x, y = int(x * frame_width), int(y * frame_height)
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Отрисовка круга
                        if i // 3 in keypoints_history:
                            keypoints_history[i // 3].append((x, y))  # Добавление точки в историю

            # Добавляем номер кадра в левом верхнем углу
            cv2.putText(frame, f'Frame: {frame_index}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)  # Запись кадра в видео
            frame_index += 1  # Увеличение индекса кадра

        cap.release()  # Освобождение ресурсов
        out.release()

    def process_behavior_type(self, behavior_type):
        # Обработка видео для заданного типа поведения
        directory_labels = f'labels_{behavior_type}/labels'  # Директория с аннотациями
        count_labels = self.count_files_in_directory(directory_labels)  # Количество файлов в директории

        for i in range(1, count_labels+1):
            annotation_file = f'{directory_labels}/{behavior_type}_{i}.txt'  # Файл с аннотациями
            video_file = f'videos_{behavior_type}/videos/{behavior_type}_{i}.mp4'  # Видеофайл
            output_file_template = f'classification/video_{self.current_video_number}.mp4'  # Шаблон для пути выходного файла
            # Тестовые видео
            if self.current_video_number in [40, 22, 55, 88, 0, 26, 39, 66, 10, 44, 85, 35, 70, 62, 12, 4, 18, 28, 49]:
                annotations, frame_width, frame_height = self.read_annotations(annotation_file)  # Чтение аннотаций
                self.visualize_annotations_on_video(video_file, annotations, output_file_template, frame_width, frame_height)  # Визуализация аннотаций
            self.current_video_number += 1  # Увеличение номера видео

# Пример использования
visualizer = RatBehaviorVisualizer()
visualizer.process_behavior_type('rearing')
visualizer.process_behavior_type('walking')
