import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import math
from PIL import Image, ImageTk
import os

class Point:
    def __init__(self, x, y, z):
        self.coordinates = np.array([x, y, z, 1.0])
        self.normal = np.array([0, 0, 1])  # Нормаль по умолчанию
        self.color = None  # Цвет вершины для Гуро
        self.texture_coords = np.array([0, 0])  # Текстурные координаты [u, v]

    def transform(self, matrix):
        transformed_coordinates = np.dot(matrix, self.coordinates)
        # Преобразуем нормаль (используем верхнюю 3x3 часть матрицы)
        normal_matrix = matrix[:3, :3]
        transformed_normal = np.dot(normal_matrix, self.normal)
        transformed_normal = transformed_normal / np.linalg.norm(transformed_normal)
        
        new_point = Point(transformed_coordinates[0], transformed_coordinates[1], transformed_coordinates[2])
        new_point.normal = transformed_normal
        new_point.color = self.color
        new_point.texture_coords = self.texture_coords.copy()
        return new_point

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def transform(self, matrix):
        self.vertices = [vertex.transform(matrix) for vertex in self.vertices]
        
    def get_center_z(self):
        return np.mean([v.coordinates[2] for v in self.vertices])
        
    def get_center(self):
        """Возвращает центр полигона"""
        if not self.vertices:
            return np.array([0, 0, 0])
        return np.mean([v.coordinates[:3] for v in self.vertices], axis=0)
        
    def get_normal(self):
        if len(self.vertices) < 3:
            return np.array([0, 0, 1])
            
        # Берем первые три вершины для вычисления нормали
        v0 = self.vertices[0].coordinates[:3]
        v1 = self.vertices[1].coordinates[:3]
        v2 = self.vertices[2].coordinates[:3]
        
        # Векторы в плоскости полигона
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Векторное произведение дает нормаль
        normal = np.cross(edge1, edge2)
        
        # Нормализуем нормаль
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
            
        return normal

    @staticmethod
    def polygons_from_vertices(vertices, faces):
        polygons = []
        for face in faces:
            polygon_vertices = [vertices[i] for i in face]
            polygons.append(Polygon(polygon_vertices))
        return polygons

class Polyhedron:
    def __init__(self, polygons):
        self.polygons = polygons

    def transform(self, matrix):
        for polygon in self.polygons:
            polygon.transform(matrix)
            
    def get_center(self):
        all_vertices = []
        for polygon in self.polygons:
            all_vertices.extend(polygon.vertices)
        
        if not all_vertices:
            return np.array([0, 0, 0])
            
        return np.mean([v.coordinates[:3] for v in all_vertices], axis=0)
    
    def save_to_obj(self, filename):
        """Сохраняет полиэдр в OBJ файл"""
        try:
            with open(filename, 'w') as f:
                f.write("# OBJ файл сгенерирован программой 3D моделирования\n")
                
                # Собираем все уникальные вершины
                all_vertices = []
                vertex_to_index = {}
                current_index = 1
                
                for polygon in self.polygons:
                    for vertex in polygon.vertices:
                        vertex_tuple = (vertex.coordinates[0], vertex.coordinates[1], vertex.coordinates[2])
                        if vertex_tuple not in vertex_to_index:
                            vertex_to_index[vertex_tuple] = current_index
                            all_vertices.append(vertex)
                            current_index += 1
                
                # Записываем вершины
                for vertex in all_vertices:
                    x, y, z = vertex.coordinates[:3]
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                
                # Записываем нормали вершин
                for vertex in all_vertices:
                    nx, ny, nz = vertex.normal
                    f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
                
                # Записываем текстурные координаты
                for vertex in all_vertices:
                    u, v = vertex.texture_coords
                    f.write(f"vt {u:.6f} {v:.6f}\n")
                
                # Записываем грани
                for polygon in self.polygons:
                    face_indices = []
                    for vertex in polygon.vertices:
                        vertex_tuple = (vertex.coordinates[0], vertex.coordinates[1], vertex.coordinates[2])
                        idx = vertex_to_index[vertex_tuple]
                        face_indices.append(f"{idx}/{idx}/{idx}")
                    
                    if len(face_indices) >= 3:
                        f.write(f"f {' '.join(face_indices)}\n")
                
            return True
        except Exception as e:
            print(f"Ошибка сохранения OBJ: {e}")
            return False

    def calculate_vertex_normals(self):
        """Вычисляет нормали вершин как среднее нормалей соседних полигонов"""
        # Создаем словарь для хранения нормалей вершин
        vertex_normals = {}
        vertex_faces = {}
        
        # Собираем все вершины и их принадлежность к полигонам
        for i, polygon in enumerate(self.polygons):
            for vertex in polygon.vertices:
                vertex_tuple = (vertex.coordinates[0], vertex.coordinates[1], vertex.coordinates[2])
                if vertex_tuple not in vertex_normals:
                    vertex_normals[vertex_tuple] = np.zeros(3)
                    vertex_faces[vertex_tuple] = []
                vertex_faces[vertex_tuple].append(i)
        
        # Вычисляем нормали для каждого полигона
        polygon_normals = []
        for polygon in self.polygons:
            polygon_normals.append(polygon.get_normal())
        
        # Суммируем нормали полигонов для каждой вершины
        for vertex_tuple, face_indices in vertex_faces.items():
            for face_idx in face_indices:
                vertex_normals[vertex_tuple] += polygon_normals[face_idx]
            # Нормализуем результирующую нормаль
            norm = np.linalg.norm(vertex_normals[vertex_tuple])
            if norm > 0:
                vertex_normals[vertex_tuple] /= norm
        
        # Применяем вычисленные нормали к вершинам
        for polygon in self.polygons:
            for vertex in polygon.vertices:
                vertex_tuple = (vertex.coordinates[0], vertex.coordinates[1], vertex.coordinates[2])
                vertex.normal = vertex_normals[vertex_tuple]

    def calculate_vertex_colors(self, lighting):
        """Вычисляет цвета вершин по модели Ламберта для шейдинга Гуро"""
        for polygon in self.polygons:
            for vertex in polygon.vertices:
                position = vertex.coordinates[:3]
                normal = vertex.normal
                # Вычисляем цвет вершины по модели Ламберта
                vertex.color = lighting.lambert_shading(position, normal)

class Camera:
    def __init__(self):
        # Позиция камеры в мировых координатах
        self.position = np.array([0, 0, 500])
        # Точка, на которую смотрит камера
        self.target = np.array([0, 0, 0])
        # Вектор "вверх" для камеры
        self.up = np.array([0, 1, 0])
        # Угол обзора (в градусах)
        self.fov = 60
        # Ближняя и дальняя плоскости отсечения
        self.near_plane = 0.1
        self.far_plane = 1000
        # Текущий угол вращения камеры вокруг объекта
        self.rotation_angle = 0
        # Радиус вращения камеры вокруг объекта
        self.orbit_radius = 500
        # Скорость вращения камеры (градусов в кадр)
        self.rotation_speed = 1
        # Флаг вращения камеры
        self.is_rotating = False
        
    def get_view_matrix(self):
        """Возвращает матрицу вида камеры"""
        # Вектор направления (от камеры к цели)
        direction = self.target - self.position
        direction = direction / np.linalg.norm(direction)
        
        # Вектор "вправо"
        right = np.cross(direction, self.up)
        right = right / np.linalg.norm(right)
        
        # Корректируем вектор "вверх"
        up = np.cross(right, direction)
        up = up / np.linalg.norm(up)
        
        # Создаем матрицу вида
        view_matrix = np.eye(4)
        
        # Первая строка - ось X (право)
        view_matrix[0, 0] = right[0]
        view_matrix[0, 1] = right[1]
        view_matrix[0, 2] = right[2]
        
        # Вторая строка - ось Y (вверх)
        view_matrix[1, 0] = up[0]
        view_matrix[1, 1] = up[1]
        view_matrix[1, 2] = up[2]
        
        # Третья строка - ось Z (направление, но инвертированное)
        view_matrix[2, 0] = -direction[0]
        view_matrix[2, 1] = -direction[1]
        view_matrix[2, 2] = -direction[2]
        
        # Трансляция (позиция камеры)
        view_matrix[0, 3] = -np.dot(right, self.position)
        view_matrix[1, 3] = -np.dot(up, self.position)
        view_matrix[2, 3] = np.dot(direction, self.position)
        
        return view_matrix
    
    def get_projection_matrix(self, aspect_ratio):
        """Возвращает матрицу перспективной проекции"""
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        projection_matrix = np.zeros((4, 4))
        
        projection_matrix[0, 0] = f / aspect_ratio
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (self.far_plane + self.near_plane) / (self.near_plane - self.far_plane)
        projection_matrix[2, 3] = (2 * self.far_plane * self.near_plane) / (self.near_plane - self.far_plane)
        projection_matrix[3, 2] = -1
        
        return projection_matrix
    
    def get_orthographic_matrix(self, aspect_ratio, scale=1.0):
        """Возвращает матрицу ортографической проекции"""
        # Для простоты используем симметричную ортографическую проекцию
        ortho_matrix = np.eye(4)
        ortho_matrix[0, 0] = scale / aspect_ratio
        ortho_matrix[1, 1] = scale
        return ortho_matrix
    
    def update_orbit(self):
        """Обновляет позицию камеры для вращения вокруг цели"""
        if not self.is_rotating:
            return
            
        self.rotation_angle += self.rotation_speed
        if self.rotation_angle >= 360:
            self.rotation_angle -= 360
            
        # Вычисляем новую позицию камеры на окружности
        angle_rad = math.radians(self.rotation_angle)
        x = self.orbit_radius * math.cos(angle_rad)
        z = self.orbit_radius * math.sin(angle_rad)
        
        self.position = np.array([x, 100, z])  # Немного приподнимаем камеру
    
    def start_rotation(self):
        """Запускает вращение камеры"""
        self.is_rotating = True
        
    def stop_rotation(self):
        """Останавливает вращение камеры"""
        self.is_rotating = False
        
    def set_position(self, x, y, z):
        """Устанавливает позицию камеры"""
        self.position = np.array([x, y, z])
        
    def set_target(self, x, y, z):
        """Устанавливает цель камеры"""
        self.target = np.array([x, y, z])
        
    def move(self, dx, dy, dz):
        """Перемещает камеру на указанные значения"""
        self.position += np.array([dx, dy, dz])
        
    def move_forward(self, distance):
        """Двигает камеру вперед"""
        direction = self.target - self.position
        direction = direction / np.linalg.norm(direction)
        self.position += direction * distance
        
    def move_backward(self, distance):
        """Двигает камеру назад"""
        direction = self.target - self.position
        direction = direction / np.linalg.norm(direction)
        self.position -= direction * distance
        
    def move_left(self, distance):
        """Двигает камеру влево"""
        direction = self.target - self.position
        direction = direction / np.linalg.norm(direction)
        right = np.cross(direction, self.up)
        right = right / np.linalg.norm(right)
        self.position -= right * distance
        
    def move_right(self, distance):
        """Двигает камеру вправо"""
        direction = self.target - self.position
        direction = direction / np.linalg.norm(direction)
        right = np.cross(direction, self.up)
        right = right / np.linalg.norm(right)
        self.position += right * distance
        
    def move_up(self, distance):
        """Двигает камеру вверх"""
        self.position += self.up * distance
        
    def move_down(self, distance):
        """Двигает камеру вниз"""
        self.position -= self.up * distance
        
    def rotate_around_target(self, angle_x, angle_y):
        """Вращает камеру вокруг цели"""
        # Вычисляем вектор от цели к камере
        camera_vector = self.position - self.target
        
        # Вращение вокруг оси Y (вертикальное)
        cos_y = math.cos(angle_y)
        sin_y = math.sin(angle_y)
        y_rotation_matrix = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        # Вращение вокруг оси X (горизонтальное)
        cos_x = math.cos(angle_x)
        sin_x = math.sin(angle_x)
        x_rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        
        # Применяем вращения
        camera_vector = np.dot(y_rotation_matrix, camera_vector)
        camera_vector = np.dot(x_rotation_matrix, camera_vector)
        
        # Обновляем позицию камеры
        self.position = self.target + camera_vector
        
        # Обновляем вектор "вверх" для камеры
        self.up = np.array([0, 1, 0])  # Пока оставляем фиксированным

class Lighting:
    def __init__(self):
        # Позиция источника света
        self.light_position = np.array([300, 300, 500])
        # Цвет источника света
        self.light_color = np.array([1.0, 1.0, 1.0])  # Белый свет
        # Цвет объекта
        self.object_color = np.array([0.7, 0.7, 0.9])  # Голубоватый
        # Коэффициенты освещения
        self.ambient_intensity = 0.3  # Фоновое освещение
        self.diffuse_intensity = 0.7  # Диффузное освещение
        self.specular_intensity = 0.4  # Зеркальное освещение
        # Блеск материала (shininess)
        self.shininess = 32
        
    def set_light_position(self, x, y, z):
        self.light_position = np.array([x, y, z])
        
    def set_object_color(self, r, g, b):
        self.object_color = np.array([r, g, b])
        
    def lambert_shading(self, point, normal):
        """Вычисляет цвет по модели Ламберта (диффузное отражение)"""
        # Вектор от точки к источнику света
        light_direction = self.light_position - point
        light_direction = light_direction / np.linalg.norm(light_direction)
        
        # Диффузное освещение
        diffuse_factor = max(np.dot(normal, light_direction), 0)
        diffuse = self.diffuse_intensity * diffuse_factor * self.object_color
        
        # Фоновое освещение
        ambient = self.ambient_intensity * self.object_color
        
        # Общий цвет
        color = ambient + diffuse
        
        # Ограничиваем значения цвета
        color = np.clip(color, 0, 1)
        
        return color
        
    def phong_shading(self, point, normal, view_direction):
        """Вычисляет цвет по модели Фонга"""
        # Вектор от точки к источнику света
        light_direction = self.light_position - point
        light_direction = light_direction / np.linalg.norm(light_direction)
        
        # Отраженный вектор
        reflect_direction = 2 * np.dot(normal, light_direction) * normal - light_direction
        reflect_direction = reflect_direction / np.linalg.norm(reflect_direction)
        
        # Фоновое освещение
        ambient = self.ambient_intensity * self.object_color
        
        # Диффузное освещение
        diffuse_factor = max(np.dot(normal, light_direction), 0)
        if diffuse_factor > 0:
            diffuse_factor = 1 if diffuse_factor > 0.8 else (0.5 if diffuse_factor > 0.4 else 0)
        else:
            diffuse_factor = 0
        
        diffuse = self.diffuse_intensity * diffuse_factor * self.object_color
        
        # Зеркальное освещение
        specular_factor = max(np.dot(view_direction, reflect_direction), 0)
        specular_factor = pow(specular_factor, self.shininess)
        specular = self.specular_intensity * specular_factor * self.light_color
        
        # Общий цвет
        color = ambient + diffuse + specular
        
        # Ограничиваем значения цвета
        color = np.clip(color, 0, 1)
        
        return color

class Texture:
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        self.data = self.generate_checkerboard()
        
    def generate_checkerboard(self, square_size=8):
        """Генерирует шахматную текстуру"""
        texture = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                # Определяем цвет на основе позиции
                if (x // square_size + y // square_size) % 2 == 0:
                    texture[y, x] = [255, 0, 0]  # Красный
                else:
                    texture[y, x] = [255, 255, 0]  # Желтый
        return texture
        
    def load_from_file(self, filename):
        """Загружает текстуру из файла"""
        try:
            image = Image.open(filename)
            image = image.resize((self.width, self.height))
            self.data = np.array(image)
            return True
        except Exception as e:
            print(f"Ошибка загрузки текстуры: {e}")
            return False
            
    def get_color(self, u, v):
        """Получает цвет текстуры по координатам (u, v)"""
        # Обеспечиваем циклическое повторение текстуры
        u = u % 1.0
        v = v % 1.0
        
        x = int(u * (self.width - 1))
        y = int(v * (self.height - 1))
        
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y, x]
        else:
            return [255, 255, 255]  # Белый цвет по умолчанию

class SurfaceGenerator:
    @staticmethod
    def generate_surface(func_str, x_range, y_range, subdivisions):
        """Генерирует поверхность по функции f(x, y) = z"""
        try:
            # Создаем сетку точек
            x_min, x_max = x_range
            y_min, y_max = y_range
            n = subdivisions
            
            # Создаем регулярную сетку
            x_vals = np.linspace(x_min, x_max, n)
            y_vals = np.linspace(y_min, y_max, n)
            
            vertices = []
            faces = []
            
            # Создаем вершины
            vertex_index = 0
            vertex_grid = {}
            
            for i, x in enumerate(x_vals):
                for j, y in enumerate(y_vals):
                    try:
                        # Вычисляем z по функции
                        z = SurfaceGenerator.evaluate_function(func_str, x, y)
                        point = Point(x, y, z)
                        
                        # Вычисляем нормаль для вершины (градиент функции)
                        eps = 0.001
                        z_dx = SurfaceGenerator.evaluate_function(func_str, x + eps, y) - z
                        z_dy = SurfaceGenerator.evaluate_function(func_str, x, y + eps) - z
                        normal = np.array([-z_dx/eps, -z_dy/eps, 1])
                        normal = normal / np.linalg.norm(normal)
                        point.normal = normal
                        
                        # Текстурные координаты
                        u = (x - x_min) / (x_max - x_min)
                        v = (y - y_min) / (y_max - y_min)
                        point.texture_coords = np.array([u, v])
                        
                        vertices.append(point)
                        vertex_grid[(i, j)] = vertex_index
                        vertex_index += 1
                    except:
                        # Если вычисление не удалось, используем 0
                        vertices.append(Point(x, y, 0))
                        vertex_grid[(i, j)] = vertex_index
                        vertex_index += 1
            
            # Создаем грани (квадраты из двух треугольников)
            for i in range(n - 1):
                for j in range(n - 1):
                    # Индексы вершин текущего квадрата
                    v00 = vertex_grid[(i, j)]
                    v01 = vertex_grid[(i, j + 1)]
                    v10 = vertex_grid[(i + 1, j)]
                    v11 = vertex_grid[(i + 1, j + 1)]
                    
                    # Два треугольника, образующих квадрат
                    faces.append([v00, v01, v11])  # Первый треугольник
                    faces.append([v00, v11, v10])  # Второй треугольник
            
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            polyhedron = Polyhedron(polygons)
            polyhedron.calculate_vertex_normals()  # Пересчитываем нормали вершин
            return polyhedron
            
        except Exception as e:
            raise Exception(f"Ошибка генерации поверхности: {e}")
    
    @staticmethod
    def evaluate_function(func_str, x, y):
        """Вычисляет значение функции f(x, y)"""
        # Заменяем математические функции на их numpy аналоги
        func_str = func_str.replace('sin', 'np.sin')
        func_str = func_str.replace('cos', 'np.cos')
        func_str = func_str.replace('tan', 'np.tan')
        func_str = func_str.replace('exp', 'np.exp')
        func_str = func_str.replace('log', 'np.log')
        func_str = func_str.replace('sqrt', 'np.sqrt')
        func_str = func_str.replace('pi', 'np.pi')
        func_str = func_str.replace('e', 'np.e')
        
        # Вычисляем значение
        return eval(func_str, {'np': np, 'x': x, 'y': y})

class RotationFigureGenerator:
    @staticmethod
    def generate_rotation_figure(generatrix_points, axis, subdivisions):
        """
        Генерирует фигуру вращения
        
        Args:
            generatrix_points: список точек образующей [(x, y), ...]
            axis: ось вращения ('x', 'y', 'z')
            subdivisions: количество разбиений
        """
        try:
            vertices = []
            faces = []
            
            # Угол поворота между сегментами
            angle_step = 2 * math.pi / subdivisions
            
            # Создаем вершины
            for i in range(subdivisions):
                angle = i * angle_step
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                
                for point in generatrix_points:
                    x, y = point
                    
                    if axis == 'x':
                        # Вращение вокруг оси X
                        new_x = x
                        new_y = y * cos_a
                        new_z = y * sin_a
                    elif axis == 'y':
                        # Вращение вокруг оси Y
                        new_x = x * cos_a
                        new_y = y
                        new_z = x * sin_a
                    elif axis == 'z':
                        # Вращение вокруг оси Z
                        new_x = x * cos_a - y * sin_a
                        new_y = x * sin_a + y * cos_a
                        new_z = 0
                    
                    point_obj = Point(new_x, new_y, new_z)
                    
                    # Текстурные координаты
                    u = i / subdivisions
                    v = (point[0] - min(p[0] for p in generatrix_points)) / (max(p[0] for p in generatrix_points) - min(p[0] for p in generatrix_points))
                    point_obj.texture_coords = np.array([u, v])
                    
                    vertices.append(point_obj)
            
            # Создаем грани
            n_points = len(generatrix_points)
            for i in range(subdivisions):
                current_slice = i
                next_slice = (i + 1) % subdivisions
                
                for j in range(n_points - 1):
                    # Индексы вершин для текущего квадрата
                    v00 = current_slice * n_points + j
                    v01 = current_slice * n_points + j + 1
                    v10 = next_slice * n_points + j
                    v11 = next_slice * n_points + j + 1
                    
                    # Создаем два треугольника для квадрата
                    faces.append([v00, v01, v11])
                    faces.append([v00, v11, v10])
            
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            polyhedron = Polyhedron(polygons)
            polyhedron.calculate_vertex_normals()  # Вычисляем нормали вершин
            return polyhedron
            
        except Exception as e:
            raise Exception(f"Ошибка генерации фигуры вращения: {e}")

class ZBuffer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.buffer = np.full((height, width), np.inf)
        self.color_buffer = np.full((height, width, 3), 255, dtype=np.uint8)
        
    def clear(self):
        self.buffer.fill(np.inf)
        self.color_buffer.fill(255)
        
    def update(self, x, y, z, color):
        if 0 <= x < self.width and 0 <= y < self.height:
            if z < self.buffer[y, x]:
                self.buffer[y, x] = z
                self.color_buffer[y, x] = color

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Polyhedra - Полная реализация")
        self.root.geometry("1400x900")
        
        # Создаем панель для разделения окна
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - холст
        canvas_frame = ttk.Frame(paned_window)
        paned_window.add(canvas_frame, weight=3)  # Больший вес для холста
        
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Правая панель - элементы управления
        controls_frame = ttk.Frame(paned_window)
        paned_window.add(controls_frame, weight=1)  # Меньший вес для элементов управления
        
        # Создаем скроллируемый фрейм для элементов управления
        self.scroll_frame = ttk.Frame(controls_frame)
        self.scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем Canvas и Scrollbar
        self.canvas_controls = tk.Canvas(self.scroll_frame, height=800)
        scrollbar = ttk.Scrollbar(self.scroll_frame, orient="vertical", command=self.canvas_controls.yview)
        self.scrollable_frame = ttk.Frame(self.canvas_controls)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_controls.configure(
                scrollregion=self.canvas_controls.bbox("all")
            )
        )
        
        self.canvas_controls.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_controls.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.canvas_controls.pack(side="left", fill="both", expand=True)
        
        self.polyhedron = None
        self.projection_type = 'perspective'
        self.backface_culling = True
        self.use_z_buffer = False
        
        # Создаем камеру
        self.camera = Camera()
        self.camera_view = False  # Режим отображения с камеры
        self.animation_id = None  # ID анимации
        
        # Создаем освещение
        self.lighting = Lighting()
        self.use_lighting = False
        self.shading_mode = 'phong'  # 'phong', 'gouraud', 'texture'
        
        # Создаем текстуру
        self.texture = Texture()
        
        # Инициализация Z-буфера
        self.z_buffer = ZBuffer(800, 800)
        
        # Цвета для разных полигонов
        self.colors = np.array([
            [135, 206, 250], [135, 20, 25], [10, 206, 25], [135, 0, 250],
            [10, 20, 250], [10, 20, 25], [200, 200, 200], [135, 206, 0]
        ])
        
        # Переменные для управления камерой мышью
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_dragging = False
        self.drag_mode = None  # 'rotate' или 'move'
        
        # Привязываем обработчики событий мыши
        self.canvas.bind("<Button-1>", self.on_mouse_down)  # ЛКМ
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)  # Перетаскивание ЛКМ
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)  # Отпускание ЛКМ
        self.canvas.bind("<Button-3>", self.on_mouse_down)  # ПКМ
        self.canvas.bind("<B3-Motion>", self.on_mouse_drag)  # Перетаскивание ПКМ
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up)  # Отпускание ПКМ
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Колесо мыши (Windows)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Колесо мыши (Linux)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Колесо мыши (Linux)
        
        # Привязываем обработчики клавиш
        self.root.bind("<KeyPress>", self.on_key_press)
        
        # Создание интерфейса
        self.create_controls_panel()
        self.render()
        
        # Запускаем цикл анимации
        self.animate()

    def create_controls_panel(self):
        # Панель выбора режима шейдинга
        shading_frame = ttk.LabelFrame(self.scrollable_frame, text="Режим шейдинга", padding=5)
        shading_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.shading_var = tk.StringVar(value=self.shading_mode)
        ttk.Radiobutton(shading_frame, text="Фонг (интерполяция нормалей)", 
                       variable=self.shading_var, value='phong',
                       command=self.update_shading_mode).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(shading_frame, text="Гуро (интерполяция цветов)", 
                       variable=self.shading_var, value='gouraud',
                       command=self.update_shading_mode).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(shading_frame, text="Текстурирование", 
                       variable=self.shading_var, value='texture',
                       command=self.update_shading_mode).pack(anchor=tk.W, pady=2)
        
        # Панель управления освещением
        lighting_frame = ttk.LabelFrame(self.scrollable_frame, text="Освещение", padding=5)
        lighting_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Включение/отключение освещения
        self.lighting_var = tk.BooleanVar(value=self.use_lighting)
        ttk.Checkbutton(lighting_frame, text="Включить освещение", 
                       variable=self.lighting_var, command=self.toggle_lighting).pack(anchor=tk.W, pady=2)
        
        # Позиция источника света
        light_pos_frame = ttk.Frame(lighting_frame)
        light_pos_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(light_pos_frame, text="Источник света (x,y,z):").pack(anchor=tk.W)
        light_input_frame = ttk.Frame(light_pos_frame)
        light_input_frame.pack(fill=tk.X, pady=2)
        
        self.light_x_entry = ttk.Entry(light_input_frame, width=5)
        self.light_x_entry.insert(0, "300")
        self.light_x_entry.pack(side=tk.LEFT, padx=2)
        
        self.light_y_entry = ttk.Entry(light_input_frame, width=5)
        self.light_y_entry.insert(0, "300")
        self.light_y_entry.pack(side=tk.LEFT, padx=2)
        
        self.light_z_entry = ttk.Entry(light_input_frame, width=5)
        self.light_z_entry.insert(0, "500")
        self.light_z_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(light_input_frame, text="Применить", 
                  command=self.update_light_position).pack(side=tk.LEFT, padx=5)
        
        # Цвет объекта
        color_frame = ttk.Frame(lighting_frame)
        color_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(color_frame, text="Цвет объекта (R,G,B):").pack(anchor=tk.W)
        color_input_frame = ttk.Frame(color_frame)
        color_input_frame.pack(fill=tk.X, pady=2)
        
        self.color_r_entry = ttk.Entry(color_input_frame, width=5)
        self.color_r_entry.insert(0, "0.7")
        self.color_r_entry.pack(side=tk.LEFT, padx=2)
        
        self.color_g_entry = ttk.Entry(color_input_frame, width=5)
        self.color_g_entry.insert(0, "0.7")
        self.color_g_entry.pack(side=tk.LEFT, padx=2)
        
        self.color_b_entry = ttk.Entry(color_input_frame, width=5)
        self.color_b_entry.insert(0, "0.9")
        self.color_b_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(color_input_frame, text="Применить", 
                  command=self.update_object_color).pack(side=tk.LEFT, padx=5)
        
        # Панель управления текстурами
        texture_frame = ttk.LabelFrame(self.scrollable_frame, text="Текстурирование", padding=5)
        texture_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(texture_frame, text="Загрузить текстуру", 
                  command=self.load_texture).pack(fill=tk.X, pady=2)
        ttk.Button(texture_frame, text="Сбросить текстуру", 
                  command=self.reset_texture).pack(fill=tk.X, pady=2)
        
        # Панель управления алгоритмами удаления невидимых граней
        algorithm_frame = ttk.LabelFrame(self.scrollable_frame, text="Алгоритмы удаления невидимых граней", padding=5)
        algorithm_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Включение/отключение Z-буфера
        self.z_buffer_var = tk.BooleanVar(value=self.use_z_buffer)
        ttk.Checkbutton(algorithm_frame, text="Использовать Z-буфер", 
                       variable=self.z_buffer_var, command=self.toggle_z_buffer).pack(anchor=tk.W, pady=2)
        
        # Включение/отключение отсечения нелицевых граней
        self.culling_var = tk.BooleanVar(value=self.backface_culling)
        ttk.Checkbutton(algorithm_frame, text="Включить отсечение нелицевых граней", 
                       variable=self.culling_var, command=self.toggle_backface_culling).pack(anchor=tk.W, pady=2)
        
        # Панель управления камерой (компактная версия)
        camera_frame = ttk.LabelFrame(self.scrollable_frame, text="Управление камерой", padding=5)
        camera_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Переключение между видами
        view_buttons_frame = ttk.Frame(camera_frame)
        view_buttons_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(view_buttons_frame, text="Вид с камеры", 
                  command=self.enable_camera_view).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(view_buttons_frame, text="Обычный вид", 
                  command=self.disable_camera_view).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Управление вращением камеры
        rotation_frame = ttk.Frame(camera_frame)
        rotation_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(rotation_frame, text="Старт вращения", 
                  command=self.start_camera_rotation).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(rotation_frame, text="Стоп вращения", 
                  command=self.stop_camera_rotation).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Быстрые настройки камеры
        quick_frame = ttk.Frame(camera_frame)
        quick_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(quick_frame, text="Спереди", 
                  command=lambda: self.set_camera_quick(0, 0, 500)).pack(side=tk.LEFT, padx=1)
        ttk.Button(quick_frame, text="Сверху", 
                  command=lambda: self.set_camera_quick(0, 500, 0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(quick_frame, text="Сбоку", 
                  command=lambda: self.set_camera_quick(500, 0, 0)).pack(side=tk.LEFT, padx=1)
        
        # Управление камерой с клавиатуры
        control_frame = ttk.LabelFrame(camera_frame, text="Управление камерой (WASD + мышь)", padding=5)
        control_frame.pack(fill=tk.X, pady=2)
        
        controls_text = """Управление камерой:
- ЛКМ + перетаскивание: вращение камеры
- ПКМ + перетаскивание: перемещение камеры
- Колесо мыши: приближение/отдаление
- WASD: перемещение камеры
- Q/E: движение вверх/вниз
- R: сброс позиции камеры"""
        
        ttk.Label(control_frame, text=controls_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Панель построения фигуры вращения
        rotation_frame = ttk.LabelFrame(self.scrollable_frame, text="Построение фигуры вращения", padding=5)
        rotation_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Образующая
        ttk.Label(rotation_frame, text="Образующая (x,y через пробел):").pack(anchor=tk.W)
        self.generatrix_entry = tk.Text(rotation_frame, width=30, height=3)
        self.generatrix_entry.insert("1.0", "0 0\n1 0\n1 2\n0.5 3\n0 2")
        self.generatrix_entry.pack(fill=tk.X, pady=2)
        
        # Ось вращения
        axis_frame = ttk.Frame(rotation_frame)
        axis_frame.pack(fill=tk.X, pady=2)
        ttk.Label(axis_frame, text="Ось вращения:").pack(side=tk.LEFT)
        self.axis_var = tk.StringVar(value="y")
        ttk.Radiobutton(axis_frame, text="X", variable=self.axis_var, value="x").pack(side=tk.LEFT)
        ttk.Radiobutton(axis_frame, text="Y", variable=self.axis_var, value="y").pack(side=tk.LEFT)
        ttk.Radiobutton(axis_frame, text="Z", variable=self.axis_var, value="z").pack(side=tk.LEFT)
        
        # Количество разбиений
        subdiv_frame = ttk.Frame(rotation_frame)
        subdiv_frame.pack(fill=tk.X, pady=2)
        ttk.Label(subdiv_frame, text="Разбиений:").pack(side=tk.LEFT)
        self.rotation_subdivisions_entry = ttk.Entry(subdiv_frame, width=5)
        self.rotation_subdivisions_entry.insert(0, "20")
        self.rotation_subdivisions_entry.pack(side=tk.LEFT, padx=5)
        
        # Кнопка построения фигуры вращения
        ttk.Button(rotation_frame, text="Построить фигуру вращения", 
                  command=self.generate_rotation_figure).pack(fill=tk.X, pady=2)
        
        # Панель построения поверхности
        surface_frame = ttk.LabelFrame(self.scrollable_frame, text="Построение поверхности", padding=5)
        surface_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Функция
        ttk.Label(surface_frame, text="f(x, y) =").pack(anchor=tk.W)
        self.func_entry = ttk.Entry(surface_frame, width=30)
        self.func_entry.insert(0, "(x**2 + y**2) / 10")
        self.func_entry.pack(fill=tk.X, pady=2)
        
        # Диапазоны
        range_frame = ttk.Frame(surface_frame)
        range_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(range_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x0_entry = ttk.Entry(range_frame, width=5)
        self.x0_entry.insert(0, "-6")
        self.x0_entry.grid(row=0, column=1, padx=2)
        
        ttk.Label(range_frame, text="до").grid(row=0, column=2)
        self.x1_entry = ttk.Entry(range_frame, width=5)
        self.x1_entry.insert(0, "6")
        self.x1_entry.grid(row=0, column=3, padx=2)
        
        ttk.Label(range_frame, text="Y:").grid(row=1, column=0, sticky=tk.W)
        self.y0_entry = ttk.Entry(range_frame, width=5)
        self.y0_entry.insert(0, "-6")
        self.y0_entry.grid(row=1, column=1, padx=2)
        
        ttk.Label(range_frame, text="до").grid(row=1, column=2)
        self.y1_entry = ttk.Entry(range_frame, width=5)
        self.y1_entry.insert(0, "6")
        self.y1_entry.grid(row=1, column=3, padx=2)
        
        # Количество разбиений
        subdiv_frame = ttk.Frame(surface_frame)
        subdiv_frame.pack(fill=tk.X, pady=2)
        ttk.Label(subdiv_frame, text="Разбиений:").pack(side=tk.LEFT)
        self.subdivisions_entry = ttk.Entry(subdiv_frame, width=5)
        self.subdivisions_entry.insert(0, "20")
        self.subdivisions_entry.pack(side=tk.LEFT, padx=5)
        
        # Кнопки построения поверхности
        ttk.Button(surface_frame, text="Построить поверхность", 
                  command=self.generate_surface).pack(fill=tk.X, pady=2)
        
        # Кнопки загрузки/сохранения OBJ
        file_frame = ttk.Frame(self.scrollable_frame)
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_frame, text="Загрузить OBJ", 
                  command=self.load_obj_file).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Сохранить OBJ", 
                  command=self.save_obj_file).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Выбор проекции
        proj_frame = ttk.LabelFrame(self.scrollable_frame, text="Тип проекции", padding=5)
        proj_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.projection_var = tk.StringVar(value=self.projection_type)
        ttk.Radiobutton(proj_frame, text="Перспективная", 
                       variable=self.projection_var, value='perspective',
                       command=lambda: self.set_projection('perspective')).pack(anchor=tk.W)
        ttk.Radiobutton(proj_frame, text="Аксонометрическая", 
                       variable=self.projection_var, value='axonometric',
                       command=lambda: self.set_projection('axonometric')).pack(anchor=tk.W)
        
        # Разделитель
        ttk.Separator(self.scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Аффинные преобразования
        transform_frame = ttk.LabelFrame(self.scrollable_frame, text="Аффинные преобразования", padding=5)
        transform_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.create_translation_controls(transform_frame)
        self.create_rotation_controls(transform_frame)
        self.create_scaling_controls(transform_frame)
        self.create_reflection_controls(transform_frame)
        self.create_arbitrary_rotation_controls(transform_frame)

    def create_translation_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Смещение", padding=5)
        frame.pack(fill=tk.X, pady=2)
        
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X)
        
        ttk.Label(input_frame, text="dx:").grid(row=0, column=0, padx=2)
        self.dx_entry = ttk.Entry(input_frame, width=5)
        self.dx_entry.insert(0, "10")
        self.dx_entry.grid(row=0, column=1, padx=2)
        
        ttk.Label(input_frame, text="dy:").grid(row=0, column=2, padx=2)
        self.dy_entry = ttk.Entry(input_frame, width=5)
        self.dy_entry.insert(0, "10")
        self.dy_entry.grid(row=0, column=3, padx=2)
        
        ttk.Label(input_frame, text="dz:").grid(row=0, column=4, padx=2)
        self.dz_entry = ttk.Entry(input_frame, width=5)
        self.dz_entry.insert(0, "10")
        self.dz_entry.grid(row=0, column=5, padx=2)
        
        ttk.Button(frame, text="Применить смещение", command=self.apply_translation).pack(pady=2)

    def create_rotation_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Вращение", padding=5)
        frame.pack(fill=tk.X, pady=2)
        
        # Вращение вокруг точки
        point_frame = ttk.Frame(frame)
        point_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(point_frame, text="Вокруг точки:").pack(side=tk.LEFT)
        self.center_x_entry = ttk.Entry(point_frame, width=5)
        self.center_x_entry.insert(0, "300")
        self.center_x_entry.pack(side=tk.LEFT, padx=2)
        
        self.center_y_entry = ttk.Entry(point_frame, width=5)
        self.center_y_entry.insert(0, "300")
        self.center_y_entry.pack(side=tk.LEFT, padx=2)
        
        self.center_z_entry = ttk.Entry(point_frame, width=5)
        self.center_z_entry.insert(0, "300")
        self.center_z_entry.pack(side=tk.LEFT, padx=2)
        
        self.angle_entry = ttk.Entry(point_frame, width=5)
        self.angle_entry.insert(0, "30")
        self.angle_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(point_frame, text="°").pack(side=tk.LEFT)
        
        ttk.Button(point_frame, text="Применить", command=self.apply_rotation_around_point).pack(side=tk.LEFT, padx=5)
        
        # Вращение вокруг центра
        center_frame = ttk.Frame(frame)
        center_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(center_frame, text="Вокруг центра:").pack(side=tk.LEFT)
        ttk.Button(center_frame, text="X", command=lambda: self.apply_rotation_around_center('x')).pack(side=tk.LEFT, padx=2)
        ttk.Button(center_frame, text="Y", command=lambda: self.apply_rotation_around_center('y')).pack(side=tk.LEFT, padx=2)
        ttk.Button(center_frame, text="Z", command=lambda: self.apply_rotation_around_center('z')).pack(side=tk.LEFT, padx=2)

    def create_scaling_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Масштабирование", padding=5)
        frame.pack(fill=tk.X, pady=2)
        
        scale_frame = ttk.Frame(frame)
        scale_frame.pack(fill=tk.X)
        
        ttk.Label(scale_frame, text="Коэффициент:").pack(side=tk.LEFT)
        self.scale_factor_entry = ttk.Entry(scale_frame, width=5)
        self.scale_factor_entry.insert(0, "1.5")
        self.scale_factor_entry.pack(side=tk.LEFT, padx=5)
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="От центра", command=self.apply_scaling_around_center).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="От точки", command=self.apply_scaling_around_point).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

    def create_reflection_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Отражение", padding=5)
        frame.pack(fill=tk.X, pady=2)
        
        button_frame = ttk.Frame(frame)
        button_frame.pack()
        
        ttk.Button(button_frame, text="XY плоскость", command=lambda: self.apply_reflection('xy')).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="XZ плоскость", command=lambda: self.apply_reflection('xz')).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="YZ плоскость", command=lambda: self.apply_reflection('yz')).pack(side=tk.LEFT, padx=2)

    def create_arbitrary_rotation_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Вращение вокруг произвольной прямой", padding=5)
        frame.pack(fill=tk.X, pady=2)
        
        p1_frame = ttk.Frame(frame)
        p1_frame.pack(fill=tk.X, pady=2)
        ttk.Label(p1_frame, text="Точка 1:").pack(side=tk.LEFT)
        self.p1_x = ttk.Entry(p1_frame, width=5)
        self.p1_x.insert(0, "200")
        self.p1_x.pack(side=tk.LEFT, padx=2)
        self.p1_y = ttk.Entry(p1_frame, width=5)
        self.p1_y.insert(0, "200")
        self.p1_y.pack(side=tk.LEFT, padx=2)
        self.p1_z = ttk.Entry(p1_frame, width=5)
        self.p1_z.insert(0, "200")
        self.p1_z.pack(side=tk.LEFT, padx=2)
        
        p2_frame = ttk.Frame(frame)
        p2_frame.pack(fill=tk.X, pady=2)
        ttk.Label(p2_frame, text="Точка 2:").pack(side=tk.LEFT)
        self.p2_x = ttk.Entry(p2_frame, width=5)
        self.p2_x.insert(0, "400")
        self.p2_x.pack(side=tk.LEFT, padx=2)
        self.p2_y = ttk.Entry(p2_frame, width=5)
        self.p2_y.insert(0, "400")
        self.p2_y.pack(side=tk.LEFT, padx=2)
        self.p2_z = ttk.Entry(p2_frame, width=5)
        self.p2_z.insert(0, "400")
        self.p2_z.pack(side=tk.LEFT, padx=2)
        
        angle_frame = ttk.Frame(frame)
        angle_frame.pack(fill=tk.X, pady=2)
        ttk.Label(angle_frame, text="Угол:").pack(side=tk.LEFT)
        self.arbitrary_angle_entry = ttk.Entry(angle_frame, width=5)
        self.arbitrary_angle_entry.insert(0, "45")
        self.arbitrary_angle_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(angle_frame, text="°").pack(side=tk.LEFT)
        
        ttk.Button(frame, text="Применить вращение", command=self.apply_arbitrary_rotation).pack(pady=2)

    # ===== НОВЫЕ МЕТОДЫ ДЛЯ ШЕЙДИНГА ГУРО И ТЕКСТУРИРОВАНИЯ =====
    
    def update_shading_mode(self):
        """Обновляет режим шейдинга"""
        self.shading_mode = self.shading_var.get()
        if self.polyhedron and self.shading_mode == 'gouraud' and self.use_lighting:
            self.polyhedron.calculate_vertex_colors(self.lighting)
        self.render()
        
    def load_texture(self):
        """Загружает текстуру из файла"""
        file_path = filedialog.askopenfilename(
            title="Загрузить текстуру",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.texture.load_from_file(file_path):
                messagebox.showinfo("Успех", "Текстура успешно загружена")
                if self.shading_mode != 'texture':
                    self.shading_var.set('texture')
                    self.update_shading_mode()
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить текстуру")
                
    def reset_texture(self):
        """Сбрасывает текстуру на шахматную доску"""
        self.texture = Texture()
        self.render()

    # ===== МЕТОДЫ УПРАВЛЕНИЯ ОСВЕЩЕНИЕМ =====
    
    def toggle_lighting(self):
        """Включение/отключение освещения"""
        self.use_lighting = self.lighting_var.get()
        if self.polyhedron and self.shading_mode == 'gouraud' and self.use_lighting:
            self.polyhedron.calculate_vertex_colors(self.lighting)
        self.render()
        
    def update_light_position(self):
        """Обновляет позицию источника света"""
        try:
            x = float(self.light_x_entry.get())
            y = float(self.light_y_entry.get())
            z = float(self.light_z_entry.get())
            self.lighting.set_light_position(x, y, z)
            if self.polyhedron and self.shading_mode == 'gouraud' and self.use_lighting:
                self.polyhedron.calculate_vertex_colors(self.lighting)
            self.render()
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные координаты источника света")
            
    def update_object_color(self):
        """Обновляет цвет объекта"""
        try:
            r = float(self.color_r_entry.get())
            g = float(self.color_g_entry.get())
            b = float(self.color_b_entry.get())
            self.lighting.set_object_color(r, g, b)
            if self.polyhedron and self.shading_mode == 'gouraud' and self.use_lighting:
                self.polyhedron.calculate_vertex_colors(self.lighting)
            self.render()
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные значения цвета")

    # ===== МЕТОДЫ УПРАВЛЕНИЯ КАМЕРОЙ =====
    
    def set_camera_quick(self, x, y, z):
        """Быстрая установка позиции камеры"""
        self.camera.set_position(x, y, z)
        if not self.camera_view:
            self.enable_camera_view()
        self.render()
        
    def enable_camera_view(self):
        """Включение вида с камеры"""
        self.camera_view = True
        self.render()
        
    def disable_camera_view(self):
        """Отключение вида с камеры"""
        self.camera_view = False
        self.render()
        
    def start_camera_rotation(self):
        """Запуск вращения камеры вокруг объекта"""
        self.camera.start_rotation()
        
    def stop_camera_rotation(self):
        """Остановка вращения камеры"""
        self.camera.stop_rotation()
        
    def reset_camera(self):
        """Сброс камеры в начальное положение"""
        self.camera.set_position(0, 0, 500)
        self.camera.set_target(0, 0, 0)
        self.render()

    def animate(self):
        """Цикл анимации"""
        if self.camera.is_rotating:
            self.camera.update_orbit()
            self.render()
            
        # Планируем следующий кадр анимации
        self.animation_id = self.root.after(16, self.animate)  # ~60 FPS

    # ===== ОБРАБОТЧИКИ СОБЫТИЙ МЫШИ =====
    
    def on_mouse_down(self, event):
        """Обработчик нажатия кнопки мыши"""
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.mouse_dragging = True
        
        # Определяем режим перетаскивания
        if event.num == 1:  # ЛКМ
            self.drag_mode = 'rotate'
        elif event.num == 3:  # ПКМ
            self.drag_mode = 'move'
    
    def on_mouse_drag(self, event):
        """Обработчик перетаскивания мыши"""
        if not self.mouse_dragging:
            return
            
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        
        if self.drag_mode == 'rotate':
            # Вращение камеры вокруг цели
            angle_x = dy * 0.01  # Чувствительность вращения
            angle_y = dx * 0.01
            self.camera.rotate_around_target(angle_x, angle_y)
        elif self.drag_mode == 'move':
            # Перемещение камеры
            move_speed = 0.5
            self.camera.move(-dx * move_speed, dy * move_speed, 0)
            # Также перемещаем цель, чтобы камера продолжала смотреть в том же направлении
            self.camera.set_target(self.camera.target[0] - dx * move_speed, 
                                 self.camera.target[1] + dy * move_speed, 
                                 self.camera.target[2])
        
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.render()
    
    def on_mouse_up(self, event):
        """Обработчик отпускания кнопки мыши"""
        self.mouse_dragging = False
        self.drag_mode = None
    
    def on_mouse_wheel(self, event):
        """Обработчик колеса мыши"""
        if event.delta > 0 or event.num == 4:  # Вверх или кнопка 4 (Linux)
            # Приближение
            self.camera.move_forward(10)
        else:  # Вниз или кнопка 5 (Linux)
            # Отдаление
            self.camera.move_backward(10)
        self.render()
    
    def on_key_press(self, event):
        """Обработчик нажатия клавиш"""
        move_speed = 10
        
        if event.char.lower() == 'w':
            # Движение вперед
            self.camera.move_forward(move_speed)
        elif event.char.lower() == 's':
            # Движение назад
            self.camera.move_backward(move_speed)
        elif event.char.lower() == 'a':
            # Движение влево
            self.camera.move_left(move_speed)
        elif event.char.lower() == 'd':
            # Движение вправо
            self.camera.move_right(move_speed)
        elif event.char.lower() == 'q':
            # Движение вверх
            self.camera.move_up(move_speed)
        elif event.char.lower() == 'e':
            # Движение вниз
            self.camera.move_down(move_speed)
        elif event.char.lower() == 'r':
            # Сброс камеры
            self.reset_camera()
        else:
            return  # Неизвестная клавиша, игнорируем
        
        self.render()

    def toggle_backface_culling(self):
        """Включение/отключение отсечения нелицевых граней"""
        self.backface_culling = self.culling_var.get()
        self.render()

    def toggle_z_buffer(self):
        """Включение/отключение Z-буфера"""
        self.use_z_buffer = self.z_buffer_var.get()
        self.render()

    def is_polygon_visible(self, polygon):
        """Проверка видимости полигона с учетом отсечения нелицевых граней"""
        if not self.backface_culling:
            return True
            
        normal = polygon.get_normal()
        
        # Если нормаль не определена, считаем полигон видимым
        if np.linalg.norm(normal) == 0:
            return True
            
        # В режиме камеры используем направление от камеры к полигону
        if self.camera_view:
            # Вычисляем вектор от камеры к центру полигона
            polygon_center = polygon.get_center()
            view_dir = polygon_center - self.camera.position
            view_dir = view_dir / np.linalg.norm(view_dir)
            
            # Используем скалярное произведение для определения видимости
            dot_product = np.dot(normal, view_dir)
            
            # Полигон видим, если угол между нормалью и вектором обзора меньше 90 градусов
            # (нормаль смотрит в сторону камеры)
            return dot_product > 0
        else:
            # В обычном режиме используем направление камеры (от камеры к цели)
            view_dir = self.camera.target - self.camera.position
            view_dir = view_dir / np.linalg.norm(view_dir)
            dot_product = np.dot(normal, view_dir)
            
            return dot_product > 0

    def generate_rotation_figure(self):
        """Генерирует фигуру вращения по заданной образующей"""
        try:
            # Читаем точки образующей
            generatrix_text = self.generatrix_entry.get("1.0", tk.END).strip()
            lines = generatrix_text.split('\n')
            
            generatrix_points = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        generatrix_points.append((x, y))
            
            if len(generatrix_points) < 2:
                messagebox.showerror("Ошибка", "Образующая должна содержать хотя бы 2 точки")
                return
            
            axis = self.axis_var.get()
            subdivisions = int(self.rotation_subdivisions_entry.get())
            
            if subdivisions < 3:
                messagebox.showerror("Ошибка", "Количество разбиений должно быть не менее 3")
                return
            
            self.polyhedron = RotationFigureGenerator.generate_rotation_figure(
                generatrix_points, axis, subdivisions
            )
            self.center_polyhedron()
            if self.shading_mode == 'gouraud' and self.use_lighting:
                self.polyhedron.calculate_vertex_colors(self.lighting)
            self.render()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить фигуру вращения: {e}")

    def generate_surface(self):
        """Генерирует поверхность по заданной функции"""
        try:
            func_str = self.func_entry.get()
            x_range = (float(self.x0_entry.get()), float(self.x1_entry.get()))
            y_range = (float(self.y0_entry.get()), float(self.y1_entry.get()))
            subdivisions = int(self.subdivisions_entry.get())
            
            if subdivisions < 2:
                messagebox.showerror("Ошибка", "Количество разбиений должно быть не менее 2")
                return
            
            self.polyhedron = SurfaceGenerator.generate_surface(func_str, x_range, y_range, subdivisions)
            self.center_polyhedron()
            if self.shading_mode == 'gouraud' and self.use_lighting:
                self.polyhedron.calculate_vertex_colors(self.lighting)
            self.render()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить поверхность: {e}")

    def save_obj_file(self):
        """Сохраняет модель в OBJ файл"""
        if not self.polyhedron:
            messagebox.showwarning("Предупреждение", "Нет модели для сохранения")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Сохранить OBJ файл",
            defaultextension=".obj",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if self.polyhedron.save_to_obj(file_path):
                    messagebox.showinfo("Успех", "Модель успешно сохранена в OBJ файл")
                else:
                    messagebox.showerror("Ошибка", "Не удалось сохранить модель")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить модель: {e}")

    def load_obj_file(self):
        """Загружает модель из OBJ файла"""
        file_path = filedialog.askopenfilename(
            title="Загрузить OBJ файл",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                vertices = []
                faces = []
                scale = 50  # Масштабирование для отображения
                
                # Словари для хранения данных из OBJ
                vertex_normals = {}
                texture_coords = {}
                
                with open(file_path, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                            
                        parts = line.split()
                        if not parts:
                            continue
                            
                        if parts[0] == 'v':  # вершина
                            if len(parts) >= 4:
                                x = float(parts[1]) * scale
                                y = float(parts[2]) * scale
                                z = float(parts[3]) * scale
                                vertices.append(Point(x, y, z))
                        elif parts[0] == 'vn':  # нормаль вершины
                            if len(parts) >= 4:
                                nx = float(parts[1])
                                ny = float(parts[2])
                                nz = float(parts[3])
                                idx = len(vertex_normals) + 1
                                vertex_normals[idx] = np.array([nx, ny, nz])
                        elif parts[0] == 'vt':  # текстурные координаты
                            if len(parts) >= 3:
                                u = float(parts[1])
                                v = float(parts[2])
                                idx = len(texture_coords) + 1
                                texture_coords[idx] = np.array([u, v])
                        elif parts[0] == 'f':  # грань
                            face_vertices = []
                            for part in parts[1:]:
                                # OBJ формат может быть v, v/vt, v/vt/vn, v//vn
                                vertex_data = part.split('/')
                                if vertex_data[0]:
                                    # Индексы в OBJ начинаются с 1, поэтому вычитаем 1
                                    vertex_index = int(vertex_data[0]) - 1
                                    if vertex_index >= 0 and vertex_index < len(vertices):
                                        # Применяем нормаль, если она есть
                                        if len(vertex_data) >= 3 and vertex_data[2]:
                                            normal_index = int(vertex_data[2])
                                            if normal_index in vertex_normals:
                                                vertices[vertex_index].normal = vertex_normals[normal_index]
                                        
                                        # Применяем текстурные координаты, если они есть
                                        if len(vertex_data) >= 2 and vertex_data[1]:
                                            tex_index = int(vertex_data[1])
                                            if tex_index in texture_coords:
                                                vertices[vertex_index].texture_coords = texture_coords[tex_index]
                                        
                                        face_vertices.append(vertex_index)
                            
                            if len(face_vertices) >= 3:
                                # Если грань имеет больше 3 вершин, разбиваем на треугольники
                                for i in range(1, len(face_vertices) - 1):
                                    faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])
                
                if vertices and faces:
                    polygons = Polygon.polygons_from_vertices(vertices, faces)
                    self.polyhedron = Polyhedron(polygons)
                    # Если нормали не были загружены из файла, вычисляем их
                    if not any(np.any(v.normal != [0, 0, 1]) for v in vertices):
                        self.polyhedron.calculate_vertex_normals()
                    self.center_polyhedron()
                    if self.shading_mode == 'gouraud' and self.use_lighting:
                        self.polyhedron.calculate_vertex_colors(self.lighting)
                    self.render()
                    
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить OBJ файл: {e}")

    def center_polyhedron(self):
        if not self.polyhedron:
            return
            
        center = self.polyhedron.get_center()
        translate_matrix = self.get_translation_matrix(300 - center[0], 300 - center[1], 300 - center[2])
        self.polyhedron.transform(translate_matrix)

    # МАТРИЧНЫЕ ПРЕОБРАЗОВАНИЯ
    def get_translation_matrix(self, dx, dy, dz):
        return np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])

    def get_rotation_matrix_x(self, angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array([
            [1, 0, 0, 0],
            [0, cos_a, -sin_a, 0],
            [0, sin_a, cos_a, 0],
            [0, 0, 0, 1]
        ])

    def get_rotation_matrix_y(self, angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array([
            [cos_a, 0, sin_a, 0],
            [0, 1, 0, 0],
            [-sin_a, 0, cos_a, 0],
            [0, 0, 0, 1]
        ])

    def get_rotation_matrix_z(self, angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def get_scaling_matrix(self, sx, sy, sz):
        return np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])

    def get_reflection_matrix(self, plane):
        if plane == 'xy':
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        elif plane == 'xz':
            return np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        elif plane == 'yz':
            return np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def get_rotation_around_axis(self, axis_vector, angle, center):
        u = axis_vector / np.linalg.norm(axis_vector)
        ux, uy, uz = u
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        one_minus_cos = 1 - cos_a
        
        R = np.array([
            [cos_a + ux*ux*one_minus_cos, ux*uy*one_minus_cos - uz*sin_a, ux*uz*one_minus_cos + uy*sin_a, 0],
            [uy*ux*one_minus_cos + uz*sin_a, cos_a + uy*uy*one_minus_cos, uy*uz*one_minus_cos - ux*sin_a, 0],
            [uz*ux*one_minus_cos - uy*sin_a, uz*uy*one_minus_cos + ux*sin_a, cos_a + uz*uz*one_minus_cos, 0],
            [0, 0, 0, 1]
        ])
        
        T = self.get_translation_matrix(-center[0], -center[1], -center[2])
        T_inv = self.get_translation_matrix(center[0], center[1], center[2])
        
        return np.dot(T_inv, np.dot(R, T))

    def apply_translation(self):
        if not self.polyhedron:
            return
            
        dx = float(self.dx_entry.get() or 0)
        dy = float(self.dy_entry.get() or 0)
        dz = float(self.dz_entry.get() or 0)
        
        matrix = self.get_translation_matrix(dx, dy, dz)
        self.polyhedron.transform(matrix)
        self.render()

    def apply_rotation_around_point(self):
        if not self.polyhedron:
            return
            
        center_x = float(self.center_x_entry.get() or 0)
        center_y = float(self.center_y_entry.get() or 0)
        center_z = float(self.center_z_entry.get() or 0)
        angle = math.radians(float(self.angle_entry.get() or 0))
        
        T = self.get_translation_matrix(-center_x, -center_y, -center_z)
        R = self.get_rotation_matrix_z(angle)
        T_inv = self.get_translation_matrix(center_x, center_y, center_z)
        
        matrix = np.dot(T_inv, np.dot(R, T))
        self.polyhedron.transform(matrix)
        self.render()

    def apply_rotation_around_center(self, axis):
        if not self.polyhedron:
            return
            
        center = self.polyhedron.get_center()
        angle = math.radians(30)
        
        if axis == 'x':
            matrix = self.get_rotation_around_axis(np.array([1, 0, 0]), angle, center)
        elif axis == 'y':
            matrix = self.get_rotation_around_axis(np.array([0, 1, 0]), angle, center)
        elif axis == 'z':
            matrix = self.get_rotation_around_axis(np.array([0, 0, 1]), angle, center)
            
        self.polyhedron.transform(matrix)
        self.render()

    def apply_scaling_around_center(self):
        if not self.polyhedron:
            return
            
        scale_factor = float(self.scale_factor_entry.get() or 1.0)
        center = self.polyhedron.get_center()
        
        T = self.get_translation_matrix(-center[0], -center[1], -center[2])
        S = self.get_scaling_matrix(scale_factor, scale_factor, scale_factor)
        T_inv = self.get_translation_matrix(center[0], center[1], center[2])
        
        matrix = np.dot(T_inv, np.dot(S, T))
        self.polyhedron.transform(matrix)
        self.render()

    def apply_scaling_around_point(self):
        if not self.polyhedron:
            return
            
        scale_factor = float(self.scale_factor_entry.get() or 1.0)
        center_x = float(self.center_x_entry.get() or 0)
        center_y = float(self.center_y_entry.get() or 0)
        center_z = float(self.center_z_entry.get() or 0)
        
        T = self.get_translation_matrix(-center_x, -center_y, -center_z)
        S = self.get_scaling_matrix(scale_factor, scale_factor, scale_factor)
        T_inv = self.get_translation_matrix(center_x, center_y, center_z)
        
        matrix = np.dot(T_inv, np.dot(S, T))
        self.polyhedron.transform(matrix)
        self.render()

    def apply_reflection(self, plane):
        if not self.polyhedron:
            return
            
        center = self.polyhedron.get_center()
        
        T_to_origin = self.get_translation_matrix(-center[0], -center[1], -center[2])
        R = self.get_reflection_matrix(plane)
        T_back = self.get_translation_matrix(center[0], center[1], center[2])
        
        matrix = np.dot(T_back, np.dot(R, T_to_origin))
        self.polyhedron.transform(matrix)
        self.render()

    def apply_arbitrary_rotation(self):
        if not self.polyhedron:
            return
            
        try:
            p1 = np.array([float(self.p1_x.get()), float(self.p1_y.get()), float(self.p1_z.get())])
            p2 = np.array([float(self.p2_x.get()), float(self.p2_y.get()), float(self.p2_z.get())])
            angle = math.radians(float(self.arbitrary_angle_entry.get()))
            
            axis_vector = p2 - p1
            if np.linalg.norm(axis_vector) < 1e-10:
                return
                
            matrix = self.get_rotation_around_axis(axis_vector, angle, p1)
            self.polyhedron.transform(matrix)
            self.render()
            
        except ValueError:
            print("Ошибка: проверьте правильность введенных данных")

    # ===== МЕТОДЫ ДЛЯ Z-БУФЕРА И ОСВЕЩЕНИЯ =====
    
    def interpolate_normal(self, x, y, vertices, normals, barycentric):
        """Интерполирует нормаль в точке с использованием барицентрических координат"""
        if len(vertices) < 3:
            return np.array([0, 0, 1])
            
        # Билинейная интерполяция нормали
        interpolated_normal = (
            barycentric[0] * normals[0] +
            barycentric[1] * normals[1] + 
            barycentric[2] * normals[2]
        )
        
        # Нормализуем результат
        norm = np.linalg.norm(interpolated_normal)
        if norm > 0:
            interpolated_normal = interpolated_normal / norm
            
        return interpolated_normal
    
    def interpolate_color(self, x, y, colors, barycentric):
        """Интерполирует цвет в точке с использованием барицентрических координат"""
        if len(colors) < 3:
            return np.array([255, 255, 255])
            
        # Билинейная интерполяция цвета
        interpolated_color = (
            barycentric[0] * colors[0] +
            barycentric[1] * colors[1] + 
            barycentric[2] * colors[2]
        )
        
        # Ограничиваем значения цвета
        interpolated_color = np.clip(interpolated_color, 0, 255)
        
        return interpolated_color.astype(int)
    
    def interpolate_texture_coords(self, x, y, tex_coords, barycentric):
        """Интерполирует текстурные координаты в точке с использованием барицентрических координат"""
        if len(tex_coords) < 3:
            return np.array([0, 0])
            
        # Билинейная интерполяция текстурных координат
        interpolated_tex = (
            barycentric[0] * tex_coords[0] +
            barycentric[1] * tex_coords[1] + 
            barycentric[2] * tex_coords[2]
        )
        
        return interpolated_tex
    
    def get_barycentric_coords(self, x, y, vertices):
        """Вычисляет барицентрические координаты для точки (x, y) относительно треугольника"""
        if len(vertices) < 3:
            return [0, 0, 0]
            
        v0, v1, v2 = vertices[:3]
        
        # Вычисляем площадь треугольника
        area = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
        
        if area < 1e-10:
            return [1/3, 1/3, 1/3]
            
        # Вычисляем барицентрические координаты
        w0 = abs((v1[0] - x) * (v2[1] - y) - (v2[0] - x) * (v1[1] - y)) / area
        w1 = abs((v2[0] - x) * (v0[1] - y) - (v0[0] - x) * (v2[1] - y)) / area
        w2 = abs((v0[0] - x) * (v1[1] - y) - (v1[0] - x) * (v0[1] - y)) / area
        
        # Нормализуем (сумма должна быть равна 1)
        total = w0 + w1 + w2
        if total > 0:
            w0 /= total
            w1 /= total
            w2 /= total
            
        return [w0, w1, w2]
    
    def rasterize_polygon_with_lighting(self, polygon, color):
        """Растеризация полигона с использованием Z-буфера и освещения"""
        if len(polygon.vertices) < 3:
            return
            
        # Получаем проекции всех вершин
        projected_points = []
        original_points = []
        original_normals = []
        vertex_colors = []
        texture_coords = []
        (x_proj, y_proj, z_proj) = (0, 0, 0)
        
        for p in polygon.vertices:
            x, y, z = p.coordinates[:3]
            
            if self.camera_view:
                # РЕЖИМ КАМЕРЫ: используем матрицы вида и проекции
                view_matrix = self.camera.get_view_matrix()
                canvas_width = max(self.canvas.winfo_width(), 1)
                canvas_height = max(self.canvas.winfo_height(), 1)
                aspect_ratio = canvas_width / canvas_height
                
                # Преобразуем точку в однородных координатах
                point_4d = np.array([x, y, z, 1.0])
                
                # Применяем матрицу вида (переход в систему координат камеры)
                view_point = np.dot(view_matrix, point_4d)
                
                # В зависимости от выбранного типа проекции применяем разные преобразования
                if self.projection_type == 'perspective':
                    # ПЕРСПЕКТИВНАЯ ПРОЕКЦИЯ: используем матрицу перспективной проекции
                    projection_matrix = self.camera.get_projection_matrix(aspect_ratio)
                    proj_point = np.dot(projection_matrix, view_point)
                    
                    # Перспективное деление
                    if proj_point[3] != 0:
                        proj_point = proj_point / proj_point[3]
                    
                    # Масштабируем и центрируем на холсте
                    x_proj = (proj_point[0] + 1) * 0.5 * canvas_width
                    y_proj = (1 - proj_point[1]) * 0.5 * canvas_height
                    z_proj = proj_point[2]
                else:
                    # АКСОНОМЕТРИЧЕСКАЯ ПРОЕКЦИЯ: используем ортографическую проекцию
                    projection_matrix = self.camera.get_orthographic_matrix(aspect_ratio, scale=0.01)
                    proj_point = np.dot(projection_matrix, view_point)
                    
                    # Масштабируем и центрируем на холсте
                    x_proj = proj_point[0] * 500 + canvas_width / 2
                    y_proj = -proj_point[1] * 500 + canvas_height / 2
                    z_proj = proj_point[2]
                
            else:
                # ОБЫЧНЫЙ РЕЖИМ: используем старую проекцию
                if self.projection_type == 'perspective':
                    d = 500
                    if z + d != 0:
                        x_proj = x * d / (z + d)
                        y_proj = y * d / (z + d)
                    else:
                        x_proj, y_proj = x, y
                else:
                    x_proj = x - z * 0.5
                    y_proj = y - z * 0.5
                
                # Центрируем на холсте
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                x_proj = x_proj + canvas_width / 2
                y_proj = y_proj + canvas_height / 2
                z_proj = z
            
            screen_x = int(x_proj)
            screen_y = int(y_proj)
            projected_points.append((screen_x, screen_y))
            original_points.append((x, y, z_proj))
            original_normals.append(p.normal)
            
            # Сохраняем цвета вершин для шейдинга Гуро
            if p.color is not None:
                vertex_colors.append((p.color * 255).astype(int))
            else:
                vertex_colors.append(color)
                
            # Сохраняем текстурные координаты
            texture_coords.append(p.texture_coords)
        
        # Находим ограничивающий прямоугольник
        min_x = max(0, min(p[0] for p in projected_points))
        max_x = min(self.z_buffer.width - 1, max(p[0] for p in projected_points))
        min_y = max(0, min(p[1] for p in projected_points))
        max_y = min(self.z_buffer.height - 1, max(p[1] for p in projected_points))
        
        if min_x >= max_x or min_y >= max_y:
            return
            
        # Растеризация методом сканирующих строк
        for y in range(min_y, max_y + 1):
            intersections = []
            n = len(projected_points)
            
            for i in range(n):
                p1 = projected_points[i]
                p2 = projected_points[(i + 1) % n]
                op1 = original_points[i]
                op2 = original_points[(i + 1) % n]
                on1 = original_normals[i]
                on2 = original_normals[(i + 1) % n]
                vc1 = vertex_colors[i]
                vc2 = vertex_colors[(i + 1) % n]
                tc1 = texture_coords[i]
                tc2 = texture_coords[(i + 1) % n]
                
                # Проверяем, пересекает ли ребро сканирующую строку
                if (p1[1] <= y and p2[1] > y) or (p2[1] <= y and p1[1] > y):
                    # Вычисляем x-координату пересечения
                    if p2[1] != p1[1]:
                        t = (y - p1[1]) / (p2[1] - p1[1])
                        x_intersect = p1[0] + t * (p2[0] - p1[0])
                        z_intersect = op1[2] + t * (op2[2] - op1[2])
                        
                        # Интерполируем нормаль
                        normal_intersect = on1 + t * (on2 - on1)
                        norm = np.linalg.norm(normal_intersect)
                        if norm > 0:
                            normal_intersect = normal_intersect / norm
                            
                        # Интерполируем цвет
                        color_intersect = vc1 + t * (vc2 - vc1)
                        
                        # Интерполируем текстурные координаты
                        tex_intersect = tc1 + t * (tc2 - tc1)
                            
                        # Интерполируем мировые координаты
                        world_x = op1[0] + t * (op2[0] - op1[0])
                        world_y = op1[1] + t * (op2[1] - op1[1])
                        world_z = op1[2] + t * (op2[2] - op1[2])
                        
                        intersections.append((x_intersect, z_intersect, normal_intersect, color_intersect, tex_intersect, (world_x, world_y, world_z)))
            
            # Сортируем точки пересечения
            intersections.sort(key = lambda inter: inter[0:1])
            
            # Заполняем пиксели между парами пересечений
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x_start = int(intersections[i][0])
                    x_end = int(intersections[i + 1][0])
                    z_start = intersections[i][1]
                    z_end = intersections[i + 1][1]
                    normal_start = intersections[i][2]
                    normal_end = intersections[i + 1][2]
                    color_start = intersections[i][3]
                    color_end = intersections[i + 1][3]
                    tex_start = intersections[i][4]
                    tex_end = intersections[i + 1][4]
                    world_start = intersections[i][5]
                    world_end = intersections[i + 1][5]
                    
                    for x in range(x_start, x_end + 1):
                        if 0 <= x < self.z_buffer.width:
                            # Интерполируем z-координату
                            if x_end != x_start:
                                t_x = (x - x_start) / (x_end - x_start)
                            else:
                                t_x = 0.5
                                
                            z = z_start + t_x * (z_end - z_start)
                            
                            # Вычисляем цвет в зависимости от режима шейдинга
                            if self.shading_mode == 'phong' and self.use_lighting:
                                # Интерполируем нормаль
                                normal = normal_start + t_x * (normal_end - normal_start)
                                norm = np.linalg.norm(normal)
                                if norm > 0:
                                    normal = normal / norm
                                    
                                # Интерполируем мировые координаты
                                world_x = world_start[0] + t_x * (world_end[0] - world_start[0])
                                world_y = world_start[1] + t_x * (world_end[1] - world_start[1])
                                world_z = world_start[2] + t_x * (world_end[2] - world_start[2])
                                
                                # Вычисляем направление взгляда (от точки к камере)
                                if self.camera_view:
                                    view_dir = self.camera.position - np.array([world_x, world_y, world_z])
                                else:
                                    view_dir = np.array([0, 0, 1])  # Направление по умолчанию
                                    
                                view_dir = view_dir / np.linalg.norm(view_dir)
                                
                                # Вычисляем цвет по модели Фонга
                                color_float = self.lighting.phong_shading(
                                    np.array([world_x, world_y, world_z]),
                                    normal,
                                    view_dir
                                )
                                
                                # Преобразуем в целочисленный цвет
                                final_color = (color_float * 255).astype(int)
                            elif self.shading_mode == 'gouraud' and self.use_lighting:
                                # Интерполируем цвет
                                final_color = color_start + t_x * (color_end - color_start)
                                final_color = final_color.astype(int)
                            elif self.shading_mode == 'texture':
                                # Интерполируем текстурные координаты
                                tex_coord = tex_start + t_x * (tex_end - tex_start)
                                # Получаем цвет из текстуры
                                final_color = self.texture.get_color(tex_coord[0], tex_coord[1])
                            else:
                                final_color = color
                            
                            self.z_buffer.update(x, y, z, final_color)

    def rasterize_polygon(self, polygon, color):
        """Растеризация полигона с использованием Z-буфера (без освещения)"""
        if len(polygon.vertices) < 3:
            return
            
        # Получаем проекции всех вершин
        projected_points = []
        original_points = []
        (x_proj, y_proj, z_proj) = (0, 0, 0)
        
        for p in polygon.vertices:
            x, y, z = p.coordinates[:3]
            
            if self.camera_view:
                # РЕЖИМ КАМЕРЫ: используем матрицы вида и проекции
                view_matrix = self.camera.get_view_matrix()
                canvas_width = max(self.canvas.winfo_width(), 1)
                canvas_height = max(self.canvas.winfo_height(), 1)
                aspect_ratio = canvas_width / canvas_height
                
                # Преобразуем точку в однородных координатах
                point_4d = np.array([x, y, z, 1.0])
                
                # Применяем матрицу вида (переход в систему координат камеры)
                view_point = np.dot(view_matrix, point_4d)
                
                # В зависимости от выбранного типа проекции применяем разные преобразования
                if self.projection_type == 'perspective':
                    # ПЕРСПЕКТИВНАЯ ПРОЕКЦИЯ: используем матрицу перспективной проекции
                    projection_matrix = self.camera.get_projection_matrix(aspect_ratio)
                    proj_point = np.dot(projection_matrix, view_point)
                    
                    # Перспективное деление
                    if proj_point[3] != 0:
                        proj_point = proj_point / proj_point[3]
                    
                    # Масштабируем и центрируем на холсте
                    x_proj = (proj_point[0] + 1) * 0.5 * canvas_width
                    y_proj = (1 - proj_point[1]) * 0.5 * canvas_height
                    z_proj = proj_point[2]
                else:
                    # АКСОНОМЕТРИЧЕСКАЯ ПРОЕКЦИЯ: используем ортографическую проекцию
                    projection_matrix = self.camera.get_orthographic_matrix(aspect_ratio, scale=0.01)
                    proj_point = np.dot(projection_matrix, view_point)
                    
                    # Масштабируем и центрируем на холсте
                    x_proj = proj_point[0] * 500 + canvas_width / 2
                    y_proj = -proj_point[1] * 500 + canvas_height / 2
                    z_proj = proj_point[2]
                
            else:
                # ОБЫЧНЫЙ РЕЖИМ: используем старую проекцию
                if self.projection_type == 'perspective':
                    d = 500
                    if z + d != 0:
                        x_proj = x * d / (z + d)
                        y_proj = y * d / (z + d)
                    else:
                        x_proj, y_proj = x, y
                else:
                    x_proj = x - z * 0.5
                    y_proj = y - z * 0.5
                
                # Центрируем на холсте
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                x_proj = x_proj + canvas_width / 2
                y_proj = y_proj + canvas_height / 2
                z_proj = z
            
            screen_x = int(x_proj)
            screen_y = int(y_proj)
            projected_points.append((screen_x, screen_y))
            original_points.append((x, y, z_proj))
        
        # Находим ограничивающий прямоугольник
        min_x = max(0, min(p[0] for p in projected_points))
        max_x = min(self.z_buffer.width - 1, max(p[0] for p in projected_points))
        min_y = max(0, min(p[1] for p in projected_points))
        max_y = min(self.z_buffer.height - 1, max(p[1] for p in projected_points))
        
        if min_x >= max_x or min_y >= max_y:
            return
            
        # Растеризация методом сканирующих строк
        for y in range(min_y, max_y + 1):
            intersections = []
            n = len(projected_points)
            
            for i in range(n):
                p1 = projected_points[i]
                p2 = projected_points[(i + 1) % n]
                op1 = original_points[i]
                op2 = original_points[(i + 1) % n]
                
                # Проверяем, пересекает ли ребро сканирующую строку
                if (p1[1] <= y and p2[1] > y) or (p2[1] <= y and p1[1] > y):
                    # Вычисляем x-координату пересечения
                    if p2[1] != p1[1]:
                        t = (y - p1[1]) / (p2[1] - p1[1])
                        x_intersect = p1[0] + t * (p2[0] - p1[0])
                        z_intersect = op1[2] + t * (op2[2] - op1[2])
                        intersections.append((x_intersect, z_intersect))
            
            # Сортируем точки пересечения
            intersections.sort()
            
            # Заполняем пиксели между парами пересечений
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x_start = int(intersections[i][0])
                    x_end = int(intersections[i + 1][0])
                    z_start = intersections[i][1]
                    z_end = intersections[i + 1][1]
                    
                    for x in range(x_start, x_end + 1):
                        if 0 <= x < self.z_buffer.width:
                            # Интерполируем z-координату
                            z = self.interpolate_z(x, x_start, x_end, z_start, z_end)
                            self.z_buffer.update(x, y, z, color)

    def interpolate_z(self, x, x_start, x_end, z_start, z_end):
        """Интерполяция z-координаты между двумя точками"""
        if x_start == x_end:
            return (z_start + z_end) / 2
        
        t = (x - x_start) / (x_end - x_start)
        z_intersect = z_start + t * (z_end - z_start)
        return z_intersect

    def render_with_z_buffer(self):
        """Рендеринг с использованием Z-буфера"""
        self.z_buffer.clear()
        
        # Обновляем размер Z-буфера в соответствии с размером холста
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 0 and canvas_height > 0:
            self.z_buffer = ZBuffer(canvas_width, canvas_height)
        
        counter = 0
        for polygon in self.polyhedron.polygons:
            # Проверка видимости полигона с учетом отсечения нелицевых граней
            if not self.is_polygon_visible(polygon):
                continue
                
            # Используем разные цвета для разных полигонов для наглядности
            color = self.colors[counter % len(self.colors)]
            counter += 1
            
            if self.use_z_buffer and (self.use_lighting or self.shading_mode == 'texture'):
                self.rasterize_polygon_with_lighting(polygon, color)
            else:
                self.rasterize_polygon(polygon, color)
        
        # Создаем изображение из буфера цвета
        image = tk.PhotoImage(width=self.z_buffer.width, height=self.z_buffer.height)
        
        for y in range(self.z_buffer.height):
            for x in range(self.z_buffer.width):
                color = self.z_buffer.color_buffer[y, x]
                if not np.array_equal(color, [255, 255, 255]):  # Если не белый фон
                    hex_color = "#%02x%02x%02x" % tuple(color)
                    image.put(hex_color, (x, y))
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image  # Сохраняем ссылку на изображение

    def render_without_z_buffer(self):
        """Рендеринг без Z-буфера (старый метод)"""
        sorted_polygons = sorted(self.polyhedron.polygons, 
                               key=lambda p: p.get_center_z(), 
                               reverse=True)
        
        visible_count = 0
        total_count = len(sorted_polygons)
        
        for polygon in sorted_polygons:
            # Проверка видимости полигона с учетом отсечения нелицевых граней
            if not self.is_polygon_visible(polygon):
                continue
                
            visible_count += 1
            projected_points = []
            
            for p in polygon.vertices:
                x, y, z = p.coordinates[:3]
                
                if self.camera_view:
                    # РЕЖИМ КАМЕРЫ: используем матрицы вида и проекции
                    view_matrix = self.camera.get_view_matrix()
                    canvas_width = max(self.canvas.winfo_width(), 1)
                    canvas_height = max(self.canvas.winfo_height(), 1)
                    aspect_ratio = canvas_width / canvas_height
                    
                    # Преобразуем точку в однородных координатах
                    point_4d = np.array([x, y, z, 1.0])
                    
                    # Применяем матрицу вида (переход в систему координат камеры)
                    view_point = np.dot(view_matrix, point_4d)
                    
                    # В зависимости от выбранного типа проекции применяем разные преобразования
                    if self.projection_type == 'perspective':
                        # ПЕРСПЕКТИВНАЯ ПРОЕКЦИЯ: используем матрицу перспективной проекции
                        projection_matrix = self.camera.get_projection_matrix(aspect_ratio)
                        proj_point = np.dot(projection_matrix, view_point)
                        
                        # Перспективное деление
                        if proj_point[3] != 0:
                            proj_point = proj_point / proj_point[3]
                        
                        # Масштабируем и центрируем на холсте
                        x_proj = (proj_point[0] + 1) * 0.5 * canvas_width
                        y_proj = (1 - proj_point[1]) * 0.5 * canvas_height
                    else:
                        # АКСОНОМЕТРИЧЕСКАЯ ПРОЕКЦИЯ: используем ортографическую проекцию
                        projection_matrix = self.camera.get_orthographic_matrix(aspect_ratio, scale=0.01)
                        proj_point = np.dot(projection_matrix, view_point)
                        
                        # Масштабируем и центрируем на холсте
                        x_proj = proj_point[0] * 500 + canvas_width / 2
                        y_proj = -proj_point[1] * 500 + canvas_height / 2
                    
                else:
                    # ОБЫЧНЫЙ РЕЖИМ: используем старую проекцию
                    if self.projection_type == 'perspective':
                        d = 500
                        if z + d != 0:
                            x_proj = x * d / (z + d)
                            y_proj = y * d / (z + d)
                        else:
                            x_proj, y_proj = x, y
                    else:
                        x_proj = x - z * 0.5
                        y_proj = y - z * 0.5
                    
                    # Центрируем на холсте
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    x_proj = x_proj + canvas_width / 2
                    y_proj = y_proj + canvas_height / 2
                
                projected_points.append((x_proj, y_proj))
            
            if len(projected_points) >= 3:
                coords = []
                for point in projected_points:
                    coords.extend(point)
                
                # Вычисляем цвет в зависимости от режима шейдинга
                if self.shading_mode == 'texture':
                    fill_color = "lightgray"  # Для текстурирования используем простой цвет
                elif self.use_lighting and self.shading_mode == 'gouraud':
                    # Для Гуро вычисляем средний цвет вершин
                    if polygon.vertices and polygon.vertices[0].color is not None:
                        avg_color = np.mean([v.color for v in polygon.vertices], axis=0)
                        fill_color = "#%02x%02x%02x" % tuple((avg_color * 255).astype(int))
                    else:
                        fill_color = "lightblue"
                elif self.use_lighting:
                    # Простое затенение по нормали полигона
                    normal = polygon.get_normal()
                    light_dir = self.lighting.light_position - polygon.get_center()
                    light_dir = light_dir / np.linalg.norm(light_dir)
                    
                    intensity = max(np.dot(normal, light_dir), 0)
                    intensity = self.lighting.ambient_intensity + self.lighting.diffuse_intensity * intensity
                    intensity = min(intensity, 1.0)
                    
                    base_color = (self.lighting.object_color * 255).astype(int)
                    shaded_color = (base_color * intensity).astype(int)
                    fill_color = "#%02x%02x%02x" % tuple(shaded_color)
                else:
                    fill_color = "lightblue"
                
                self.canvas.create_polygon(
                    coords, 
                    outline="black", 
                    fill=fill_color, 
                    width=1,
                    smooth=False
                )
        
        # Отображение информации о режиме
        mode_text = f"Режим: {'КАМЕРА' if self.camera_view else 'ОБЫЧНЫЙ'}\n"
        mode_text += f"Проекция: {'ПЕРСПЕКТИВНАЯ' if self.projection_type == 'perspective' else 'АКСОНОМЕТРИЧЕСКАЯ'}\n"
        mode_text += f"Освещение: {'ВКЛ' if self.use_lighting else 'ВЫКЛ'}\n"
        mode_text += f"Шейдинг: {self.shading_mode.upper()}\n"
        mode_text += f"Грани: {visible_count}/{total_count}"
        
        if self.camera_view:
            mode_text += f"\nПозиция: ({self.camera.position[0]:.0f}, {self.camera.position[1]:.0f}, {self.camera.position[2]:.0f})"
            mode_text += f"\nУправление: WASD, мышь"
        
        self.canvas.create_text(
            10, 10, 
            text=mode_text, 
            anchor=tk.NW, 
            fill="black", 
            font=("Arial", 10, "bold"),
            justify=tk.LEFT
        )

    def render(self):
        """Основной метод рендеринга"""
        self.canvas.delete("all")
        
        if not self.polyhedron:
            return
            
        if self.use_z_buffer:
            self.render_with_z_buffer()
        else:
            self.render_without_z_buffer()

    def set_projection(self, projection_type):
        self.projection_type = projection_type
        self.render()

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
