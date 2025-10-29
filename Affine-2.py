import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ipywidgets as widgets
from IPython.display import display

class Point:
    def __init__(self, x, y, z):
        self.coordinates = np.array([x, y, z, 1.0])

    def transform(self, matrix):
        transformed_coordinates = np.dot(matrix, self.coordinates)
        return Point(transformed_coordinates[0], transformed_coordinates[1], transformed_coordinates[2])

    def copy(self):
        return Point(self.coordinates[0], self.coordinates[1], self.coordinates[2])

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def transform(self, matrix):
        self.vertices = [vertex.transform(matrix) for vertex in self.vertices]

    def copy(self):
        return Polygon([v.copy() for v in self.vertices])

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
        self.save_state()

    def transform(self, matrix):
        for polygon in self.polygons:
            polygon.transform(matrix)

    def save_state(self):
        self.saved_polygons = [polygon.copy() for polygon in self.polygons]

    def reset(self):
        self.polygons = [polygon.copy() for polygon in self.saved_polygons]

class PolyhedronViewer:
    def __init__(self):
        self.polyhedron = None
        self.projection_type = 'perspective'
        self.fig = None
        self.ax = None
        self.current_scale = 1.0
        self.rotation_angle = 0
        self.create_controls()
        self.create_polyhedron('Tetrahedron')

    def create_controls(self):
        # Выбор многогранника
        self.polyhedron_dropdown = widgets.Dropdown(
            options=['Tetrahedron', 'Cube', 'Octahedron', 'Icosahedron', 'Dodecahedron'],
            value='Tetrahedron',
            description='Многогранник:'
        )
        self.polyhedron_dropdown.observe(self.on_polyhedron_change, names='value')

        # Выбор проекции
        self.projection_dropdown = widgets.Dropdown(
            options=[('Перспективная', 'perspective'), ('Аксонометрическая', 'axonometric')],
            value='perspective',
            description='Проекция:'
        )
        self.projection_dropdown.observe(self.on_projection_change, names='value')

        # Масштабирование
        self.scale_slider = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=3.0,
            step=0.1,
            description='Масштаб:'
        )

        # Кнопки отражения
        self.reflect_xy_btn = widgets.Button(description='Отражение XY', button_style='primary')
        self.reflect_xz_btn = widgets.Button(description='Отражение XZ', button_style='primary')
        self.reflect_yz_btn = widgets.Button(description='Отражение YZ', button_style='primary')

        self.reflect_xy_btn.on_click(lambda x: self.apply_reflection('xy'))
        self.reflect_xz_btn.on_click(lambda x: self.apply_reflection('xz'))
        self.reflect_yz_btn.on_click(lambda x: self.apply_reflection('yz'))

        # Кнопка сброса
        self.reset_btn = widgets.Button(description='Сброс', button_style='warning')
        self.reset_btn.on_click(lambda x: self.on_reset())

        # Кнопка масштабирования
        self.scale_btn = widgets.Button(description='Применить масштаб', button_style='success')
        self.scale_btn.on_click(lambda x: self.on_scale_apply())

        # Кнопка вращения
        self.rotate_btn = widgets.Button(description='Вращение', button_style='info')
        self.rotate_btn.on_click(lambda x: self.on_rotate())

        # Группировка элементов управления
        reflection_box = widgets.HBox([self.reflect_xy_btn, self.reflect_xz_btn, self.reflect_yz_btn])
        scale_box = widgets.HBox([self.scale_slider, self.scale_btn])

        controls_box = widgets.VBox([
            self.polyhedron_dropdown,
            self.projection_dropdown,
            scale_box,
            reflection_box,
            widgets.HBox([self.rotate_btn, self.reset_btn])
        ])

        display(controls_box)

    def on_polyhedron_change(self, change):
        self.create_polyhedron(change['new'])
        self.plot()

    def on_projection_change(self, change):
        self.projection_type = change['new']
        self.plot()

    def on_scale_apply(self):
        scale_factor = self.scale_slider.value
        self.apply_scaling_own_center(scale_factor)
        self.plot()

    def on_rotate(self):
        self.rotation_angle += 15
        self.apply_rotation()
        self.plot()

    def on_reset(self):
        if self.polyhedron:
            self.polyhedron.reset()
            self.scale_slider.value = 1.0
            self.current_scale = 1.0
            self.rotation_angle = 0
            self.plot()

    def create_polyhedron(self, shape):
        if shape == 'Tetrahedron':
            # Тетраэдр
            vertices = [
                Point(1, 1, 1),
                Point(1, -1, -1),
                Point(-1, 1, -1),
                Point(-1, -1, 1)
            ]
            faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

        elif shape == 'Cube':
            # Куб
            vertices = [
                Point(-1, -1, -1), Point(1, -1, -1), Point(1, 1, -1), Point(-1, 1, -1),
                Point(-1, -1, 1), Point(1, -1, 1), Point(1, 1, 1), Point(-1, 1, 1)
            ]
            faces = [
                [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
            ]

        elif shape == 'Octahedron':
            # Октаэдр
            vertices = [
                Point(1, 0, 0), Point(-1, 0, 0), Point(0, 1, 0),
                Point(0, -1, 0), Point(0, 0, 1), Point(0, 0, -1)
            ]
            faces = [
                [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
                [1, 2, 5], [1, 5, 3], [1, 3, 4], [1, 4, 2]
            ]

        elif shape == 'Icosahedron':
            # Икосаэдр - правильные координаты из трех "золотых" прямоугольников
            phi = (1 + math.sqrt(5)) / 2  # золотое сечение

            vertices = [
                # (0, ±1, ±φ)
                Point(0, 1, phi), Point(0, 1, -phi), Point(0, -1, phi), Point(0, -1, -phi),
                # (±φ, 0, ±1)
                Point(phi, 0, 1), Point(phi, 0, -1), Point(-phi, 0, 1), Point(-phi, 0, -1),
                # (±1, ±φ, 0)
                Point(1, phi, 0), Point(1, -phi, 0), Point(-1, phi, 0), Point(-1, -phi, 0)
            ]

            # Правильные грани икосаэдра (20 треугольников)
            faces = [
                [0, 2, 4], [0, 4, 8], [0, 8, 6], [0, 6, 10], [0, 10, 2],
                [1, 3, 5], [1, 5, 8], [1, 8, 4], [1, 4, 9], [1, 9, 3],
                [2, 6, 7], [2, 7, 11], [2, 11, 4], [3, 5, 7], [3, 7, 11],
                [5, 7, 8], [6, 10, 7], [8, 9, 5], [9, 11, 3], [10, 11, 6]
            ]

        elif shape == 'Dodecahedron':
            # Додекаэдр - правильные координаты
            phi = (1 + math.sqrt(5)) / 2  # золотое сечение
            inv_phi = 1 / phi  # 1/φ

            vertices = []

            # 1. (±1, ±1, ±1) - 8 вершин
            for x in [-1, 1]:
                for y in [-1, 1]:
                    for z in [-1, 1]:
                        vertices.append(Point(x, y, z))

            # 2. (0, ±1/φ, ±φ) - 4 вершины
            for y in [-inv_phi, inv_phi]:
                for z in [-phi, phi]:
                    vertices.append(Point(0, y, z))

            # 3. (±1/φ, ±φ, 0) - 4 вершины
            for x in [-inv_phi, inv_phi]:
                for y in [-phi, phi]:
                    vertices.append(Point(x, y, 0))

            # 4. (±φ, 0, ±1/φ) - 4 вершины
            for x in [-phi, phi]:
                for z in [-inv_phi, inv_phi]:
                    vertices.append(Point(x, 0, z))

            # Правильные грани додекаэдра (12 пятиугольников)
            faces = [
                [0, 8, 10, 2, 16],    # Грань 1
                [0, 16, 18, 1, 8],     # Грань 2
                [1, 9, 11, 3, 17],     # Грань 3
                [2, 10, 11, 3, 18],    # Грань 4
                [4, 12, 13, 5, 19],    # Грань 5
                [4, 19, 15, 6, 12],    # Грань 6
                [5, 13, 14, 7, 17],    # Грань 7
                [6, 15, 16, 0, 12],    # Грань 8
                [7, 14, 18, 2, 19],    # Грань 9
                [8, 1, 17, 7, 14],     # Грань 10
                [9, 4, 19, 2, 18],     # Грань 11
                [10, 5, 17, 3, 11]     # Грань 12
            ]

        else:
            vertices = []
            faces = []

        polygons = Polygon.polygons_from_vertices(vertices, faces)
        self.polyhedron = Polyhedron(polygons)
        self.current_scale = 1.0
        self.rotation_angle = 0
        self.center_and_scale()

    def center_and_scale(self):
        if not self.polyhedron:
            return

        # Находим центр многогранника
        all_vertices = []
        for polygon in self.polyhedron.polygons:
            all_vertices.extend(polygon.vertices)

        if not all_vertices:
            return

        center = np.mean([v.coordinates[:3] for v in all_vertices], axis=0)

        # Масштабируем и перемещаем в центр
        scale = 0.8  # Уменьшенный масштаб для лучшего отображения
        translate_matrix = self.get_translation_matrix(-center[0], -center[1], -center[2])
        scale_matrix = self.get_scaling_matrix(0, 0, 0, scale)

        self.polyhedron.transform(translate_matrix)
        self.polyhedron.transform(scale_matrix)
        self.polyhedron.save_state()  # Сохраняем центрированное состояние

    def get_translation_matrix(self, dx, dy, dz):
        return np.array([[1, 0, 0, dx],
                         [0, 1, 0, dy],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])

    def get_scaling_matrix(self, center_x, center_y, center_z, scale_factor):
        # Масштабирование относительно точки (center_x, center_y, center_z)
        translate_to_origin = self.get_translation_matrix(-center_x, -center_y, -center_z)
        scale = np.array([[scale_factor, 0, 0, 0],
                         [0, scale_factor, 0, 0],
                         [0, 0, scale_factor, 0],
                         [0, 0, 0, 1]])
        translate_back = self.get_translation_matrix(center_x, center_y, center_z)

        # Комбинируем преобразования
        return np.dot(translate_back, np.dot(scale, translate_to_origin))

    def get_rotation_matrix(self, angle_x, angle_y, angle_z):
        # Матрицы вращения вокруг осей
        angle_x_rad = np.radians(angle_x)
        angle_y_rad = np.radians(angle_y)
        angle_z_rad = np.radians(angle_z)

        # Вращение вокруг X
        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(angle_x_rad), -np.sin(angle_x_rad), 0],
                       [0, np.sin(angle_x_rad), np.cos(angle_x_rad), 0],
                       [0, 0, 0, 1]])

        # Вращение вокруг Y
        Ry = np.array([[np.cos(angle_y_rad), 0, np.sin(angle_y_rad), 0],
                       [0, 1, 0, 0],
                       [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad), 0],
                       [0, 0, 0, 1]])

        # Вращение вокруг Z
        Rz = np.array([[np.cos(angle_z_rad), -np.sin(angle_z_rad), 0, 0],
                       [np.sin(angle_z_rad), np.cos(angle_z_rad), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        # Комбинируем вращения
        return np.dot(Rz, np.dot(Ry, Rx))

    def get_reflection_matrix(self, plane):
        if plane == 'xy':
            return np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
        elif plane == 'xz':
            return np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        elif plane == 'yz':
            return np.array([[-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    def apply_reflection(self, plane):
        if not self.polyhedron:
            return

        matrix = self.get_reflection_matrix(plane)
        self.polyhedron.transform(matrix)
        self.plot()

    def apply_rotation(self):
        if not self.polyhedron:
            return

        # Находим центр многогранника
        all_vertices = []
        for polygon in self.polyhedron.polygons:
            all_vertices.extend(polygon.vertices)

        center = np.mean([v.coordinates[:3] for v in all_vertices], axis=0)

        # Вращение вокруг центра
        translate_to_origin = self.get_translation_matrix(-center[0], -center[1], -center[2])
        rotation_matrix = self.get_rotation_matrix(0, self.rotation_angle, 0)
        translate_back = self.get_translation_matrix(center[0], center[1], center[2])

        # Комбинируем преобразования
        matrix = np.dot(translate_back, np.dot(rotation_matrix, translate_to_origin))
        self.polyhedron.transform(matrix)

    def apply_scaling_own_center(self, scale_factor):
        if not self.polyhedron:
            return

        # Находим центр многогранника
        all_vertices = []
        for polygon in self.polyhedron.polygons:
            all_vertices.extend(polygon.vertices)

        center = np.mean([v.coordinates[:3] for v in all_vertices], axis=0)

        center_x = center[0]
        center_y = center[1]
        center_z = center[2]

        matrix = self.get_scaling_matrix(center_x, center_y, center_z, scale_factor)
        self.polyhedron.transform(matrix)
        self.current_scale = scale_factor

    def plot(self):
        # Создаем новую фигуру каждый раз для гарантии обновления
        plt.close('all')  # Закрываем предыдущие фигуры
        self.fig = plt.figure(figsize=(10, 8))

        if self.projection_type == 'perspective':
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111, projection='3d')

        if not self.polyhedron:
            plt.show()
            return

        # Собираем все грани для отрисовки
        polygons_3d = []
        for polygon in self.polyhedron.polygons:
            vertices_array = np.array([v.coordinates[:3] for v in polygon.vertices])
            polygons_3d.append(vertices_array)

        # Создаем коллекцию полигонов
        poly_collection = Poly3DCollection(polygons_3d,
                                         alpha=0.9,
                                         linewidths=2,
                                         edgecolor='darkblue')

        # Разные цвета для разных многогранников
        colors = {
            'Tetrahedron': 'lightcoral',
            'Cube': 'lightgreen',
            'Octahedron': 'lightyellow',
            'Icosahedron': 'lightcyan',
            'Dodecahedron': 'lavender'
        }

        current_color = colors.get(self.polyhedron_dropdown.value, 'lightblue')
        poly_collection.set_facecolor(current_color)

        self.ax.add_collection3d(poly_collection)

        # Настройки отображения
        scale = 2.0
        self.ax.set_xlim([-scale, scale])
        self.ax.set_ylim([-scale, scale])
        self.ax.set_zlim([-scale, scale])

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Настройка проекции
        if self.projection_type == 'perspective':
            self.ax.set_proj_type('persp')
            # Добавляем небольшой наклон для лучшего обзора
            self.ax.view_init(elev=20, azim=45)
        else:
            self.ax.set_proj_type('ortho')
            # Аксонометрическая проекция
            self.ax.view_init(elev=20, azim=45)

        title = f"{self.polyhedron_dropdown.value}\nПроекция: {'Перспективная' if self.projection_type == 'perspective' else 'Аксонометрическая'}"
        self.ax.set_title(title, fontsize=14, pad=20)

        # Добавляем информационную панель
        info_text = f"Масштаб: {self.current_scale:.1f}\nВращение: {self.rotation_angle}°"
        self.ax.text2D(0.05, 0.95, info_text, transform=self.ax.transAxes, fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.show()

# Создаем и запускаем визуализатор
print("3D Визуализатор многогранников")
print("=" * 40)
print("Функции:")
print("- Выбор многогранника")
print("- Перспективная/Аксонометрическая проекции")
print("- Масштабирование относительно центра")
print("- Отражение относительно координатных плоскостей")
print("- Вращение")
print("- Сброс к исходному состоянию")
print("\nИспользуйте элементы управления для взаимодействия:")
viewer = PolyhedronViewer()