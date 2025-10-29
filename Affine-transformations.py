import tkinter as tk
from tkinter import ttk
import numpy as np
import math

class Point:
    def __init__(self, x, y, z):
        self.coordinates = np.array([x, y, z, 1.0])  # Добавляем однородную координату

    def transform(self, matrix):
        transformed_coordinates = np.dot(matrix, self.coordinates)
        return Point(transformed_coordinates[0], transformed_coordinates[1], transformed_coordinates[2])

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def transform(self, matrix):
        self.vertices = [vertex.transform(matrix) for vertex in self.vertices]

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

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Polyhedra")
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack()
        self.polyhedron = None
        self.projection_type = 'perspective'  # По умолчанию перспективная проекция
        self.create_polyhedron('Tetrahedron')
        self.create_dropdown()
        self.create_buttons()
        self.create_projection_buttons()
        self.create_rotation_controls()

    def create_dropdown(self):
        self.polyhedron_var = tk.StringVar(value='Tetrahedron')
        self.dropdown = ttk.Combobox(self.root, textvariable=self.polyhedron_var)
        self.dropdown['values'] = ['Tetrahedron', 'Cube', 'Octahedron', 'Icosahedron', 'Dodecahedron']
        self.dropdown.bind("<<ComboboxSelected>>", self.change_polyhedron)
        self.dropdown.pack()       

    def change_polyhedron(self, event):
        self.create_polyhedron(self.polyhedron_var.get())
        self.render()
        
    def create_polyhedron(self, shape):
        scale = 100  # Базовый масштаб
        
        if shape == 'Tetrahedron':
            # Тетраэдр - 4 вершины, 4 грани-треугольника
            vertices = [
                Point(-scale, scale, -scale), 
                Point(scale, scale, scale), 
                Point(scale, -scale, -scale), 
                Point(-scale, -scale, scale)
            ]
            faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        elif shape == 'Cube':
            # Куб (Hexahedron) - 8 вершин, 6 граней-четырехугольников
            vertices = [
                Point(-scale, scale, -scale),
                Point(scale, scale, -scale),
                Point(scale, -scale, -scale),
                Point(-scale, -scale, -scale),
                Point(-scale, scale, scale),
                Point(scale, scale, scale),
                Point(scale, -scale, scale),
                Point(-scale, -scale, scale)
            ]
            faces = [
                [0, 1, 2, 3],  # задняя грань
                [4, 5, 6, 7],  # передняя грань
                [0, 1, 5, 4],  # верхняя грань
                [2, 3, 7, 6],  # нижняя грань
                [0, 3, 7, 4],  # левая грань
                [1, 2, 6, 5]   # правая грань
            ]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        elif shape == 'Octahedron':
            # Октаэдр - 6 вершин, 8 граней-треугольников
            vertices = [
                Point(0, 0, -scale),        # 0 - нижняя вершина
                Point(-scale, 0, 0),        # 1
                Point(0, scale, 0),         # 2
                Point(scale, 0, 0),         # 3
                Point(0, -scale, 0),        # 4
                Point(0, 0, scale)          # 5 - верхняя вершина
            ]
            faces = [
                [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],  # нижние грани
                [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]   # верхние грани
            ]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        elif shape == 'Icosahedron':
            # Икосаэдр - 12 вершин, 20 граней-треугольников
            vertices = []
            
            # Верхняя вершина
            vertices.append(Point(0, 0, math.sqrt(5)/2 * scale))
            
            # Верхнее кольцо (5 вершин)
            for i in range(5):
                angle = 2 * i * 72 * math.pi / 360
                vertices.append(Point(
                    scale * math.cos(angle),
                    scale * math.sin(angle),
                    scale * 0.5
                ))
            
            # Нижнее кольцо (5 вершин)
            for i in range(5):
                angle = 2 * (36 + i * 72) * math.pi / 360
                vertices.append(Point(
                    scale * math.cos(angle),
                    scale * math.sin(angle),
                    scale * -0.5
                ))
            
            # Нижняя вершина
            vertices.append(Point(0, 0, -math.sqrt(5)/2 * scale))
            
            # Грани икосаэдра
            faces = [
                # Верхняя шапка (5 треугольников)
                [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
                
                # Средний пояс (10 треугольников)
                [1, 2, 6], [2, 3, 7], [3, 4, 8], [4, 5, 9], [5, 1, 10],
                [2, 6, 7], [3, 7, 8], [4, 8, 9], [5, 9, 10], [1, 10, 6],
                
                # Нижняя шапка (5 треугольников)
                [11, 7, 6], [11, 8, 7], [11, 9, 8], [11, 10, 9], [11, 6, 10]
            ]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        elif shape == 'Dodecahedron':
            # Додекаэдр - 20 вершин, 12 граней-пятиугольников
            # Создаем икосаэдр как основу
            ico_vertices = []
            
            # Верхняя вершина
            ico_vertices.append(Point(0, 0, math.sqrt(5)/2 * scale))
            
            # Верхнее кольцо (5 вершин)
            for i in range(5):
                angle = 2 * i * 72 * math.pi / 360
                ico_vertices.append(Point(
                    scale * math.cos(angle),
                    scale * math.sin(angle),
                    scale * 0.5
                ))
            
            # Нижнее кольцо (5 вершин)
            for i in range(5):
                angle = 2 * (36 + i * 72) * math.pi / 360
                ico_vertices.append(Point(
                    scale * math.cos(angle),
                    scale * math.sin(angle),
                    scale * -0.5
                ))
            
            # Нижняя вершина
            ico_vertices.append(Point(0, 0, -math.sqrt(5)/2 * scale))
            
            # Получаем грани икосаэдра
            ico_faces = [
                [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
                [1, 2, 6], [2, 3, 7], [3, 4, 8], [4, 5, 9], [5, 1, 10],
                [2, 6, 7], [3, 7, 8], [4, 8, 9], [5, 9, 10], [1, 10, 6],
                [11, 7, 6], [11, 8, 7], [11, 9, 8], [11, 10, 9], [11, 6, 10]
            ]
            
            # Создаем вершины додекаэдра как центры граней икосаэдра
            vertices = []
            for face in ico_faces:
                # Центр грани
                x = sum(ico_vertices[i].coordinates[0] for i in face) / 3
                y = sum(ico_vertices[i].coordinates[1] for i in face) / 3
                z = sum(ico_vertices[i].coordinates[2] for i in face) / 3
                vertices.append(Point(x, y, z))
            
            # Грани додекаэдра (12 пятиугольников)
            faces = [
                [0, 1, 2, 3, 4],        # Верхняя шапка
                [5, 6, 7, 8, 9],        # Верхнее кольцо
                [10, 11, 12, 13, 14],   # Средний пояс
                [15, 16, 17, 18, 19],   # Нижнее кольцо
                
                # Боковые грани
                [0, 5, 10, 6, 1],
                [1, 6, 11, 7, 2],
                [2, 7, 12, 8, 3],
                [3, 8, 13, 9, 4],
                [4, 9, 14, 5, 0],
                
                [10, 15, 16, 11, 6],
                [11, 16, 17, 12, 7],
                [12, 17, 18, 13, 8],
                [13, 18, 19, 14, 9],
                [14, 19, 15, 10, 5]
            ]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        else:
            polygons = []
            
        self.polyhedron = Polyhedron(polygons)
        # Центрируем многогранник
        self.center_polyhedron()
        self.render()

    def center_polyhedron(self):
        # Находим центр многогранника
        all_vertices = []
        for polygon in self.polyhedron.polygons:
            all_vertices.extend(polygon.vertices)
        
        if not all_vertices:
            return
            
        center = np.mean([v.coordinates[:3] for v in all_vertices], axis=0)
        
        # Перемещаем в центр холста
        translate_matrix = self.get_translation_matrix(300 - center[0], 300 - center[1], 300 - center[2])
        self.polyhedron.transform(translate_matrix)

    def create_projection_buttons(self):
        projection_frame = tk.Frame(self.root)
        projection_frame.pack()
        
        self.projection_var = tk.StringVar(value=self.projection_type)
        
        tk.Radiobutton(projection_frame, text="Перспективная проекция", 
                      variable=self.projection_var, 
                      value='perspective', 
                      command=lambda: self.set_projection('perspective')).pack(side=tk.LEFT)
        
        tk.Radiobutton(projection_frame, text="Аксонометрическая проекция", 
                      variable=self.projection_var, 
                      value='axonometric', 
                      command=lambda: self.set_projection('axonometric')).pack(side=tk.LEFT)

    def set_projection(self, projection_type):
        self.projection_type = projection_type
        self.render()

    def create_rotation_controls(self):
        # ПЕРЕМЕЩЕНА КНОПКА "ПРИМЕНИТЬ ВРАЩЕНИЕ" ВЫШЕ
        arbitrary_rotation_frame = tk.Frame(self.root)
        arbitrary_rotation_frame.pack(pady=5)
        
        tk.Label(arbitrary_rotation_frame, text="Вращение вокруг произвольной прямой:").pack()
        
        # Верхняя строка с кнопкой ПРИМЕНИТЬ ВРАЩЕНИЕ
        top_row = tk.Frame(arbitrary_rotation_frame)
        top_row.pack(pady=5)
        
        tk.Button(top_row, text="Применить вращение", 
                 command=self.apply_arbitrary_rotation).pack(side=tk.LEFT, padx=10)
        
        tk.Label(top_row, text="Угол (градусы):").pack(side=tk.LEFT)
        self.angle_entry = tk.Entry(top_row, width=5)
        self.angle_entry.insert(0, "30")
        self.angle_entry.pack(side=tk.LEFT, padx=5)
        
        # Нижняя строка с точками
        points_frame = tk.Frame(arbitrary_rotation_frame)
        points_frame.pack()
        
        # Точка 1
        tk.Label(points_frame, text="Точка 1:").grid(row=0, column=0)
        self.p1_x = tk.Entry(points_frame, width=5)
        self.p1_x.insert(0, "200")
        self.p1_x.grid(row=0, column=1)
        self.p1_y = tk.Entry(points_frame, width=5)
        self.p1_y.insert(0, "200")
        self.p1_y.grid(row=0, column=2)
        self.p1_z = tk.Entry(points_frame, width=5)
        self.p1_z.insert(0, "200")
        self.p1_z.grid(row=0, column=3)
        
        # Точка 2
        tk.Label(points_frame, text="Точка 2:").grid(row=1, column=0)
        self.p2_x = tk.Entry(points_frame, width=5)
        self.p2_x.insert(0, "400")
        self.p2_x.grid(row=1, column=1)
        self.p2_y = tk.Entry(points_frame, width=5)
        self.p2_y.insert(0, "400")
        self.p2_y.grid(row=1, column=2)
        self.p2_z = tk.Entry(points_frame, width=5)
        self.p2_z.insert(0, "400")
        self.p2_z.grid(row=1, column=3)
        
        # Вращение вокруг координатных осей через центр многогранника
        axis_rotation_frame = tk.Frame(self.root)
        axis_rotation_frame.pack(pady=5)
        
        tk.Label(axis_rotation_frame, text="Вращение вокруг осей через центр:").pack(side=tk.LEFT)
        
        tk.Button(axis_rotation_frame, text="Вращать вокруг X", 
                command=lambda: self.apply_axis_rotation_through_center('x')).pack(side=tk.LEFT, padx=2)
        tk.Button(axis_rotation_frame, text="Вращать вокруг Y", 
                command=lambda: self.apply_axis_rotation_through_center('y')).pack(side=tk.LEFT, padx=2)
        tk.Button(axis_rotation_frame, text="Вращать вокруг Z", 
                command=lambda: self.apply_axis_rotation_through_center('z')).pack(side=tk.LEFT, padx=2)

    def create_buttons(self):
        transform_frame = tk.Frame(self.root)
        transform_frame.pack()

        # Смещение
        offset_frame = tk.Frame(transform_frame)
        offset_frame.grid(row=0, column=0, columnspan=6, pady=5)
        
        tk.Label(offset_frame, text="Смещение:").pack(side=tk.LEFT)
        tk.Label(offset_frame, text="dx:").pack(side=tk.LEFT)
        self.dx_entry = tk.Entry(offset_frame, width=5)
        self.dx_entry.insert(0, "0")
        self.dx_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(offset_frame, text="dy:").pack(side=tk.LEFT)
        self.dy_entry = tk.Entry(offset_frame, width=5)
        self.dy_entry.insert(0, "0")
        self.dy_entry.pack(side=tk.LEFT, padx=2)

        tk.Label(offset_frame, text="dz:").pack(side=tk.LEFT)
        self.dz_entry = tk.Entry(offset_frame, width=5)
        self.dz_entry.insert(0, "0")
        self.dz_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Button(offset_frame, text="Применить смещение", command=self.apply_offset).pack(side=tk.LEFT, padx=5)

        # Отражение
        reflection_frame = tk.Frame(transform_frame)
        reflection_frame.grid(row=1, column=0, columnspan=6, pady=5)
        
        tk.Label(reflection_frame, text="Отражение:").pack(side=tk.LEFT)
        tk.Button(reflection_frame, text="Отражение XY", command=self.apply_reflection_xy).pack(side=tk.LEFT, padx=2)
        tk.Button(reflection_frame, text="Отражение XZ", command=self.apply_reflection_xz).pack(side=tk.LEFT, padx=2)
        tk.Button(reflection_frame, text="Отражение YZ", command=self.apply_reflection_yz).pack(side=tk.LEFT, padx=2)

        # Масштабирование
        scale_frame = tk.Frame(transform_frame)
        scale_frame.grid(row=2, column=0, columnspan=6, pady=5)
        
        tk.Label(scale_frame, text="Масштабирование:").pack(side=tk.LEFT)
        tk.Label(scale_frame, text="Масштаб:").pack(side=tk.LEFT)
        self.scale_factor_entry = tk.Entry(scale_frame, width=5)
        self.scale_factor_entry.insert(0, "1.5")
        self.scale_factor_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Button(scale_frame, text="От центра фигуры", 
                 command=self.apply_scaling_own_center).pack(side=tk.LEFT, padx=2)
        
        tk.Button(scale_frame, text="От точки", 
                 command=self.apply_scaling).pack(side=tk.LEFT, padx=2)

        # Вращение вокруг точки
        rotation_frame = tk.Frame(transform_frame)
        rotation_frame.grid(row=3, column=0, columnspan=6, pady=5)
        
        tk.Label(rotation_frame, text="Вращение:").pack(side=tk.LEFT)
        tk.Label(rotation_frame, text="Центр X:").pack(side=tk.LEFT)
        self.center_x_entry = tk.Entry(rotation_frame, width=5)
        self.center_x_entry.insert(0, "300")
        self.center_x_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(rotation_frame, text="Центр Y:").pack(side=tk.LEFT)
        self.center_y_entry = tk.Entry(rotation_frame, width=5)
        self.center_y_entry.insert(0, "300")
        self.center_y_entry.pack(side=tk.LEFT, padx=2)

        tk.Label(rotation_frame, text="Центр Z:").pack(side=tk.LEFT)
        self.center_z_entry = tk.Entry(rotation_frame, width=5)
        self.center_z_entry.insert(0, "300")
        self.center_z_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(rotation_frame, text="Угол:").pack(side=tk.LEFT)
        self.degrees_entry = tk.Entry(rotation_frame, width=5)
        self.degrees_entry.insert(0, "30")
        self.degrees_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Button(rotation_frame, text="Вокруг точки", command=self.apply_rotation).pack(side=tk.LEFT, padx=5)
        
        tk.Button(rotation_frame, text="Вокруг центра фигуры", command=self.apply_rotation_own_center).pack(side=tk.LEFT, padx=5)

    def apply_offset(self):
        """Смещение фигуры на указанные расстояния по осям X, Y, Z"""
        if not self.polyhedron:
            return
            
        dx = int(self.dx_entry.get() or 0)
        dy = int(self.dy_entry.get() or 0)
        dz = int(self.dz_entry.get() or 0)
        
        self.apply_transform(self.get_translation_matrix(dx, dy, dz))

    def apply_rotation(self):
        """Вращение фигуры вокруг указанной точки на заданный угол"""
        if not self.polyhedron:
            return
            
        center_x = int(self.center_x_entry.get() or 0)
        center_y = int(self.center_y_entry.get() or 0)
        center_z = int(self.center_z_entry.get() or 0)
        degrees = int(self.degrees_entry.get() or 0)
        
        self.apply_transform(self.get_rotation_matrix(center_x, center_y, center_z, degrees))

    def apply_rotation_own_center(self):
        """Вращение фигуры вокруг ее собственного центра на заданный угол"""
        if not self.polyhedron:
            return
            
        degrees = int(self.degrees_entry.get() or 0)
        
        # Находим центр многогранника
        all_vertices = []
        for polygon in self.polyhedron.polygons:
            all_vertices.extend(polygon.vertices)

        center = np.mean([v.coordinates[:3] for v in all_vertices], axis=0)
        
        center_x = center[0]
        center_y = center[1]
        center_z = center[2]
            
        self.apply_transform(self.get_rotation_matrix(center_x, center_y, center_z, degrees))

    def apply_scaling(self):
        """Масштабирование фигуры относительно указанной точки"""
        if not self.polyhedron:
            return
            
        scale_factor = float(self.scale_factor_entry.get() or 1.0)
        center_x = int(self.center_x_entry.get() or 0)
        center_y = int(self.center_y_entry.get() or 0)
        center_z = int(self.center_z_entry.get() or 0)
        
        self.apply_transform(self.get_scaling_matrix(center_x, center_y, center_z, scale_factor))

    def apply_reflection_xy(self):
        matrix = self.get_reflection_matrix('xy')
        self.apply_transform(matrix)

    def apply_reflection_xz(self):
        matrix = self.get_reflection_matrix('xz')
        self.apply_transform(matrix)

    def apply_reflection_yz(self):
        matrix = self.get_reflection_matrix('yz')
        self.apply_transform(matrix)

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

    def apply_scaling_own_center(self):
        """Масштабирование фигуры относительно ее собственного центра"""
        if not self.polyhedron:
            return
            
        scale_factor = float(self.scale_factor_entry.get() or 1.0)

        # Находим центр многогранника
        all_vertices = []
        for polygon in self.polyhedron.polygons:
            all_vertices.extend(polygon.vertices)

        center = np.mean([v.coordinates[:3] for v in all_vertices], axis=0)
        
        center_x = center[0]
        center_y = center[1]
        center_z = center[2]

        self.apply_transform(self.get_scaling_matrix(center_x, center_y, center_z, scale_factor))

    def apply_axis_rotation_through_center(self, axis):
        """Вращение фигуры вокруг оси, проходящей через ее центр"""
        # Находим центр многогранника
        all_vertices = []
        for polygon in self.polyhedron.polygons:
            all_vertices.extend(polygon.vertices)

        center = np.mean([v.coordinates[:3] for v in all_vertices], axis=0)
        
        # Угол вращения (в радианах)
        angle = math.radians(30)  # 30 градусов
        
        # Создаем прямую через центр многогранника, параллельную выбранной оси
        if axis == 'x':
            # Прямая параллельная оси X через центр
            p1 = np.array([center[0] - 100, center[1], center[2]])
            p2 = np.array([center[0] + 100, center[1], center[2]])
        elif axis == 'y':
            # Прямая параллельная оси Y через центр
            p1 = np.array([center[0], center[1] - 100, center[2]])
            p2 = np.array([center[0], center[1] + 100, center[2]])
        elif axis == 'z':
            # Прямая параллельная оси Z через центр
            p1 = np.array([center[0], center[1], center[2] - 100])
            p2 = np.array([center[0], center[1], center[2] + 100])
        
        # Создаем матрицу вращения вокруг этой прямой
        matrix = self.get_rotation_around_line(p1, p2, angle)
        self.apply_transform(matrix)

    def apply_arbitrary_rotation(self):
        """Вращение фигуры вокруг произвольной прямой, заданной двумя точками"""
        try:
            # Получаем координаты точек и угол из полей ввода
            p1 = np.array([float(self.p1_x.get()), float(self.p1_y.get()), float(self.p1_z.get())])
            p2 = np.array([float(self.p2_x.get()), float(self.p2_y.get()), float(self.p2_z.get())])
            angle = math.radians(float(self.angle_entry.get()))
            
            # Создаем матрицу вращения вокруг произвольной прямой
            matrix = self.get_rotation_around_line(p1, p2, angle)
            self.apply_transform(matrix)
        except ValueError:
            print("Ошибка: проверьте правильность введенных данных")

    def get_rotation_around_line(self, p1, p2, angle):
        # Вращение вокруг произвольной прямой, заданной двумя точками p1 и p2
        
        # Вектор направления прямой
        v = p2 - p1
        v_length = np.linalg.norm(v)
        if v_length == 0:
            return np.eye(4)  # Если точки совпадают, возвращаем единичную матрицу
        v = v / v_length  # Нормализуем
        
        # Перенос в начало координат (к точке p1)
        T = self.get_translation_matrix(-p1[0], -p1[1], -p1[2])
        
        # Вращение, чтобы ось совпала с осью Z
        # Вычисляем углы для совмещения оси с Z
        if abs(v[2]) < 1e-10:  # Если ось перпендикулярна Z
            if abs(v[0]) < 1e-10 and abs(v[1]) < 1e-10:
                # Ось уже совпадает с Z
                R_align = np.eye(4)
            else:
                # Поворачиваем вокруг оси Y, затем вокруг оси Z
                alpha = math.atan2(v[1], v[0])
                R_alpha = np.array([
                    [math.cos(alpha), -math.sin(alpha), 0, 0],
                    [math.sin(alpha), math.cos(alpha), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                
                # После поворота на alpha, вектор v становится (sqrt(v[0]^2+v[1]^2), 0, v[2])
                v_temp = np.dot(R_alpha[:3, :3], v)
                beta = math.atan2(v_temp[0], v_temp[2])
                R_beta = np.array([
                    [math.cos(beta), 0, math.sin(beta), 0],
                    [0, 1, 0, 0],
                    [-math.sin(beta), 0, math.cos(beta), 0],
                    [0, 0, 0, 1]
                ])
                
                R_align = np.dot(R_beta, R_alpha)
        else:
            # Ось не перпендикулярна Z
            alpha = math.atan2(v[1], v[0])
            R_alpha = np.array([
                [math.cos(alpha), -math.sin(alpha), 0, 0],
                [math.sin(alpha), math.cos(alpha), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            v_temp = np.dot(R_alpha[:3, :3], v)
            beta = math.atan2(v_temp[0], v_temp[2])
            R_beta = np.array([
                [math.cos(beta), 0, math.sin(beta), 0],
                [0, 1, 0, 0],
                [-math.sin(beta), 0, math.cos(beta), 0],
                [0, 0, 0, 1]
            ])
            
            R_align = np.dot(R_beta, R_alpha)
        
        # Вращение вокруг оси Z на заданный угол
        R_z = np.array([
            [math.cos(angle), -math.sin(angle), 0, 0],
            [math.sin(angle), math.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Обратные преобразования
        R_align_inv = np.linalg.inv(R_align)
        T_inv = self.get_translation_matrix(p1[0], p1[1], p1[2])
        
        # Комбинируем все преобразования: T_inv * R_align_inv * R_z * R_align * T
        temp = np.dot(R_align, T)
        temp = np.dot(R_z, temp)
        temp = np.dot(R_align_inv, temp)
        result = np.dot(T_inv, temp)
        
        return result

    def apply_transform(self, matrix):
        self.polyhedron.transform(matrix)
        self.render()

    def get_translation_matrix(self, dx, dy, dz):
        return np.array([[1, 0, 0, dx],
                         [0, 1, 0, dy],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])

    def get_rotation_matrix(self, center_x, center_y, center_z, angle):
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        # Вращение вокруг оси Z
        return np.array([[cos_a, -sin_a, 0, center_x - center_x * cos_a + center_y * sin_a],
                         [sin_a, cos_a, 0, center_y - center_x * sin_a - center_y * cos_a],
                         [0, 0, 1, 0],
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
        temp = np.dot(translate_back, scale)
        return np.dot(temp, translate_to_origin)

    def render(self):
        self.canvas.delete("all")
        
        for polygon in self.polyhedron.polygons:
            coords = []
            for p in polygon.vertices:
                x, y, z = p.coordinates[:3]
                
                # Применяем проекцию
                if self.projection_type == 'perspective':
                    # Перспективная проекция
                    d = 500  # расстояние до плоскости проекции
                    if z + d != 0:
                        x_proj = x * d / (z + d)
                        y_proj = y * d / (z + d)
                    else:
                        x_proj, y_proj = x, y
                else:
                    # Аксонометрическая проекция
                    x_proj = x - z * 0.5
                    y_proj = y - z * 0.5
                
                coords.extend([x_proj + 300, y_proj + 300])  # Центрируем на холсте
            
            if len(coords) >= 6:  # Минимум 3 точки для полигона
                self.canvas.create_polygon(coords, outline="black", fill="lightblue", width=2)

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()