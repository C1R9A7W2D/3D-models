import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import math

class Point:
    def __init__(self, x, y, z):
        self.coordinates = np.array([x, y, z, 1.0])

    def transform(self, matrix):
        transformed_coordinates = np.dot(matrix, self.coordinates)
        return Point(transformed_coordinates[0], transformed_coordinates[1], transformed_coordinates[2])

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def transform(self, matrix):
        self.vertices = [vertex.transform(matrix) for vertex in self.vertices]
        
    def get_center_z(self):
        return np.mean([v.coordinates[2] for v in self.vertices])
        
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
        self.root.title("3D Polyhedra - Аффинные преобразования с Z-буфером")
        self.root.geometry("800x700")
        
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.polyhedron = None
        self.projection_type = 'perspective'
        self.use_z_buffer = True  # Флаг использования Z-буфера
        
        # Инициализация Z-буфера
        self.z_buffer = ZBuffer(600, 600)
        
        # Создание интерфейса
        self.create_controls_panel()
        self.render()

        self.colors = np.array([[135, 206, 250],[135, 20, 25],[10, 206, 25],[135, 0, 250],[10, 20, 250],[10, 20, 25],[200, 200, 200],[135, 206, 0]])

    def create_controls_panel(self):
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Кнопка загрузки OBJ файла
        tk.Button(controls_frame, text="Загрузить OBJ файл", command=self.load_obj_file).pack(fill=tk.X, pady=(0, 10))
        
        # Выбор проекции
        tk.Label(controls_frame, text="Тип проекции:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.projection_var = tk.StringVar(value=self.projection_type)
        tk.Radiobutton(controls_frame, text="Перспективная", 
                      variable=self.projection_var, value='perspective',
                      command=lambda: self.set_projection('perspective')).pack(anchor=tk.W)
        tk.Radiobutton(controls_frame, text="Аксонометрическая", 
                      variable=self.projection_var, value='axonometric',
                      command=lambda: self.set_projection('axonometric')).pack(anchor=tk.W)
        
        # Включение/выключение Z-буфера
        tk.Label(controls_frame, text="Алгоритм удаления невидимых граней:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.z_buffer_var = tk.BooleanVar(value=self.use_z_buffer)
        tk.Checkbutton(controls_frame, text="Использовать Z-буфер", 
                      variable=self.z_buffer_var,
                      command=self.toggle_z_buffer).pack(anchor=tk.W)
        
        # Разделитель
        ttk.Separator(controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Аффинные преобразования
        tk.Label(controls_frame, text="Аффинные преобразования:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.create_translation_controls(controls_frame)
        self.create_rotation_controls(controls_frame)
        self.create_scaling_controls(controls_frame)
        self.create_reflection_controls(controls_frame)
        self.create_arbitrary_rotation_controls(controls_frame)

    def create_translation_controls(self, parent):
        frame = tk.LabelFrame(parent, text="Смещение", padx=5, pady=5)
        frame.pack(fill=tk.X, pady=5)
        
        input_frame = tk.Frame(frame)
        input_frame.pack(fill=tk.X)
        
        tk.Label(input_frame, text="dx:").grid(row=0, column=0, padx=2)
        self.dx_entry = tk.Entry(input_frame, width=5)
        self.dx_entry.insert(0, "10")
        self.dx_entry.grid(row=0, column=1, padx=2)
        
        tk.Label(input_frame, text="dy:").grid(row=0, column=2, padx=2)
        self.dy_entry = tk.Entry(input_frame, width=5)
        self.dy_entry.insert(0, "10")
        self.dy_entry.grid(row=0, column=3, padx=2)
        
        tk.Label(input_frame, text="dz:").grid(row=0, column=4, padx=2)
        self.dz_entry = tk.Entry(input_frame, width=5)
        self.dz_entry.insert(0, "10")
        self.dz_entry.grid(row=0, column=5, padx=2)
        
        tk.Button(frame, text="Применить смещение", command=self.apply_translation).pack(pady=5)

    def create_rotation_controls(self, parent):
        frame = tk.LabelFrame(parent, text="Вращение", padx=5, pady=5)
        frame.pack(fill=tk.X, pady=5)
        
        # Вращение вокруг точки
        point_frame = tk.Frame(frame)
        point_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(point_frame, text="Вокруг точки:").pack(side=tk.LEFT)
        self.center_x_entry = tk.Entry(point_frame, width=5)
        self.center_x_entry.insert(0, "300")
        self.center_x_entry.pack(side=tk.LEFT, padx=2)
        
        self.center_y_entry = tk.Entry(point_frame, width=5)
        self.center_y_entry.insert(0, "300")
        self.center_y_entry.pack(side=tk.LEFT, padx=2)
        
        self.center_z_entry = tk.Entry(point_frame, width=5)
        self.center_z_entry.insert(0, "300")
        self.center_z_entry.pack(side=tk.LEFT, padx=2)
        
        self.angle_entry = tk.Entry(point_frame, width=5)
        self.angle_entry.insert(0, "30")
        self.angle_entry.pack(side=tk.LEFT, padx=2)
        tk.Label(point_frame, text="°").pack(side=tk.LEFT)
        
        tk.Button(point_frame, text="Применить", command=self.apply_rotation_around_point).pack(side=tk.LEFT, padx=5)
        
        # Вращение вокруг центра
        center_frame = tk.Frame(frame)
        center_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(center_frame, text="Вокруг центра:").pack(side=tk.LEFT)
        tk.Button(center_frame, text="X", command=lambda: self.apply_rotation_around_center('x')).pack(side=tk.LEFT, padx=2)
        tk.Button(center_frame, text="Y", command=lambda: self.apply_rotation_around_center('y')).pack(side=tk.LEFT, padx=2)
        tk.Button(center_frame, text="Z", command=lambda: self.apply_rotation_around_center('z')).pack(side=tk.LEFT, padx=2)

    def create_scaling_controls(self, parent):
        frame = tk.LabelFrame(parent, text="Масштабирование", padx=5, pady=5)
        frame.pack(fill=tk.X, pady=5)
        
        scale_frame = tk.Frame(frame)
        scale_frame.pack(fill=tk.X)
        
        tk.Label(scale_frame, text="Коэффициент:").pack(side=tk.LEFT)
        self.scale_factor_entry = tk.Entry(scale_frame, width=5)
        self.scale_factor_entry.insert(0, "1.5")
        self.scale_factor_entry.pack(side=tk.LEFT, padx=5)
        
        button_frame = tk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(button_frame, text="От центра", command=self.apply_scaling_around_center).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="От точки", command=self.apply_scaling_around_point).pack(side=tk.LEFT, padx=2)

    def create_reflection_controls(self, parent):
        frame = tk.LabelFrame(parent, text="Отражение", padx=5, pady=5)
        frame.pack(fill=tk.X, pady=5)
        
        button_frame = tk.Frame(frame)
        button_frame.pack()
        
        tk.Button(button_frame, text="XY плоскость", command=lambda: self.apply_reflection('xy')).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="XZ плоскость", command=lambda: self.apply_reflection('xz')).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="YZ плоскость", command=lambda: self.apply_reflection('yz')).pack(side=tk.LEFT, padx=2)

    def create_arbitrary_rotation_controls(self, parent):
        frame = tk.LabelFrame(parent, text="Вращение вокруг произвольной прямой", padx=5, pady=5)
        frame.pack(fill=tk.X, pady=5)
        
        p1_frame = tk.Frame(frame)
        p1_frame.pack(fill=tk.X, pady=2)
        tk.Label(p1_frame, text="Точка 1:").pack(side=tk.LEFT)
        self.p1_x = tk.Entry(p1_frame, width=5)
        self.p1_x.insert(0, "200")
        self.p1_x.pack(side=tk.LEFT, padx=2)
        self.p1_y = tk.Entry(p1_frame, width=5)
        self.p1_y.insert(0, "200")
        self.p1_y.pack(side=tk.LEFT, padx=2)
        self.p1_z = tk.Entry(p1_frame, width=5)
        self.p1_z.insert(0, "200")
        self.p1_z.pack(side=tk.LEFT, padx=2)
        
        p2_frame = tk.Frame(frame)
        p2_frame.pack(fill=tk.X, pady=2)
        tk.Label(p2_frame, text="Точка 2:").pack(side=tk.LEFT)
        self.p2_x = tk.Entry(p2_frame, width=5)
        self.p2_x.insert(0, "400")
        self.p2_x.pack(side=tk.LEFT, padx=2)
        self.p2_y = tk.Entry(p2_frame, width=5)
        self.p2_y.insert(0, "400")
        self.p2_y.pack(side=tk.LEFT, padx=2)
        self.p2_z = tk.Entry(p2_frame, width=5)
        self.p2_z.insert(0, "400")
        self.p2_z.pack(side=tk.LEFT, padx=2)
        
        angle_frame = tk.Frame(frame)
        angle_frame.pack(fill=tk.X, pady=2)
        tk.Label(angle_frame, text="Угол:").pack(side=tk.LEFT)
        self.arbitrary_angle_entry = tk.Entry(angle_frame, width=5)
        self.arbitrary_angle_entry.insert(0, "45")
        self.arbitrary_angle_entry.pack(side=tk.LEFT, padx=2)
        tk.Label(angle_frame, text="°").pack(side=tk.LEFT)
        
        tk.Button(frame, text="Применить вращение", command=self.apply_arbitrary_rotation).pack(pady=5)

    def toggle_z_buffer(self):
        self.use_z_buffer = self.z_buffer_var.get()
        self.render()

    def set_projection(self, projection_type):
        self.projection_type = projection_type
        self.render()

    def load_obj_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите OBJ файл",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                vertices = []
                faces = []
                scale = 50  #Здесь задаём масштаб, так как объекты в .obj-файлах слишком маленькие для нашего графика
                
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
                                
                        elif parts[0] == 'f':  # грань
                            face_vertices = []
                            for part in parts[1:]:
                                # OBJ формат может быть v, v/vt, v/vt/vn, v//vn
                                vertex_data = part.split('/')[0]
                                if vertex_data:
                                    # Индексы в OBJ начинаются с 1, поэтому вычитаем 1
                                    vertex_index = int(vertex_data) - 1
                                    face_vertices.append(vertex_index)
                            
                            if len(face_vertices) >= 3:
                                faces.append(face_vertices)
                
                if vertices and faces:
                    polygons = Polygon.polygons_from_vertices(vertices, faces)
                    self.polyhedron = Polyhedron(polygons)
                    self.center_polyhedron()
                    self.render()
                    print(f"Загружен OBJ файл: {len(vertices)} вершин, {len(faces)} граней")
                else:
                    print("Ошибка: файл не содержит вершин или граней")
                    
            except Exception as e:
                print(f"Ошибка загрузки OBJ файла: {e}")

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
            
        # Получаем центр многогранника
        center = self.polyhedron.get_center()
        
        T_to_origin = self.get_translation_matrix(-center[0], -center[1], -center[2])
        R = self.get_reflection_matrix(plane)
        T_back = self.get_translation_matrix(center[0], center[1], center[2])
        
        # Комбинированная матрица: T_back * R * T_to_origin
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

    def is_polygon_visible(self, polygon):
        if len(polygon.vertices) < 3:
            return False
            
        normal = polygon.get_normal()
        
        # Вектор от камеры к центру грани
        center = np.mean([v.coordinates[:3] for v in polygon.vertices], axis=0)
        view_vector = center - np.array([0, 0, -1000])  # Камера находится в (0, 0, -1000)
        
        # Если скалярное произведение положительное, грань видима
        return np.dot(normal, view_vector) > 0

    def rasterize_polygon(self, polygon, color):
        if len(polygon.vertices) < 3:
            return
            
        # Получаем проекции всех вершин
        projected_points = []
        original_points = []
        for p in polygon.vertices:
            x, y, z = p.coordinates[:3]
            
            if self.projection_type == 'perspective':
                d = 500  # расстояние до плоскости проекции
                if z + d != 0:
                    x_proj = x * d / (z + d)
                    y_proj = y * d / (z + d)
                else:
                    x_proj, y_proj = x, y
            else:  # аксонометрическая
                x_proj = x - z * 0.5
                y_proj = y - z * 0.5
            
            screen_x = int(x_proj + 300)
            screen_y = int(y_proj + 300)
            projected_points.append((screen_x, screen_y))
            original_points.append((x, y, z))
        
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
        if (x_start == x_end):
            return (z_start + z_end) / 2
        
        t = (x - x_start) / (x_end - x_start)
        z_intersect = z_start + t * (z_end - z_start)
        return z_intersect

    def render_with_z_buffer(self):
        self.z_buffer.clear()
        
        counter = 0
        for polygon in self.polyhedron.polygons:
            #if not self.is_polygon_visible(polygon):
                #continue
            
            # Используем разные цвета для разных полигонов для наглядности
            color = self.colors[counter % len(self.colors)]
            counter += 1
            self.rasterize_polygon(polygon, color)
        
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
        # Оригинальный метод рендеринга (сортировка по глубине)
        sorted_polygons = sorted(self.polyhedron.polygons, 
                               key=lambda p: p.get_center_z(), 
                               reverse=True)
        
        for polygon in sorted_polygons:
            # Проецируем все вершины полигона
            projected_points = []
            for p in polygon.vertices:
                x, y, z = p.coordinates[:3]
                
                if self.projection_type == 'perspective':
                    d = 500  # расстояние до плоскости проекции
                    if z + d != 0:
                        x_proj = x * d / (z + d)
                        y_proj = y * d / (z + d)
                    else:
                        x_proj, y_proj = x, y
                else:  # аксонометрическая
                    x_proj = x - z * 0.5
                    y_proj = y - z * 0.5
                
                projected_points.append((x_proj + 300, y_proj + 300))
            
            # Рисуем полигон только если есть хотя бы 3 точки
            if len(projected_points) >= 3:
                # Создаем список координат для полигона
                coords = []
                for point in projected_points:
                    coords.extend(point)
                
                # Рисуем заполненный полигон
                self.canvas.create_polygon(
                    coords, 
                    outline="black", 
                    fill="lightblue", 
                    width=2,
                    smooth=False  # Отключаем сглаживание для четких граней
                )

    def render(self):
        self.canvas.delete("all")
        
        if not self.polyhedron:
            return
            
        if self.use_z_buffer:
            self.render_with_z_buffer()
        else:
            self.render_without_z_buffer()

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
