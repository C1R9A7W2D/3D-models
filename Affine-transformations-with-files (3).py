import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
                
                # Записываем грани
                for polygon in self.polygons:
                    face_indices = []
                    for vertex in polygon.vertices:
                        vertex_tuple = (vertex.coordinates[0], vertex.coordinates[1], vertex.coordinates[2])
                        face_indices.append(str(vertex_to_index[vertex_tuple]))
                    
                    if len(face_indices) >= 3:
                        f.write(f"f {' '.join(face_indices)}\n")
                
            return True
        except Exception as e:
            print(f"Ошибка сохранения OBJ: {e}")
            return False

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
                        vertices.append(Point(x, y, z))
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
            return Polyhedron(polygons)
            
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

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Polyhedra - Аффинные преобразования и поверхности")
        self.root.geometry("800x700")
        
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.polyhedron = None
        self.projection_type = 'perspective'
        
        # Создание интерфейса
        self.create_controls_panel()
        self.render()

    def create_controls_panel(self):
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Панель построения поверхности
        surface_frame = tk.LabelFrame(controls_frame, text="Построение поверхности", padx=5, pady=5)
        surface_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Функция
        tk.Label(surface_frame, text="f(x, y) =").pack(anchor=tk.W)
        self.func_entry = tk.Entry(surface_frame, width=20)
        self.func_entry.insert(0, "(x**2 + y**2) / 10")
        self.func_entry.pack(fill=tk.X, pady=2)
        
        # Диапазоны
        range_frame = tk.Frame(surface_frame)
        range_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(range_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x0_entry = tk.Entry(range_frame, width=5)
        self.x0_entry.insert(0, "-6")
        self.x0_entry.grid(row=0, column=1, padx=2)
        
        tk.Label(range_frame, text="до").grid(row=0, column=2)
        self.x1_entry = tk.Entry(range_frame, width=5)
        self.x1_entry.insert(0, "6")
        self.x1_entry.grid(row=0, column=3, padx=2)
        
        tk.Label(range_frame, text="Y:").grid(row=1, column=0, sticky=tk.W)
        self.y0_entry = tk.Entry(range_frame, width=5)
        self.y0_entry.insert(0, "-6")
        self.y0_entry.grid(row=1, column=1, padx=2)
        
        tk.Label(range_frame, text="до").grid(row=1, column=2)
        self.y1_entry = tk.Entry(range_frame, width=5)
        self.y1_entry.insert(0, "6")
        self.y1_entry.grid(row=1, column=3, padx=2)
        
        # Количество разбиений
        subdiv_frame = tk.Frame(surface_frame)
        subdiv_frame.pack(fill=tk.X, pady=2)
        tk.Label(subdiv_frame, text="Разбиений:").pack(side=tk.LEFT)
        self.subdivisions_entry = tk.Entry(subdiv_frame, width=5)
        self.subdivisions_entry.insert(0, "20")
        self.subdivisions_entry.pack(side=tk.LEFT, padx=5)
        
        # Кнопки построения поверхности
        button_frame = tk.Frame(surface_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(button_frame, text="Построить поверхность", 
                 command=self.generate_surface).pack(side=tk.LEFT, padx=2)
        
        # Кнопки загрузки/сохранения OBJ
        file_frame = tk.Frame(controls_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(file_frame, text="Загрузить OBJ", 
                 command=self.load_obj_file).pack(side=tk.LEFT, padx=2)
        tk.Button(file_frame, text="Сохранить OBJ", 
                 command=self.save_obj_file).pack(side=tk.LEFT, padx=2)
        
        # Выбор проекции
        tk.Label(controls_frame, text="Тип проекции:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.projection_var = tk.StringVar(value=self.projection_type)
        tk.Radiobutton(controls_frame, text="Перспективная", 
                      variable=self.projection_var, value='perspective',
                      command=lambda: self.set_projection('perspective')).pack(anchor=tk.W)
        tk.Radiobutton(controls_frame, text="Аксонометрическая", 
                      variable=self.projection_var, value='axonometric',
                      command=lambda: self.set_projection('axonometric')).pack(anchor=tk.W)
        
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
            self.render()
            messagebox.showinfo("Успех", f"Поверхность построена: {len(self.polyhedron.polygons)} полигонов")
            
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
                                    if vertex_index >= 0 and vertex_index < len(vertices):
                                        face_vertices.append(vertex_index)
                            
                            if len(face_vertices) >= 3:
                                # Если грань имеет больше 3 вершин, разбиваем на треугольники
                                for i in range(1, len(face_vertices) - 1):
                                    faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])
                
                if vertices and faces:
                    polygons = Polygon.polygons_from_vertices(vertices, faces)
                    self.polyhedron = Polyhedron(polygons)
                    self.center_polyhedron()
                    self.render()
                    messagebox.showinfo("Успех", f"Загружен OBJ файл: {len(vertices)} вершин, {len(faces)} граней")
                else:
                    messagebox.showerror("Ошибка", "Файл не содержит вершин или граней")
                    
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

    def is_polygon_visible(self, polygon):
        """Упрощенная проверка видимости - всегда возвращает True для тестирования"""
        return True

    def render(self):
        self.canvas.delete("all")
        
        if not self.polyhedron:
            return
            
        sorted_polygons = sorted(self.polyhedron.polygons, 
                               key=lambda p: p.get_center_z(), 
                               reverse=True)
        
        for polygon in sorted_polygons:
            # Временно отключаем проверку видимости для отладки
            # if not self.is_polygon_visible(polygon):
            #     continue
                
            projected_points = []
            for p in polygon.vertices:
                x, y, z = p.coordinates[:3]
                
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
                
                projected_points.append((x_proj + 300, y_proj + 300))
            
            if len(projected_points) >= 3:
                coords = []
                for point in projected_points:
                    coords.extend(point)
                
                self.canvas.create_polygon(
                    coords, 
                    outline="black", 
                    fill="lightblue", 
                    width=1,
                    smooth=False
                )

    def set_projection(self, projection_type):
        self.projection_type = projection_type
        self.render()

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()