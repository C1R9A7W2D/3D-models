import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

class Point:
    def __init__(self, x, y, z):
        self.coordinates = np.array([x, y, z, 1.0])

    def transform(self, matrix):
        transformed_coordinates = np.dot(matrix, self.coordinates)
        return Point(transformed_coordinates[0], transformed_coordinates[1], transformed_coordinates[2])
        
    def __repr__(self):
        return f"Point({self.coordinates[0]}, {self.coordinates[1]}, {self.coordinates[2]})"

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
            
        v0 = self.vertices[0].coordinates[:3]
        v1 = self.vertices[1].coordinates[:3]
        v2 = self.vertices[2].coordinates[:3]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        normal = np.cross(edge1, edge2)
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

class RevolutionFigure:
    @staticmethod
    def create_from_generatrix(generatrix_points, axis='z', n_segments=12):
        """Создает фигуру вращения из образующей"""
        vertices = []
        faces = []
        
        angle_step = 2 * math.pi / n_segments
        
        # Создаем вершины
        for i in range(n_segments):
            angle = i * angle_step
            for point in generatrix_points:
                x, y, z = point.coordinates[:3]
                
                if axis == 'z':
                    # Вращение вокруг оси Z
                    new_x = x * math.cos(angle) - y * math.sin(angle)
                    new_y = x * math.sin(angle) + y * math.cos(angle)
                    new_z = z
                elif axis == 'x':
                    # Вращение вокруг оси X
                    new_x = x
                    new_y = y * math.cos(angle) - z * math.sin(angle)
                    new_z = y * math.sin(angle) + z * math.cos(angle)
                elif axis == 'y':
                    # Вращение вокруг оси Y
                    new_x = x * math.cos(angle) + z * math.sin(angle)
                    new_y = y
                    new_z = -x * math.sin(angle) + z * math.cos(angle)
                
                vertices.append(Point(new_x, new_y, new_z))
        
        # Создаем грани
        n_points = len(generatrix_points)
        for i in range(n_segments):
            for j in range(n_points - 1):
                idx1 = i * n_points + j
                idx2 = i * n_points + (j + 1)
                next_i = (i + 1) % n_segments
                idx3 = next_i * n_points + j
                idx4 = next_i * n_points + (j + 1)
                
                # Создаем два треугольника для каждой ячейки
                faces.append([idx1, idx2, idx4])
                faces.append([idx1, idx4, idx3])
        
        polygons = Polygon.polygons_from_vertices(vertices, faces)
        return Polyhedron(polygons)

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Фигура Вращения")
        self.root.geometry("1200x800")
        
        # Основной фрейм
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - холст
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(left_frame, width=600, height=600, bg="white")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)
        
        # Правая панель - управление
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.polyhedron = None
        self.projection_type = 'perspective'
        self.generatrix_points = []
        
        # Создание интерфейса
        self.create_revolution_controls(right_frame)
        self.create_controls_panel(right_frame)
        
        # График образующей
        self.create_generatrix_plot(left_frame)
        
        self.render()

    def create_revolution_controls(self, parent):
        """Создает элементы управления для фигуры вращения"""
        revolution_frame = tk.LabelFrame(parent, text="Построение фигуры вращения", padx=5, pady=5)
        revolution_frame.pack(fill=tk.X, pady=5)
        
        # Ввод образующей
        generatrix_frame = tk.Frame(revolution_frame)
        generatrix_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(generatrix_frame, text="Образующая (формат: x,y,z; x,y,z; ...):").pack(anchor=tk.W)
        self.generatrix_entry = tk.Entry(generatrix_frame, width=40)
        self.generatrix_entry.insert(0, "0,0,0; 0,50,0; 30,50,0; 30,0,0")
        self.generatrix_entry.pack(fill=tk.X, pady=2)
        
        # Параметры вращения
        params_frame = tk.Frame(revolution_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(params_frame, text="Ось вращения:").grid(row=0, column=0, sticky=tk.W)
        self.axis_var = tk.StringVar(value='z')
        axis_combo = ttk.Combobox(params_frame, textvariable=self.axis_var, width=10)
        axis_combo['values'] = ['x', 'y', 'z']
        axis_combo.grid(row=0, column=1, padx=5)
        
        tk.Label(params_frame, text="Количество разбиений:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.segments_var = tk.StringVar(value='12')
        segments_entry = tk.Entry(params_frame, textvariable=self.segments_var, width=10)
        segments_entry.grid(row=1, column=1, padx=5)
        
        # Кнопки управления
        button_frame = tk.Frame(revolution_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(button_frame, text="Построить фигуру", command=self.build_revolution_figure).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Сохранить фигуру", command=self.save_figure).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Загрузить фигуру", command=self.load_figure).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Очистить", command=self.clear_figure).pack(side=tk.LEFT, padx=2)

    def create_generatrix_plot(self, parent):
        """Создает график для отображения образующей"""
        plot_frame = tk.Frame(parent)
        plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(plot_frame, text="График образующей:").pack(anchor=tk.W)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.update_generatrix_plot()

    def update_generatrix_plot(self):
        """Обновляет график образующей"""
        self.ax.clear()
        
        if self.generatrix_points:
            x = [p.coordinates[0] for p in self.generatrix_points]
            y = [p.coordinates[1] for p in self.generatrix_points]
            self.ax.plot(x, y, 'bo-', linewidth=2, markersize=6)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.grid(True)
            self.ax.set_title('Образующая фигуры вращения')
            self.ax.axis('equal')
        
        self.canvas_plot.draw()

    def create_controls_panel(self, parent):
        """Создает панель управления аффинными преобразованиями"""
        controls_frame = tk.LabelFrame(parent, text="Аффинные преобразования", padx=5, pady=5)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Выбор проекции
        tk.Label(controls_frame, text="Тип проекции:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.projection_var = tk.StringVar(value=self.projection_type)
        tk.Radiobutton(controls_frame, text="Перспективная", 
                      variable=self.projection_var, value='perspective',
                      command=lambda: self.set_projection('perspective')).pack(anchor=tk.W)
        tk.Radiobutton(controls_frame, text="Аксонометрическая", 
                      variable=self.projection_var, value='axonometric',
                      command=lambda: self.set_projection('axonometric')).pack(anchor=tk.W)
        
        # Смещение
        trans_frame = tk.Frame(controls_frame)
        trans_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(trans_frame, text="Смещение:").grid(row=0, column=0, sticky=tk.W)
        self.dx_entry = tk.Entry(trans_frame, width=5)
        self.dx_entry.insert(0, "10")
        self.dx_entry.grid(row=0, column=1, padx=2)
        
        self.dy_entry = tk.Entry(trans_frame, width=5)
        self.dy_entry.insert(0, "10")
        self.dy_entry.grid(row=0, column=2, padx=2)
        
        self.dz_entry = tk.Entry(trans_frame, width=5)
        self.dz_entry.insert(0, "10")
        self.dz_entry.grid(row=0, column=3, padx=2)
        
        tk.Button(trans_frame, text="Применить", command=self.apply_translation).grid(row=0, column=4, padx=5)
        
        # Вращение
        rot_frame = tk.Frame(controls_frame)
        rot_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(rot_frame, text="Вращение вокруг центра (град):").grid(row=0, column=0, sticky=tk.W)
        self.rot_x_entry = tk.Entry(rot_frame, width=5)
        self.rot_x_entry.insert(0, "30")
        self.rot_x_entry.grid(row=0, column=1, padx=2)
        
        self.rot_y_entry = tk.Entry(rot_frame, width=5)
        self.rot_y_entry.insert(0, "30")
        self.rot_y_entry.grid(row=0, column=2, padx=2)
        
        self.rot_z_entry = tk.Entry(rot_frame, width=5)
        self.rot_z_entry.insert(0, "30")
        self.rot_z_entry.grid(row=0, column=3, padx=2)
        
        tk.Button(rot_frame, text="Применить", command=self.apply_rotation).grid(row=0, column=4, padx=5)
        
        # Масштабирование
        scale_frame = tk.Frame(controls_frame)
        scale_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(scale_frame, text="Масштаб:").grid(row=0, column=0, sticky=tk.W)
        self.scale_entry = tk.Entry(scale_frame, width=5)
        self.scale_entry.insert(0, "1.2")
        self.scale_entry.grid(row=0, column=1, padx=2)
        
        tk.Button(scale_frame, text="Применить", command=self.apply_scaling).grid(row=0, column=2, padx=5)

    def parse_generatrix(self, generatrix_str):
        """Парсит строку образующей в список точек"""
        points = []
        try:
            point_strs = generatrix_str.split(';')
            for point_str in point_strs:
                coords = point_str.strip().split(',')
                if len(coords) == 3:
                    x, y, z = map(float, coords)
                    points.append(Point(x, y, z))
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Неверный формат образующей: {e}")
        return points

    def build_revolution_figure(self):
        """Строит фигуру вращения"""
        generatrix_str = self.generatrix_entry.get()
        self.generatrix_points = self.parse_generatrix(generatrix_str)
        
        if not self.generatrix_points:
            messagebox.showerror("Ошибка", "Неверно задана образующая")
            return
        
        try:
            n_segments = int(self.segments_var.get())
            axis = self.axis_var.get()
            
            self.polyhedron = RevolutionFigure.create_from_generatrix(
                self.generatrix_points, axis, n_segments
            )
            self.center_polyhedron()
            self.update_generatrix_plot()
            self.render()
            
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Ошибка построения: {e}")

    def clear_figure(self):
        """Очищает фигуру"""
        self.polyhedron = None
        self.generatrix_points = []
        self.update_generatrix_plot()
        self.render()

    def save_figure(self):
        """Сохраняет фигуру в файл"""
        if not self.polyhedron:
            messagebox.showwarning("Предупреждение", "Нет фигуры для сохранения")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                data = {
                    'generatrix': [
                        [p.coordinates[0], p.coordinates[1], p.coordinates[2]] 
                        for p in self.generatrix_points
                    ],
                    'axis': self.axis_var.get(),
                    'segments': int(self.segments_var.get())
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                # Сохраняем график
                plot_filename = filename.replace('.json', '_plot.png')
                self.fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                
                messagebox.showinfo("Успех", f"Фигура сохранена в {filename}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения: {e}")

    def load_figure(self):
        """Загружает фигуру из файла"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Восстанавливаем образующую
                generatrix_points = []
                for coords in data['generatrix']:
                    generatrix_points.append(Point(coords[0], coords[1], coords[2]))
                
                self.generatrix_points = generatrix_points
                self.generatrix_entry.delete(0, tk.END)
                self.generatrix_entry.insert(0, '; '.join(
                    [f"{p.coordinates[0]},{p.coordinates[1]},{p.coordinates[2]}" 
                     for p in generatrix_points]
                ))
                
                self.axis_var.set(data['axis'])
                self.segments_var.set(str(data['segments']))
                
                # Перестраиваем фигуру
                self.polyhedron = RevolutionFigure.create_from_generatrix(
                    generatrix_points, data['axis'], data['segments']
                )
                self.center_polyhedron()
                self.update_generatrix_plot()
                self.render()
                
                messagebox.showinfo("Успех", f"Фигура загружена из {filename}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки: {e}")

    def center_polyhedron(self):
        """Центрирует полиэдр"""
        if not self.polyhedron:
            return
            
        center = self.polyhedron.get_center()
        translate_matrix = self.get_translation_matrix(300 - center[0], 300 - center[1], 300 - center[2])
        self.polyhedron.transform(translate_matrix)

    # МАТРИЧНЫЕ ПРЕОБРАЗОВАНИЯ (аналогично вашей лабораторной работе)
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

    def apply_translation(self):
        if not self.polyhedron:
            return
            
        dx = float(self.dx_entry.get() or 0)
        dy = float(self.dy_entry.get() or 0)
        dz = float(self.dz_entry.get() or 0)
        
        matrix = self.get_translation_matrix(dx, dy, dz)
        self.polyhedron.transform(matrix)
        self.render()

    def apply_rotation(self):
        if not self.polyhedron:
            return
            
        center = self.polyhedron.get_center()
        
        angle_x = math.radians(float(self.rot_x_entry.get() or 0))
        angle_y = math.radians(float(self.rot_y_entry.get() or 0))
        angle_z = math.radians(float(self.rot_z_entry.get() or 0))
        
        T = self.get_translation_matrix(-center[0], -center[1], -center[2])
        R_x = self.get_rotation_matrix_x(angle_x)
        R_y = self.get_rotation_matrix_y(angle_y)
        R_z = self.get_rotation_matrix_z(angle_z)
        T_inv = self.get_translation_matrix(center[0], center[1], center[2])
        
        # Комбинируем вращения
        R = np.dot(R_z, np.dot(R_y, R_x))
        matrix = np.dot(T_inv, np.dot(R, T))
        
        self.polyhedron.transform(matrix)
        self.render()

    def apply_scaling(self):
        if not self.polyhedron:
            return
            
        scale_factor = float(self.scale_entry.get() or 1.0)
        center = self.polyhedron.get_center()
        
        T = self.get_translation_matrix(-center[0], -center[1], -center[2])
        S = self.get_scaling_matrix(scale_factor, scale_factor, scale_factor)
        T_inv = self.get_translation_matrix(center[0], center[1], center[2])
        
        matrix = np.dot(T_inv, np.dot(S, T))
        self.polyhedron.transform(matrix)
        self.render()

    def set_projection(self, projection_type):
        self.projection_type = projection_type
        self.render()

    def is_polygon_visible(self, polygon):
        if len(polygon.vertices) < 3:
            return False
            
        normal = polygon.get_normal()
        center = np.mean([v.coordinates[:3] for v in polygon.vertices], axis=0)
        view_vector = center - np.array([0, 0, -1000])
        return np.dot(normal, view_vector) > 0

    def render(self):
        self.canvas.delete("all")
        
        if not self.polyhedron:
            return
            
        sorted_polygons = sorted(self.polyhedron.polygons, 
                               key=lambda p: p.get_center_z(), 
                               reverse=True)
        
        for polygon in sorted_polygons:
            if not self.is_polygon_visible(polygon):
                continue
                
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
                    width=2,
                    smooth=False
                )

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
