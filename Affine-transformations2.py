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
        if shape == 'Tetrahedron':
            vertices = [
                Point(0, 0, 0), 
                Point(1, 0, 0), 
                Point(0.5, math.sqrt(3)/2, 0), 
                Point(0.5, math.sqrt(3)/6, math.sqrt(2/3))
            ]
            faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        elif shape == 'Cube':
            vertices = [
                Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0),
                Point(0, 0, 1), Point(1, 0, 1), Point(1, 1, 1), Point(0, 1, 1)
            ]
            faces = [
                [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
            ]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        elif shape == 'Octahedron':
            vertices = [
                Point(0.5, 0.5, 0), Point(0.5, 0.5, 1), 
                Point(0, 0.5, 0.5), Point(1, 0.5, 0.5),
                Point(0.5, 0, 0.5), Point(0.5, 1, 0.5)
            ]
            faces = [
                [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
                [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5]
            ]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        elif shape == 'Icosahedron':
            # Создание икосаэдра
            phi = (1 + math.sqrt(5)) / 2  # золотое сечение
            vertices = [
                Point(0, 1, phi), Point(0, 1, -phi), Point(0, -1, phi), Point(0, -1, -phi),
                Point(1, phi, 0), Point(1, -phi, 0), Point(-1, phi, 0), Point(-1, -phi, 0),
                Point(phi, 0, 1), Point(phi, 0, -1), Point(-phi, 0, 1), Point(-phi, 0, -1)
            ]
            faces = [
                [0, 2, 8], [0, 2, 10], [0, 4, 6], [0, 4, 8], [0, 6, 10],
                [1, 3, 9], [1, 3, 11], [1, 4, 6], [1, 4, 9], [1, 6, 11],
                [2, 5, 8], [2, 5, 10], [3, 5, 9], [3, 5, 11],
                [4, 8, 9], [6, 10, 11],
                [7, 8, 10], [7, 8, 9], [7, 10, 11], [7, 9, 11]
            ]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        elif shape == 'Dodecahedron':
            # Создание додекаэдра
            phi = (1 + math.sqrt(5)) / 2  # золотое сечение
            vertices = []
            # Вершины додекаэдра
            for i in [-1, 1]:
                for j in [-1, 1]:
                    vertices.append(Point(0, i/phi, j*phi))
                    vertices.append(Point(i/phi, j*phi, 0))
                    vertices.append(Point(i*phi, 0, j/phi))
            
            faces = [
                [0, 2, 4, 6, 8], [1, 3, 5, 7, 9],
                [0, 2, 10, 12, 14], [1, 3, 11, 13, 15],
                [4, 6, 16, 18, 20], [5, 7, 17, 19, 21],
                [8, 10, 16, 18, 22], [9, 11, 17, 19, 23],
                [12, 14, 20, 22, 24], [13, 15, 21, 23, 25],
                [0, 8, 10, 12, 14], [1, 9, 11, 13, 15]
            ]
            polygons = Polygon.polygons_from_vertices(vertices, faces)
            
        else:
            polygons = []
            
        self.polyhedron = Polyhedron(polygons)
        # Масштабируем и центрируем многогранник
        self.center_and_scale()
        self.render()

    def center_and_scale(self):
        # Находим центр многогранника
        all_vertices = []
        for polygon in self.polyhedron.polygons:
            all_vertices.extend(polygon.vertices)
        
        if not all_vertices:
            return
            
        center = np.mean([v.coordinates[:3] for v in all_vertices], axis=0)
        
        # Масштабируем и перемещаем в центр холста
        scale = 200
        translate_matrix = self.get_translation_matrix(300 - center[0]*scale, 300 - center[1]*scale, 300 - center[2]*scale)
        scale_matrix = self.get_scaling_matrix(0, 0, 0, scale)
        
        self.polyhedron.transform(scale_matrix)
        self.polyhedron.transform(translate_matrix)

    def create_projection_buttons(self):
        projection_frame = tk.Frame(self.root)
        projection_frame.pack()
        
        tk.Radiobutton(projection_frame, text="Перспективная проекция", 
                      variable=tk.StringVar(value=self.projection_type), 
                      value='perspective', 
                      command=lambda: self.set_projection('perspective')).pack(side=tk.LEFT)
        
        tk.Radiobutton(projection_frame, text="Аксонометрическая проекция", 
                      variable=tk.StringVar(value=self.projection_type), 
                      value='axonometric', 
                      command=lambda: self.set_projection('axonometric')).pack(side=tk.LEFT)

    def set_projection(self, projection_type):
        self.projection_type = projection_type
        self.render()

    def create_buttons(self):
        transform_frame = tk.Frame(self.root)
        transform_frame.pack()

        # Отражение
        reflection_frame = tk.Frame(transform_frame)
        reflection_frame.grid(row=0, column=0, columnspan=6, pady=5)
        
        tk.Button(reflection_frame, text="Отражение XY", command=self.apply_reflection_xy).pack(side=tk.LEFT, padx=2)
        tk.Button(reflection_frame, text="Отражение XZ", command=self.apply_reflection_xz).pack(side=tk.LEFT, padx=2)
        tk.Button(reflection_frame, text="Отражение YZ", command=self.apply_reflection_yz).pack(side=tk.LEFT, padx=2)

        # Масштабирование
        scale_frame = tk.Frame(transform_frame)
        scale_frame.grid(row=1, column=0, columnspan=6, pady=5)
        
        tk.Label(scale_frame, text="Масштаб:").pack(side=tk.LEFT)
        self.scale_factor_entry = tk.Entry(scale_frame, width=5)
        self.scale_factor_entry.insert(0, "1.5")
        self.scale_factor_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Button(scale_frame, text="Масштабировать от центра", 
                 command=self.apply_scaling_own_center).pack(side=tk.LEFT, padx=2)

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

    def apply_transform(self, matrix):
        self.polyhedron.transform(matrix)
        self.render()

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