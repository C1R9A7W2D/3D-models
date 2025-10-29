import tkinter as tk
from tkinter import ttk
import numpy as np

class Point:
    def __init__(self, x, y, z):
        self.coordinates = np.array([x, y, z])

    def transform(self, matrix):
        transformed_coordinates = np.dot(matrix, np.append(self.coordinates, 1))
        return Point(transformed_coordinates[0], transformed_coordinates[1], transformed_coordinates[2])

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def transform(self, matrix):
        self.vertices = [vertex.transform(matrix) for vertex in self.vertices]

    def polygons_from_vertices(vertices):
        if len(vertices) == 4:
            polygons = [Polygon([vertices[0], vertices[1], vertices[2]]),
                        Polygon([vertices[0], vertices[1], vertices[3]]),
                        Polygon([vertices[0], vertices[2], vertices[3]]),
                        Polygon([vertices[1], vertices[2], vertices[3]])]
            return polygons
        elif len(vertices) == 8:
            polygons = [Polygon([vertices[0], vertices[1], vertices[2], vertices[3]]),
                        Polygon([vertices[4], vertices[5], vertices[6], vertices[7]]),
                        Polygon([vertices[0], vertices[2], vertices[4], vertices[6]]),
                        Polygon([vertices[1], vertices[3], vertices[5], vertices[7]]),
                        Polygon([vertices[0], vertices[1], vertices[4], vertices[5]]),
                        Polygon([vertices[2], vertices[3], vertices[6], vertices[7]])]
            return polygons
        if len(vertices) == 6:
            polygons = [Polygon([vertices[0], vertices[1], vertices[2]]),
                        Polygon([vertices[0], vertices[1], vertices[4]]),
                        Polygon([vertices[0], vertices[2], vertices[3]]),
                        Polygon([vertices[0], vertices[3], vertices[4]]),
                        Polygon([vertices[5], vertices[1], vertices[2]]),
                        Polygon([vertices[5], vertices[1], vertices[4]]),
                        Polygon([vertices[5], vertices[2], vertices[3]]),
                        Polygon([vertices[5], vertices[3], vertices[4]])]
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
        self.create_polyhedron('Tetrahedron')
        self.create_dropdown()
        self.create_buttons()

    def create_dropdown(self):
        self.polyhedron_var = tk.StringVar(value='Tetrahedron')
        self.dropdown = ttk.Combobox(self.root, textvariable=self.polyhedron_var)
        self.dropdown['values'] = ['Tetrahedron', 'Cube', 'Octahedron']
        self.dropdown.bind("<<ComboboxSelected>>", self.change_polyhedron)
        self.dropdown.pack()       

    def change_polyhedron(self, event):
        self.create_polyhedron(self.polyhedron_var.get())
        self.render()
        
    def create_polyhedron(self, shape):
        if shape == 'Tetrahedron':
            vertices = [
                Point(300, 400, 300), 
                Point(200, 200, 400), 
                Point(400, 200, 400), 
                Point(300, 200, 200)
            ]
        elif shape == 'Cube':
            vertices = [
                Point(400, 400, 400), Point(400, 400, 200), Point(400, 200, 400), Point(400, 200, 200),
                Point(200, 400, 400), Point(200, 400, 200), Point(200, 200, 400), Point(200, 200, 200)
            ]
        elif shape == 'Octahedron':
            vertices = [
                Point(300, 400, 300), Point(400, 300, 300), Point(300, 300, 400), Point(200, 300, 300),
                Point(300, 300, 200), Point(300, 200, 300)
            ]
        else:
            vertices = []
        self.polyhedron = Polyhedron(polygons=Polygon.polygons_from_vertices(vertices))
        self.render()

    def create_buttons(self):
        transform_frame = tk.Frame(self.root)
        transform_frame.pack()

        tk.Label(transform_frame, text="dx:").grid(row=0, column=0)
        self.dx_entry = tk.Entry(transform_frame, width=5)
        self.dx_entry.grid(row=0, column=1)
        
        tk.Label(transform_frame, text="dy:").grid(row=0, column=2)
        self.dy_entry = tk.Entry(transform_frame, width=5)
        self.dy_entry.grid(row=0, column=3)

        tk.Label(transform_frame, text="dz:").grid(row=0, column=4)
        self.dz_entry = tk.Entry(transform_frame, width=5)
        self.dz_entry.grid(row=0, column=5)
        
        offset_button = tk.Button(transform_frame, text="Смещение", command=self.apply_offset)
        offset_button.grid(row=0, column=6, padx=5)

        tk.Label(transform_frame, text="Центр X:").grid(row=1, column=0)
        self.center_x_entry = tk.Entry(transform_frame, width=5)
        self.center_x_entry.grid(row=1, column=1)
        
        tk.Label(transform_frame, text="Центр Y:").grid(row=1, column=2)
        self.center_y_entry = tk.Entry(transform_frame, width=5)
        self.center_y_entry.grid(row=1, column=3)

        tk.Label(transform_frame, text="Центр Z:").grid(row=1, column=4)
        self.center_z_entry = tk.Entry(transform_frame, width=5)
        self.center_z_entry.grid(row=1, column=5)
        
        tk.Label(transform_frame, text="Угол:").grid(row=1, column=6)
        self.degrees_entry = tk.Entry(transform_frame, width=5)
        self.degrees_entry.grid(row=1, column=7)
        
        rotate_button = tk.Button(transform_frame, text="Поворот вокруг точки", command=self.apply_rotation)
        rotate_button.grid(row=1, column=8, padx=5)
        
        rotate_own_center_button = tk.Button(transform_frame, text="Поворот вокруг центра", command=self.apply_rotation_own_center)
        rotate_own_center_button.grid(row=1, column=9, padx=5)

        tk.Label(transform_frame, text="Масштаб:").grid(row=2, column=0)
        self.scale_factor_entry = tk.Entry(transform_frame, width=5)
        self.scale_factor_entry.grid(row=2, column=1)
        
        scale_button = tk.Button(transform_frame, text="Масштабирование от точки", command=self.apply_scaling)
        scale_button.grid(row=2, column=2, padx=5)
        
        scale_own_center_button = tk.Button(transform_frame, text="Масштабирование от центра", command=self.apply_scaling_own_center)
        scale_own_center_button.grid(row=2, column=3, padx=5)

    def apply_transform(self, matrix):
        self.polyhedron.transform(matrix)
        self.render()

    def apply_offset(self):
        if not self.polyhedron:
            return
            
        dx = int(self.dx_entry.get() or 0)
        dy = int(self.dy_entry.get() or 0)
        dz = int(self.dz_entry.get() or 0)
        
        self.apply_transform(self.get_translation_matrix(dx, dy, dz))
        
    def apply_rotation(self):
        if not self.polyhedron:
            return
            
        center_x = int(self.center_x_entry.get() or 0)
        center_y = int(self.center_y_entry.get() or 0)
        center_z = int(self.center_z_entry.get() or 0)
        degrees = int(self.degrees_entry.get() or 0)
        
        self.apply_transform(self.get_rotation_matrix(center_x, center_y, degrees))

    def apply_rotation_own_center(self):
        if not self.polyhedron:
            return
            
        degrees = int(self.degrees_entry.get() or 0)
        
        vertices = []        
        for i in range(len(self.polyhedron.polygons)):
            polygon = self.polyhedron.polygons[i]
            vertices += polygon.vertices

        center = np.mean([vert.coordinates for vert in vertices], axis=0)
        
        center_x = center[0]
        center_y = center[1]
        center_z = center[2]
            
        self.apply_transform(self.get_rotation_matrix(center_x, center_y, degrees))

    def apply_scaling(self):
        if not self.polyhedron:
            return
            
        scale_factor = float(self.scale_factor_entry.get() or 1.0)
        center_x = int(self.center_x_entry.get() or 0)
        center_y = int(self.center_y_entry.get() or 0)
        center_z = int(self.center_z_entry.get() or 0)
        
        self.apply_transform(self.get_scaling_matrix(center_x, center_y, center_z, scale_factor))
    
    def apply_scaling_own_center(self):
        if not self.polyhedron:
            return
            
        scale_factor = float(self.scale_factor_entry.get() or 1.0)

        vertices = []        
        for i in range(len(self.polyhedron.polygons)):
            polygon = self.polyhedron.polygons[i]
            vertices += polygon.vertices

        center = np.mean([vert.coordinates for vert in vertices], axis=0)
        
        center_x = center[0]
        center_y = center[1]
        center_z = center[2]

        self.apply_transform(self.get_scaling_matrix(center_x, center_y, center_z, scale_factor))

    def get_translation_matrix(self, dx, dy, dz):
        return np.array([[1, 0, 0, dx],
                         [0, 1, 0, dy],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])

    def get_rotation_matrix(self, center_x, center_y, angle):
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[cos_a, -sin_a, 0, center_x - center_x * cos_a + center_y * sin_a],
                         [sin_a, cos_a, 0, center_y - center_x * sin_a - center_y * cos_a],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def get_scaling_matrix(self, center_x, center_y, center_z, scale_factor):
        return np.array([[scale_factor, 0, 0, center_x * (1 - scale_factor)],
                         [0, scale_factor, 0, center_y * (1 - scale_factor)],
                         [0, 0, scale_factor, center_z * (1 - scale_factor)],
                         [0, 0, 0, 1]])

    def render(self):
        self.canvas.delete("all")
        
        for polygon in self.polyhedron.polygons:
            coords = [p.coordinates[:2] for p in polygon.vertices]  # Проекция в 2D
            if len(coords) == 3:
                self.canvas.create_polygon(coords[0][0], coords[0][1], coords[1][0], coords[1][1], coords[2][0], coords[2][1], outline="black", fill="", width=2)
            elif len(coords) == 4:
                self.canvas.create_polygon(coords[0][0], coords[0][1], coords[1][0], coords[1][1], coords[2][0], coords[2][1], coords[3][0], coords[3][1], outline="black", fill="", width=2)

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
