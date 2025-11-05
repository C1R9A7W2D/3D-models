import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Callable, List, Tuple

class SurfaceModel:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.function_name = ""
        
    def build_surface(self, func: Callable, x_range: Tuple[float, float], 
                     y_range: Tuple[float, float], num_points: int):
        self.function_name = func.__name__
        x0, x1 = x_range
        y0, y1 = y_range
        
        x = np.linspace(x0, x1, num_points) #разбиение
        y = np.linspace(y0, y1, num_points)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)
        
        # Сохранение вершин
        self.vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Построение граней (треугольников)
        self.faces = []
        for i in range(num_points - 1):
            for j in range(num_points - 1):
                # Два треугольника на каждый квадрат сетки
                idx1 = i * num_points + j
                idx2 = i * num_points + (j + 1)
                idx3 = (i + 1) * num_points + j
                idx4 = (i + 1) * num_points + (j + 1)
                
                self.faces.append([idx1, idx2, idx3])  # Первый треугольник
                self.faces.append([idx2, idx4, idx3])  # Второй треугольник
    
    def apply_transformation(self, transformation_matrix: np.ndarray):
        if self.vertices is not None:
            # Добавляем столбец единиц для аффинных преобразований
            homogeneous_vertices = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
            transformed_vertices = homogeneous_vertices @ transformation_matrix.T
            self.vertices = transformed_vertices[:, :3]

class SurfacePlotter:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.model = SurfaceModel()
    
    def create_figure(self):
        if self.fig is not None:
            plt.close(self.fig)  # Закрываем предыдущую фигуру
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
    
    def plot_surface(self, title: str = ""):
        if self.ax is None:
            self.create_figure()
        else:
            self.ax.clear()
        
        if self.model.vertices is not None and self.model.faces is not None:
            vertices = self.model.vertices
            
            # Отрисовка каждой грани
            for face in self.model.faces:
                triangle = vertices[face]
                self.ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                                   alpha=0.7, linewidth=0.2, antialiased=True)
            
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title(title)
    
    def show(self):
        if self.fig is not None:
            plt.tight_layout()
            plt.show()
        else:
            print("Нет данных для отображения")

# Примеры математических функций
def parabolic(x, y):
    return x**2 + y**2

def hyperbolic(x, y):
    return x**2 - y**2

def sine_wave(x, y):
    return np.sin(x) + np.cos(y)

def sphere(x, y):
    return np.sqrt(np.maximum(0, 1 - x**2 - y**2))

def ripples(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

# Аффинные преобразования
def translation_matrix(dx: float, dy: float, dz: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

def rotation_x_matrix(angle_degrees: float) -> np.ndarray:
    angle = np.radians(angle_degrees)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_y_matrix(angle_degrees: float) -> np.ndarray:
    angle = np.radians(angle_degrees)
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_z_matrix(angle_degrees: float) -> np.ndarray:
    angle = np.radians(angle_degrees)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def apply_transformations_interactive(plotter):
    while True:
        print("\n--- ПРЕОБРАЗОВАНИЯ ---")
        print("1. Перенос")
        print("2. Поворот вокруг X")
        print("3. Поворот вокруг Y")
        print("4. Поворот вокруг Z")
        print("5. Масштабирование")
        print("6. Завершить преобразования")
        
        choice = input("Выберите преобразование (1-6): ").strip()
        
        if choice == "6":
            break
            
        try:
            if choice == "1":
                dx = float(input("Смещение по X: ") or 0)
                dy = float(input("Смещение по Y: ") or 0)
                dz = float(input("Смещение по Z: ") or 0)
                plotter.model.apply_transformation(translation_matrix(dx, dy, dz))
                print("Перенос применен.")
                
            elif choice == "2":
                angle = float(input("Угол поворота вокруг X (градусы): ") or 0)
                plotter.model.apply_transformation(rotation_x_matrix(angle))
                print("Поворот применен.")
                
            elif choice == "3":
                angle = float(input("Угол поворота вокруг Y (градусы): ") or 0)
                plotter.model.apply_transformation(rotation_y_matrix(angle))
                print("Поворот применен.")
                
            elif choice == "4":
                angle = float(input("Угол поворота вокруг Z (градусы): ") or 0)
                plotter.model.apply_transformation(rotation_z_matrix(angle))
                print("Поворот применен.")
                
            elif choice == "5":
                sx = float(input("Масштаб по X: ") or 1)
                sy = float(input("Масштаб по Y: ") or 1)
                sz = float(input("Масштаб по Z: ") or 1)
                plotter.model.apply_transformation(scale_matrix(sx, sy, sz))
                print("Масштабирование применено.")
                
            else:
                print("Неверный выбор.")
                continue
                
        except ValueError:
            print("Неверный ввод параметров.")
            continue
        
        # Закрываем предыдущее окно и показываем новое с преобразованной фигурой
        print("Показываю фигуру после преобразования...")
        plotter.plot_surface(f"Поверхность после преобразований: {plotter.model.function_name}")
        plotter.show()

def create_new_surface():
    print("\n--- СОЗДАНИЕ НОВОЙ ПОВЕРХНОСТИ ---")
    
    # Выбор функции
    functions = [parabolic, hyperbolic, sine_wave, sphere, ripples]
    names = ["Параболоид", "Гиперболический параболоид", "Синусоида", "Полусфера", "Круговые волны"]
    
    print("Доступные функции:")
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")
    
    try:
        func_choice = int(input("Выберите функцию (1-5): ")) - 1
        if func_choice < 0 or func_choice >= len(functions):
            print("Неверный выбор функции.")
            return
    except ValueError:
        print("Неверный ввод.")
        return
    
    # Параметры построения
    try:
        x0 = float(input("Начало диапазона X (по умолчанию -2): ") or -2)
        x1 = float(input("Конец диапазона X (по умолчанию 2): ") or 2)
        y0 = float(input("Начало диапазона Y (по умолчанию -2): ") or -2)
        y1 = float(input("Конец диапазона Y (по умолчанию 2): ") or 2)
        points = int(input("Количество точек разбиения (по умолчанию 20): ") or 20)
    except ValueError:
        print("Неверный ввод параметров.")
        return
    
    # Построение поверхности
    plotter = SurfacePlotter()
    plotter.model.build_surface(functions[func_choice], (x0, x1), (y0, y1), points)
    plotter.plot_surface(f"Новая поверхность: {names[func_choice]}")
    plotter.show()
    
    # Применение преобразований
    apply_transformations_interactive(plotter)

if __name__ == "__main__":
    create_new_surface()