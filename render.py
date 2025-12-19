import numpy as np
from typing import Tuple
import struct
from dataclasses import dataclass
from math import sqrt

# Вспомогательные классы

class Vec3:
    """Вектор/точка в 3D пространстве"""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)   
        self.y = float(y)
        self.z = float(z)
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (int, float)):
            return Vec3(self.x * other, self.y * other, self.z * other)
        return None
    
    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return sqrt(self.dot(self))
    
    def normalize(self):
        l = self.length()
        if l > 0:
            return self * (1.0 / l)
        return Vec3(0, 0, 0)
    
    def __getitem__(self, index):
        return [self.x, self.y, self.z][index]
    
    def to_list(self):
        return [self.x, self.y, self.z]

class Vec2:
    """Вектор/точка в 2D пространстве"""
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
    
    def __getitem__(self, index):
        return [self.x, self.y][index]
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def to_list(self):
        return [self.x, self.y]

class Mat4:
    """Матрица 4x4 для преобразований"""
    def __init__(self, data=None):
        if data is None:
            self.data = np.identity(4, dtype=np.float32)
        else:
            self.data = np.array(data, dtype=np.float32).reshape(4, 4)
    
    @staticmethod
    def identity():
        return Mat4()
    
    @staticmethod
    def translate(x, y, z):
        m = Mat4.identity()
        m.data[0, 3] = x
        m.data[1, 3] = y
        m.data[2, 3] = z
        return m
    
    @staticmethod
    def scale(x, y, z):
        m = Mat4.identity()
        m.data[0, 0] = x
        m.data[1, 1] = y
        m.data[2, 2] = z
        return m
    
    @staticmethod
    def rotate_x(angle):
        m = Mat4.identity()
        c = np.cos(angle)
        s = np.sin(angle)
        m.data[1, 1] = c
        m.data[1, 2] = -s
        m.data[2, 1] = s
        m.data[2, 2] = c
        return m
    
    @staticmethod
    def rotate_y(angle):
        m = Mat4.identity()
        c = np.cos(angle)
        s = np.sin(angle)
        m.data[0, 0] = c
        m.data[0, 2] = s
        m.data[2, 0] = -s
        m.data[2, 2] = c
        return m
    
    @staticmethod
    def rotate_z(angle):
        m = Mat4.identity()
        c = np.cos(angle)
        s = np.sin(angle)
        m.data[0, 0] = c
        m.data[0, 1] = -s
        m.data[1, 0] = s
        m.data[1, 1] = c
        return m
    
    def __mul__(self, other):
        if isinstance(other, Mat4):
            result = Mat4()
            result.data = np.dot(self.data, other.data)
            return result
        elif isinstance(other, Vec3):
            # Преобразование вектора с добавлением w=1
            v = np.array([other.x, other.y, other.z, 1.0])
            result = np.dot(self.data, v)
            w = result[3]
            if abs(w) > 1e-8:
                return Vec3(result[0]/w, result[1]/w, result[2]/w)
            return Vec3(result[0], result[1], result[2])
        return None
    
    def transpose(self):
        m = Mat4()
        m.data = self.data.T
        return m

class Camera:
    """Класс камеры с матрицами вида и проекции"""
    def __init__(self, position=Vec3(0, 0, 5), target=Vec3(0, 0, 0), up=Vec3(0, 1, 0),
                 fov=60.0, aspect_ratio=1.0, near=0.1, far=100.0):
        self.position = position
        self.target = target
        self.up = up
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far
        
        self.update_view_matrix()
        self.update_projection_matrix()
    
    def update_view_matrix(self):
        """Создание матрицы вида"""
        forward = (self.target - self.position).normalize()
        right = forward.cross(self.up).normalize()
        up = right.cross(forward)
        
        # Матрица вида
        self.view_matrix = Mat4([
            [right.x, right.y, right.z, -right.dot(self.position)],
            [up.x, up.y, up.z, -up.dot(self.position)],
            [-forward.x, -forward.y, -forward.z, forward.dot(self.position)],
            [0, 0, 0, 1]
        ])
    
    def update_projection_matrix(self):
        """Создание матрицы перспективной проекции"""
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        self.projection_matrix = Mat4([
            [f / self.aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far + self.near) / (self.near - self.far), 
             (2 * self.far * self.near) / (self.near - self.far)],
            [0, 0, -1, 0]
        ])
    
    def get_view_projection_matrix(self):
        """Комбинированная матрица вида-проекции"""
        return self.projection_matrix * self.view_matrix

# Классы для рендера

@dataclass
class Material:
    """Материал для Phong освещения"""
    ambient: Vec3 = Vec3(0.2, 0.2, 0.2)
    diffuse: Vec3 = Vec3(0.7, 0.7, 0.7)
    specular: Vec3 = Vec3(1.0, 1.0, 1.0)
    shininess: float = 32.0

class Model:
    """Модель из OBJ файла"""
    def __init__(self, filename=None):
        self.vertices = []
        self.normals = []
        self.faces = []
        self.material = Material()
        if filename:
            self.load_obj(filename)
    
    def load_obj(self, filename):
        """Загрузка OBJ файла"""
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    self.vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == 'vn':
                    self.normals.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == 'f':
                    face_verts = []
                    face_norms = []
                    for part in parts[1:]:
                        indices = part.split('/')
                        v_idx = int(indices[0]) - 1
                        face_verts.append(v_idx)
                        
                        if len(indices) >= 3 and indices[2]:
                            n_idx = int(indices[2]) - 1
                            face_norms.append(n_idx)
                        else:
                            face_norms.append(-1)
                    
                    # Преобразование в треугольники если нужно
                    if len(face_verts) >= 3:
                        for i in range(1, len(face_verts) - 1):
                            self.faces.append((
                                [face_verts[0], face_verts[i], face_verts[i+1]],
                                [face_norms[0], face_norms[i], face_norms[i+1]]
                            ))

# Псевдошейдеры

class PhongShader:
    """Шейдер для Phong освещения"""
    def __init__(self):
        self.light_pos = Vec3(5, 5, 5)
        self.light_color = Vec3(1, 1, 1)
        self.view_pos = Vec3(0, 0, 5)
        
    def vertex_shader(self, vertex: Vec3, normal: Vec3, mvp_matrix: Mat4, 
                     model_matrix: Mat4, view_matrix: Mat4) -> Tuple[Vec3, Vec3]:
        """Вершинный шейдер"""
        # Преобразование позиции
        world_pos = model_matrix * vertex
        clip_pos = mvp_matrix * vertex
        
        # Преобразование нормали (транспонированная обратная матрица модели)
        normal_matrix = model_matrix
        world_normal = normal_matrix * normal
        world_normal = world_normal.normalize()
        
        return clip_pos, world_normal, world_pos
    
    def fragment_shader(self, normal: Vec3, world_pos: Vec3, material: Material) -> Vec3:
        """Фрагментный шейдер (Phong освещение)"""
        # Нормализация
        N = normal.normalize()
        
        # Направление к свету
        L_dir = (self.light_pos - world_pos).normalize()
        
        # Направление к камере
        V_dir = (self.view_pos - world_pos).normalize()
        
        # Отраженный луч
        R_dir = (N * (2.0 * N.dot(L_dir)) - L_dir).normalize()
        
        # Вычисление компонентов освещения
        # Фон
        ambient = material.ambient * Vec3(0.2, 0.2, 0.2)
        
        # Диффуз
        diff = max(N.dot(L_dir), 0.0)
        diffuse = material.diffuse * self.light_color * diff
        
        # Блики
        spec = max(V_dir.dot(R_dir), 0.0)
        spec = pow(spec, material.shininess)
        specular = material.specular * self.light_color * spec
        
        # Итоговый цвет
        color = ambient + diffuse + specular
        
        # Ограничение значений
        color.x = min(max(color.x, 0.0), 1.0)
        color.y = min(max(color.y, 0.0), 1.0)
        color.z = min(max(color.z, 0.0), 1.0)
        
        return color

# Растеризатор

class Rasterizer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.framebuffer = np.zeros((height, width, 3), dtype=np.float32)
        self.z_buffer = np.full((height, width), float('inf'), dtype=np.float32)
        self.shader = PhongShader()
        
    def clear(self, color=Vec3(0, 0, 0)):
        """Очистка буферов"""
        self.framebuffer[:, :] = color.to_list()
        self.z_buffer.fill(float('inf'))
    
    def barycentric(self, p: Vec2, a: Vec2, b: Vec2, c: Vec2):
        """Вычисление барицентрических координат"""
        v0 = Vec2(b.x - a.x, b.y - a.y)
        v1 = Vec2(c.x - a.x, c.y - a.y)
        v2 = Vec2(p.x - a.x, p.y - a.y)
        
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            return None
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return u, v, w
    
    def rasterize_triangle(self, v0, v1, v2, n0, n1, n2, 
                          world0, world1, world2, material):
        """Растеризация треугольника с барицентрической интерполяцией"""
        # Поиск баундбокса
        min_x = max(0, int(min(v0.x, v1.x, v2.x)))
        max_x = min(self.width - 1, int(max(v0.x, v1.x, v2.x)))
        min_y = max(0, int(min(v0.y, v1.y, v2.y)))
        max_y = min(self.height - 1, int(max(v0.y, v1.y, v2.y)))
        
        if min_x > max_x or min_y > max_y:
            return
        
        # Преобразование в 2D для барицентрических координат
        a = Vec2(v0.x, v0.y)
        b = Vec2(v1.x, v1.y)
        c = Vec2(v2.x, v2.y)
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = Vec2(x + 0.5, y + 0.5)  # центр пикселя
                
                # Барицентрические координаты
                coords = self.barycentric(p, a, b, c)
                if coords is None:
                    continue
                
                u, v, w = coords
                
                # Проверка на попадание в треугольник
                if u < 0 or v < 0 or w < 0:
                    continue
                
                # Интерполяция глубины
                z = u * v0.z + v * v1.z + w * v2.z
                
                # Проверка зет буфера
                if z >= self.z_buffer[y, x]:
                    continue
                
                # Интерполяция нормали и позиции в мировых координатах
                interp_normal = (n0 * u + n1 * v + n2 * w).normalize()
                interp_world = world0 * u + world1 * v + world2 * w
                
                # Затемнение
                color = self.shader.fragment_shader(interp_normal, interp_world, material)
                
                # Обновление буферов
                self.z_buffer[y, x] = z
                self.framebuffer[y, x] = color.to_list()
    
    def render(self, model: Model, camera: Camera):
        """Рендеринг модели"""
        self.clear(Vec3(0.1, 0.1, 0.1))
        
        # Матрицы преобразований
        model_matrix = Mat4.identity()
        mvp_matrix = camera.get_view_projection_matrix() * model_matrix
        view_matrix = camera.view_matrix
        
        # Обновление позиции камеры в шейдере
        self.shader.view_pos = camera.position
        
        # Рендеринг каждого треугольника
        for face in model.faces:
            vertices_idx, normals_idx = face
            triangle_verts = []
            triangle_norms = []
            triangle_world = []
            
            for i in range(3):
                # Получение вершины и нормали
                vert = model.vertices[vertices_idx[i]]
                if normals_idx[i] >= 0:
                    norm = model.normals[normals_idx[i]]
                else:
                    norm = Vec3(0, 1, 0)  # нормаль по умолчанию
                
                # Применение шейдера
                clip_pos, world_norm, world_pos = self.shader.vertex_shader(
                    vert, norm, mvp_matrix, model_matrix, view_matrix
                )
                
                # Преобразование в экранные координаты
                screen_x = (clip_pos.x + 1.0) * 0.5 * self.width
                screen_y = (1.0 - clip_pos.y) * 0.5 * self.height
                screen_pos = Vec3(screen_x, screen_y, clip_pos.z)
                
                triangle_verts.append(screen_pos)
                triangle_norms.append(world_norm)
                triangle_world.append(world_pos)
            
            # Растеризация треугольника
            self.rasterize_triangle(
                triangle_verts[0], triangle_verts[1], triangle_verts[2],
                triangle_norms[0], triangle_norms[1], triangle_norms[2],
                triangle_world[0], triangle_world[1], triangle_world[2],
                model.material
            )
    
    def save_tga(self, filename):
        """Сохранение изображения в формате TGA"""
        # Конвертация в 8-битный формат
        img_data = (self.framebuffer * 255).astype(np.uint8)
        
        # Важная штука!!
        # Заголовок TGA с флагом для хранения сверху вниз
        header = struct.pack('B' * 18,
            0,  # ID length
            0,  # Color map type
            2,  # Image type: uncompressed true-color
            0, 0,  # Color map origin
            0, 0,  # Color map length
            0,  # Color map depth
            0, 0,  # X origin
            0, 0,  # Y origin
            self.width & 0xFF, (self.width >> 8) & 0xFF,  # Width
            self.height & 0xFF, (self.height >> 8) & 0xFF,  # Height
            24,  # Pixel depth
            32   # Image descriptor: 32 = верхняя строка первая (0x20)
        )
        
        # Запись файла
        with open(filename, 'wb') as f:
            f.write(header)
            # TGA хранит данные в порядке BGR потому что его делал психически нездоровый чел
            for y in range(self.height):
                for x in range(self.width):
                    b, g, r = img_data[y, x]
                    f.write(struct.pack('BBB', b, g, r))

# На случай если всё плохо

def create_simple_cube():
    """Создание простой кубической модели для тестирования"""
    model = Model()
    
    # Вершины куба
    model.vertices = [
        Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
        Vec3(-1, -1, 1), Vec3(1, -1, 1), Vec3(1, 1, 1), Vec3(-1, 1, 1)
    ]
    
    # Нормали для каждой грани
    model.normals = [
        Vec3(0, 0, -1),   # передняя
        Vec3(0, 0, 1),    # задняя
        Vec3(0, -1, 0),   # нижняя
        Vec3(0, 1, 0),    # верхняя
        Vec3(-1, 0, 0),   # левая
        Vec3(1, 0, 0)     # правая
    ]
    
    # Грани куба
    faces = [
        # Передняя грань
        ([0, 1, 2], [0, 0, 0]),
        ([0, 2, 3], [0, 0, 0]),
        # Задняя грань
        ([4, 6, 5], [1, 1, 1]),
        ([4, 7, 6], [1, 1, 1]),
        # Нижняя грань
        ([0, 4, 5], [2, 2, 2]),
        ([0, 5, 1], [2, 2, 2]),
        # Верхняя грань
        ([3, 2, 6], [3, 3, 3]),
        ([3, 6, 7], [3, 3, 3]),
        # Левая грань
        ([0, 3, 7], [4, 4, 4]),
        ([0, 7, 4], [4, 4, 4]),
        # Правая грань
        ([1, 5, 6], [5, 5, 5]),
        ([1, 6, 2], [5, 5, 5])
    ]
    
    model.faces = faces
    model.material = Material(
        ambient=Vec3(0.3, 0.3, 0.3),
        diffuse=Vec3(0.65, 0.65, 0.65),
        specular=Vec3(1.0, 1.0, 1.0),
        shininess=32.0
    )
    
    return model

def main():
    rasterizer = Rasterizer(800, 600)
    
    camera = Camera(
        position=Vec3(3, 3, 3),
        target=Vec3(0, 0, 0),
        up=Vec3(0, 1, 0),
        fov=60.0,
        aspect_ratio=800/600,
        near=0.1,
        far=100.0
    )
    
    try:
        model = Model("model.obj")
        print("Модель загружена из model.obj")
    except FileNotFoundError:
        print("Файл model.obj не найден. Используется тестовый куб.")
        model = create_simple_cube()
    
    print("Рендеринг...")
    rasterizer.render(model, camera)
    
    output_file = "output.tga"
    rasterizer.save_tga(output_file)
    print(f"Изображение сохранено в {output_file}")

if __name__ == "__main__":
    main()
