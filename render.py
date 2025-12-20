import numpy as np
from typing import Tuple, Optional
import struct
from dataclasses import dataclass
from math import sqrt
from PIL import Image
import os

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

class Texture:
    """Класс для работы с текстурами"""
    def __init__(self, filename=None):
        self.width = 0
        self.height = 0
        self.data = None
        if filename:
            self.load_texture(filename)
    
    def load_texture(self, filename):
        """Загрузка текстуры из файла"""
        try:
            img = Image.open(filename)
            img = img.convert('RGB')
            self.width, self.height = img.size
            self.data = np.array(img, dtype=np.uint8)
            print(f"Текстура загружена: {filename} ({self.width}x{self.height})")
        except Exception as e:
            print(f"Ошибка загрузки текстуры {filename}: {e}")
            # Дефолтная шахматная доска
            self.create_default_texture()
    
    def create_default_texture(self):
        """Создание текстуры по умолчанию (шахматная доска)"""
        self.width = self.height = 256
        self.data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Шахматная доска 8x8
        cell_size = 32
        for y in range(self.height):
            for x in range(self.width):
                cell_x = x // cell_size
                cell_y = y // cell_size
                if (cell_x + cell_y) % 2 == 0:
                    self.data[y, x] = [255, 255, 255]  # Белый
                else:
                    self.data[y, x] = [0, 0, 255]  # Синий
    
    def get_color(self, u, v):
        """Получение цвета текстуры по координатам (u, v)"""
        if self.data is None:
            return Vec3(1, 1, 1)  # Белый цвет по умолчанию
        
        # Приведение координат к диапазону [0, 1] и ограничение
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        
        # Преобразование в координаты текстуры
        x = int(u * (self.width - 1))
        y = int((1.0 - v) * (self.height - 1))  # Инвертируем v для корректного отображения
        
        # Получение цвета
        color = self.data[y, x]
        return Vec3(color[0]/255.0, color[1]/255.0, color[2]/255.0)

@dataclass
class Material:
    """Материал для Phong освещения"""
    ambient: Vec3 = Vec3(0.5, 0.5, 0.5)
    diffuse: Vec3 = Vec3(0.7, 0.7, 0.7)
    specular: Vec3 = Vec3(1.0, 1.0, 1.0)
    shininess: float = 32.0
    texture: Optional[Texture] = None

class Model:
    """Модель из OBJ файла"""
    def __init__(self, filename=None):
        self.vertices = []
        self.normals = []
        self.tex_coords = []
        self.faces = []
        self.materials = {}
        self.current_material = Material()
        if filename:
            self.load_obj(filename)
    
    def load_obj(self, filename):
        """Загрузка OBJ файла с поддержкой материалов"""
        obj_dir = os.path.dirname(os.path.abspath(filename))
        if obj_dir == "":
            obj_dir = "."
        
        print(f"Загрузка модели из: {filename}")
        print(f"Директория модели: {obj_dir}")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        current_material_name = None
        materials_loaded = {}
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            cmd = parts[0]
            
            if cmd == 'mtllib':
                # Загрузка файла материалов
                mtl_filename = parts[1]
                mtl_path = os.path.join(obj_dir, mtl_filename)
                print(f"Загрузка материалов из: {mtl_path}")
                materials_loaded = self.load_mtl(mtl_path)
            
            elif cmd == 'usemtl':
                # Использование конкретного материала
                material_name = parts[1]
                if material_name in materials_loaded:
                    self.current_material = materials_loaded[material_name]
                    current_material_name = material_name
                    print(f"Используется материал: {material_name}")
                else:
                    print(f"Предупреждение: материал {material_name} не найден, используется стандартный")
                    self.current_material = Material()
                    current_material_name = "default"
            
            elif cmd == 'v':
                # Вершина: v x y z
                if len(parts) >= 4:
                    self.vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                else:
                    print(f"Предупреждение строка {line_num}: некорректная вершина")
            
            elif cmd == 'vn':
                # Нормаль: vn x y z
                if len(parts) >= 4:
                    self.normals.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                else:
                    print(f"Предупреждение строка {line_num}: некорректная нормаль")
            
            elif cmd == 'vt':
                # Текстурная координата: vt u v
                if len(parts) >= 3:
                    u = float(parts[1])
                    v = float(parts[2])
                    self.tex_coords.append(Vec2(u, v))
                elif len(parts) >= 2:
                    u = float(parts[1])
                    self.tex_coords.append(Vec2(u, 0.0))
                else:
                    print(f"Предупреждение строка {line_num}: некорректная текстурная координата")
            
            elif cmd == 'f':
                # Грань: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
                if len(parts) < 4:
                    print(f"Предупреждение строка {line_num}: грань должна иметь минимум 3 вершины")
                    continue
                
                face_verts = []
                face_texs = []
                face_norms = []
                
                for vertex_str in parts[1:]:
                    # Индексы: vertex/texture/normal
                    indices = vertex_str.split('/')
                    
                    # Индекс вершины (обязательный)
                    if indices[0]:
                        v_idx = int(indices[0]) - 1  # OBJ использует 1-индексацию
                        if v_idx < 0 or v_idx >= len(self.vertices):
                            print(f"Ошибка строка {line_num}: индекс вершины {v_idx+1} вне диапазона")
                            v_idx = 0
                        face_verts.append(v_idx)
                    else:
                        print(f"Ошибка строка {line_num}: отсутствует индекс вершины")
                        face_verts.append(0)
                    
                    # Текстурная координата (опциональна)
                    if len(indices) > 1 and indices[1]:
                        t_idx = int(indices[1]) - 1
                        if t_idx < 0 or t_idx >= len(self.tex_coords):
                            print(f"Ошибка строка {line_num}: индекс текстурной координаты {t_idx+1} вне диапазона")
                            t_idx = 0 if self.tex_coords else -1
                        face_texs.append(t_idx)
                    else:
                        face_texs.append(-1)  # -1 означает отсутствие текстурной координаты
                    
                    # Нормаль (опциональна)
                    if len(indices) > 2 and indices[2]:
                        n_idx = int(indices[2]) - 1
                        if n_idx < 0 or n_idx >= len(self.normals):
                            print(f"Ошибка строка {line_num}: индекс нормали {n_idx+1} вне диапазона")
                            n_idx = 0 if self.normals else -1
                        face_norms.append(n_idx)
                    else:
                        face_norms.append(-1)  # -1 означает отсутствие нормали
                
                # Разбивка грани на треугольники (триангуляция)
                if len(face_verts) >= 3:
                    # Для полигонов с более чем 3 вершинами используется веер треугольников
                    for i in range(1, len(face_verts) - 1):
                        self.faces.append((
                            [face_verts[0], face_verts[i], face_verts[i+1]],
                            [face_texs[0], face_texs[i], face_texs[i+1]],
                            [face_norms[0], face_norms[i], face_norms[i+1]],
                            current_material_name if current_material_name else "default"
                        ))
                else:
                    print(f"Ошибка строка {line_num}: недостаточно вершин в грани")
        
        print(f"Загрузка завершена:")
        print(f"  Вершин: {len(self.vertices)}")
        print(f"  Нормалей: {len(self.normals)}")
        print(f"  Текстурных координат: {len(self.tex_coords)}")
        print(f"  Граней (треугольников): {len(self.faces)}")
        print(f"  Материалов: {len(materials_loaded)}")
        
        # Сохраняем загруженные материалы
        self.materials = materials_loaded
    
    def load_mtl(self, mtl_path):
        """Загрузка файла материалов MTL"""
        materials = {}
        current_mtl = None
        
        try:
            print(f"Попытка загрузить MTL: {mtl_path}")
            
            if not os.path.exists(mtl_path):
                print(f"MTL файл не найден: {mtl_path}")
                return materials
            
            with open(mtl_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    cmd = parts[0]
                    
                    if cmd == 'newmtl':
                        # Новый материал
                        current_mtl = parts[1]
                        materials[current_mtl] = Material()
                        print(f"  Найден материал: {current_mtl}")
                    
                    elif current_mtl:
                        if cmd == 'Ka':
                            # Ambient цвет
                            if len(parts) >= 4:
                                materials[current_mtl].ambient = Vec3(
                                    float(parts[1]), 
                                    float(parts[2]), 
                                    float(parts[3])
                                )
                        
                        elif cmd == 'Kd':
                            # Diffuse цвет
                            if len(parts) >= 4:
                                materials[current_mtl].diffuse = Vec3(
                                    float(parts[1]), 
                                    float(parts[2]), 
                                    float(parts[3])
                                )
                        
                        elif cmd == 'Ks':
                            # Specular цвет
                            if len(parts) >= 4:
                                materials[current_mtl].specular = Vec3(
                                    float(parts[1]), 
                                    float(parts[2]), 
                                    float(parts[3])
                                )
                        
                        elif cmd == 'Ns':
                            # Shininess
                            if len(parts) >= 2:
                                materials[current_mtl].shininess = float(parts[1])
                        
                        elif cmd == 'map_Kd':
                            # Текстура диффузного цвета
                            tex_filename = parts[1]
                            # Убираем возможные кавычки
                            tex_filename = tex_filename.strip('"')
                            
                            # Построение полного пути к текстуре
                            mtl_dir = os.path.dirname(mtl_path)
                            if mtl_dir == "":
                                mtl_dir = "."
                            
                            tex_path = os.path.join(mtl_dir, tex_filename)
                            print(f"  Загрузка текстуры для {current_mtl}: {tex_path}")
                            
                            # Создаем текстуру
                            materials[current_mtl].texture = Texture(tex_path)
                        
                        elif cmd == 'd' or cmd == 'Tr':
                            # Прозрачность
                            if len(parts) >= 2:
                                transparency = float(parts[1])
                                # Можно сохранить прозрачность в материале
                                # materials[current_mtl].transparency = transparency
        
        except Exception as e:
            print(f"Ошибка при загрузке MTL файла {mtl_path}: {e}")
        
        print(f"Загружено материалов: {len(materials)}")
        return materials

    def compute_normals(self):
        """Вычисление нормалей для модели, если они не загружены из OBJ"""
        if not self.normals:
            # Инициализируем нормали нулевыми векторами
            self.normals = [Vec3(0, 0, 0) for _ in range(len(self.vertices))]
            
            # Для каждого треугольника вычисляем нормаль и добавляем к вершинам
            for face in self.faces:
                vertices_idx, texs_idx, normals_idx, material_name = face
                
                if len(vertices_idx) == 3:
                    v0 = self.vertices[vertices_idx[0]]
                    v1 = self.vertices[vertices_idx[1]]
                    v2 = self.vertices[vertices_idx[2]]
                    
                    # Вычисляем нормаль треугольника
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    face_normal = edge1.cross(edge2).normalize()
                    
                    # Добавляем нормаль треугольника к каждой вершине
                    for v_idx in vertices_idx:
                        self.normals[v_idx] = self.normals[v_idx] + face_normal
            
            # Нормализуем все нормали
            for i in range(len(self.normals)):
                self.normals[i] = self.normals[i].normalize()
            
            print(f"Вычислены нормали для {len(self.normals)} вершин")
        
        # Обновляем индексы нормалей в гранях
        for i, face in enumerate(self.faces):
            vertices_idx, texs_idx, normals_idx, material_name = face
            # Используем те же индексы, что и для вершин (предполагаем per-vertex нормали)
            new_normals_idx = vertices_idx.copy()
            self.faces[i] = (vertices_idx, texs_idx, new_normals_idx, material_name)

    def compute_smooth_normals(self):
        """Вычисление сглаженных нормалей по методу усреднения"""
        # Создаем список нормалей для каждой вершины
        vertex_normals = [[] for _ in range(len(self.vertices))]
        
        # Для каждого треугольника
        for face in self.faces:
            vertices_idx, texs_idx, normals_idx, material_name = face
            
            if len(vertices_idx) == 3:
                v0 = self.vertices[vertices_idx[0]]
                v1 = self.vertices[vertices_idx[1]]
                v2 = self.vertices[vertices_idx[2]]
                
                # Вычисляем нормаль треугольника
                edge1 = v1 - v0
                edge2 = v2 - v0
                face_normal = edge1.cross(edge2).normalize()
                
                # Добавляем нормаль треугольника к списку нормалей каждой вершины
                for v_idx in vertices_idx:
                    vertex_normals[v_idx].append(face_normal)
        
        # Вычисляем усредненные нормали для каждой вершины
        self.normals = []
        for normals_list in vertex_normals:
            if normals_list:
                # Суммируем все нормали
                avg_normal = Vec3(0, 0, 0)
                for normal in normals_list:
                    avg_normal = avg_normal + normal
                # Нормализуем результат
                self.normals.append(avg_normal.normalize())
            else:
                # Если у вершины нет нормалей, используем стандартную
                self.normals.append(Vec3(0, 1, 0))
        
        print(f"Вычислены сглаженные нормали для {len(self.normals)} вершин")

# Псевдошейдеры

class PhongShader:
    """Шейдер для Phong освещения, но, внезапно, с текстурами"""
    def __init__(self):
        self.light_pos = Vec3(5, 5, 5)
        self.light_color = Vec3(1, 1, 1)
        self.view_pos = Vec3(0, 0, 5)
        
    def vertex_shader(self, vertex: Vec3, tex_coord: Vec2, normal: Vec3, 
                 mvp_matrix: Mat4, model_matrix: Mat4, view_matrix: Mat4) -> Tuple[Vec3, Vec2, Vec3, Vec3]:
        """Вершинный шейдер"""
        # Преобразование позиции
        world_pos = model_matrix * vertex
        clip_pos = mvp_matrix * vertex
        
        # ПРАВИЛЬНОЕ преобразование нормали
        # Для нормалей нужно использовать транспонированную обратную матрицу модели
        # Упрощенный вариант для изотропного масштабирования:
        normal_matrix = model_matrix
        world_normal = normal_matrix * normal
        world_normal = world_normal.normalize()
        
        return clip_pos, tex_coord, world_normal, world_pos
    
    def fragment_shader(self, tex_coord: Vec2, normal: Vec3, world_pos: Vec3, 
                       material: Material, texture: Texture = None) -> Vec3:
        """Фрагментный шейдер (Phong освещение с текстурой)"""
        # Нормализация
        N = normal.normalize()
        
        # Направление к свету
        L_dir = (self.light_pos - world_pos).normalize()
        
        # Направление к камере
        V_dir = (self.view_pos - world_pos).normalize()
        
        # Отраженный луч
        R_dir = (N * (2.0 * N.dot(L_dir)) - L_dir).normalize()
        
        # Базовый цвет: из текстуры или материала
        if texture:
            base_color = texture.get_color(tex_coord.x, tex_coord.y)
        else:
            base_color = material.diffuse
        
        # Фон
        ambient = Vec3(
            material.ambient.x * 0.2 * base_color.x,
            material.ambient.y * 0.2 * base_color.y,
            material.ambient.z * 0.2 * base_color.z
        )
        
        # Диффуз
        diff = max(N.dot(L_dir), 0.0)
        diffuse = Vec3(
            base_color.x * self.light_color.x * diff,
            base_color.y * self.light_color.y * diff,
            base_color.z * self.light_color.z * diff
        )
        
        # Блики
        spec = max(V_dir.dot(R_dir), 0.0)
        spec = pow(spec, material.shininess)
        specular = Vec3(
            material.specular.x * self.light_color.x * spec,
            material.specular.y * self.light_color.y * spec,
            material.specular.z * self.light_color.z * spec
        )
        
        # Итоговый цвет
        color = Vec3(
            ambient.x + diffuse.x + specular.x,
            ambient.y + diffuse.y + specular.y,
            ambient.z + diffuse.z + specular.z
        )
        
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
    
    def barycentric(self, p, a, b, c):
        """Быстрое вычисление барицентрических координат"""
        v0 = Vec2(b.x - a.x, b.y - a.y)
        v1 = Vec2(c.x - a.x, c.y - a.y)
        v2 = Vec2(p.x - a.x, p.y - a.y)
        
        d00 = v0.x * v0.x + v0.y * v0.y
        d01 = v0.x * v1.x + v0.y * v1.y
        d11 = v1.x * v1.x + v1.y * v1.y
        d20 = v2.x * v0.x + v2.y * v0.y
        d21 = v2.x * v1.x + v2.y * v1.y
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            return None
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return u, v, w
    
    def rasterize_triangle(self, v0, v1, v2, t0, t1, t2, n0, n1, n2, 
                      world0, world1, world2, material, texture=None):
        """Растеризация треугольника с барицентрической интерполяцией"""
        # # Backface Culling - отсечение нелицевых граней
        # edge1 = v1 - v0
        # edge2 = v2 - v0
        # normal_z = edge1.x * edge2.y - edge1.y * edge2.x
        
        # if normal_z <= 0:  # Если грань смотрит от камеры - пропускаем
        #     return
        
        if (v0.x < 0 and v1.x < 0 and v2.x < 0) or \
        (v0.x >= self.width and v1.x >= self.width and v2.x >= self.width) or \
        (v0.y < 0 and v1.y < 0 and v2.y < 0) or \
        (v0.y >= self.height and v1.y >= self.height and v2.y >= self.height):
            return

        # Находим bounding box
        min_x = max(0, int(min(v0.x, v1.x, v2.x)))
        max_x = min(self.width - 1, int(max(v0.x, v1.x, v2.x)))
        min_y = max(0, int(min(v0.y, v1.y, v2.y)))
        max_y = min(self.height - 1, int(max(v0.y, v1.y, v2.y)))
        
        if min_x > max_x or min_y > max_y:
            return
        
        # Преобразуем в 2D для барицентрических координат
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
                
                # Проверка z-буфера
                if z >= self.z_buffer[y, x]:
                    continue
                
                # Интерполяция текстурных координат
                interp_tex = Vec2(
                    t0.x * u + t1.x * v + t2.x * w,
                    t0.y * u + t1.y * v + t2.y * w
                )
                
                # Интерполяция нормали
                interp_normal = (n0 * u + n1 * v + n2 * w).normalize()
                
                # Интерполяция позиции в мировых координатах
                interp_world = world0 * u + world1 * v + world2 * w
                
                # Шейдинг с текстурой
                color = self.shader.fragment_shader(
                    interp_tex, interp_normal, interp_world, material, texture
                )
                
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
        
        MAX_TRIANGLES = 170000
        faces_to_render = model.faces[:MAX_TRIANGLES]
        faces_rendered = 0
    
        print(f"Рендеринг {len(faces_to_render)} треугольников (из {len(model.faces)})...")
        
        # Рендеринг каждого треугольника
        for face_idx, face in enumerate(faces_to_render):
            vertices_idx, texs_idx, normals_idx, material_name = face
            
            # Получаем материал для этой грани
            if material_name and material_name in model.materials:
                material = model.materials[material_name]
            else:
                material = model.current_material
            
            # Получаем текстуру из материала (если есть)
            texture = material.texture if hasattr(material, 'texture') else None
            
            triangle_verts = []
            triangle_texs = []
            triangle_norms = []
            triangle_world = []
            
            # Обрабатываем каждую вершину треугольника
            for i in range(3):
                # Получаем индекс вершины
                v_idx = vertices_idx[i]
                if v_idx < 0 or v_idx >= len(model.vertices):
                    print(f"Предупреждение: индекс вершины {v_idx} вне диапазона, используется 0")
                    v_idx = 0
                
                # Получаем вершину
                vert = model.vertices[v_idx]
                
                # Получаем текстурную координату
                t_idx = texs_idx[i]
                if t_idx >= 0 and t_idx < len(model.tex_coords):
                    tex = model.tex_coords[t_idx]
                else:
                    tex = Vec2(0.0, 0.0)  # координата по умолчанию
                
                # Получаем нормаль
                n_idx = normals_idx[i]
                if n_idx >= 0 and n_idx < len(model.normals):
                    norm = model.normals[n_idx]
                else:
                    # Если нормаль не указана, вычисляем приблизительную
                    norm = Vec3(0.0, 1.0, 0.0)
                
                # Применяем вершинный шейдер
                clip_pos, tex_coord, world_norm, world_pos = self.shader.vertex_shader(
                    vert, tex, norm, mvp_matrix, model_matrix, view_matrix
                )
                
                # Преобразование в экранные координаты
                screen_x = (clip_pos.x + 1.0) * 0.5 * self.width
                screen_y = (1.0 - clip_pos.y) * 0.5 * self.height
                screen_pos = Vec3(screen_x, screen_y, clip_pos.z)
                
                # Добавляем в списки для треугольника
                triangle_verts.append(screen_pos)
                triangle_texs.append(tex_coord)
                triangle_norms.append(world_norm)
                triangle_world.append(world_pos)
            
            # Проверяем, находится ли треугольник перед камерой (z > near plane)
            # Это помогает отсечь треугольники, которые находятся за камерой
            if all(v.z > camera.near for v in triangle_verts):
                # Растеризация треугольника
                self.rasterize_triangle(
                    triangle_verts[0], triangle_verts[1], triangle_verts[2],
                    triangle_texs[0], triangle_texs[1], triangle_texs[2],
                    triangle_norms[0], triangle_norms[1], triangle_norms[2],
                    triangle_world[0], triangle_world[1], triangle_world[2],
                    material, texture
                )
            
            # Вывод прогресса для больших моделей
            faces_rendered += 1
            if face_idx % 1000 == 0:
                print(f"  Обработано {face_idx}/{len(faces_to_render)} треугольников...")
        
        print(f"Рендеринг завершен! Только {faces_rendered} из первых {MAX_TRIANGLES} треугольников.")
    
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
                    f.write(struct.pack('BBB', r, g, b))

# На случай если всё плохо

def create_simple_cube():
    """Создание простой кубической модели для тестирования"""
    model = Model()
    
    # Вершины куба
    model.vertices = [
        Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
        Vec3(-1, -1, 1), Vec3(1, -1, 1), Vec3(1, 1, 1), Vec3(-1, 1, 1)
    ]
    
    # Текстурные координаты
    model.tex_coords = [
        Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)  # 4 координаты на куб
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
    
    # Грани куба с текстурными координатами
    # Каждой вершине соответствует текстурная координата (0-3)
    faces = [
        # Передняя грань
        ([0, 1, 2], [0, 1, 2], [0, 0, 0]),
        ([0, 2, 3], [0, 2, 3], [0, 0, 0]),
        # Задняя грань
        ([4, 6, 5], [0, 2, 1], [1, 1, 1]),
        ([4, 7, 6], [0, 3, 2], [1, 1, 1]),
        # Нижняя грань
        ([0, 4, 5], [0, 3, 1], [2, 2, 2]),
        ([0, 5, 1], [0, 1, 2], [2, 2, 2]),
        # Верхняя грань
        ([3, 2, 6], [0, 1, 2], [3, 3, 3]),
        ([3, 6, 7], [0, 2, 3], [3, 3, 3]),
        # Левая грань
        ([0, 3, 7], [0, 1, 2], [4, 4, 4]),
        ([0, 7, 4], [0, 2, 3], [4, 4, 4]),
        # Правая грань
        ([1, 5, 6], [0, 1, 2], [5, 5, 5]),
        ([1, 6, 2], [0, 2, 3], [5, 5, 5])
    ]
    
    model.faces = faces
    model.material = Material(
        ambient=Vec3(0.2, 0.2, 0.5),
        diffuse=Vec3(0.5, 0.5, 0.8),
        specular=Vec3(1.0, 1.0, 1.0),
        shininess=32.0
    )
    
    # Создаем тестовую текстуру (шахматную доску)
    model.texture = Texture()
    model.texture.create_default_texture()
    
    return model

def main():
    rasterizer = Rasterizer(800, 600)
    
    camera = Camera(
        position=Vec3(1, 2, 3), 
        target=Vec3(0, 1, 0),
        up=Vec3(0, 1, 0),
        fov=50.0,
        aspect_ratio=4/3,
        near=0.1,
        far=100.0
    )
    
    try:
        model = Model("monke.obj")
        model.compute_normals()
        model.compute_smooth_normals()
        print("Модель загружена из monke.obj")
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
