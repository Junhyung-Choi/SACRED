import numpy as np
from .bbox import BoundingBox
from typing import List, Tuple, Set, Dict

# edge 구조체 대체 (2-요소 튜플로 정의)
Edge = Tuple[int, int]

# C++의 uniqueEdge() 함수 대체
def unique_edge(v1: int, v2: int) -> Edge:
    return tuple(sorted((v1, v2)))

class Trimesh:
    def __init__(self, _vertices: List[float] = None, _tris: List[int] = None):
        # C++ protected 멤버 변수
        self._vertices: np.ndarray = np.array([], dtype=np.float64)  # vertices
        self._tris: np.ndarray = np.array([], dtype=np.int32)        # tris
        self._vertices_norm: np.ndarray = np.array([], dtype=np.float64) # verticesNorm
        self._tris_norm: np.ndarray = np.array([], dtype=np.float64)     # trisNorm
        
        # 인접 정보 (C++: std::vector<std::vector<int>>)
        self._v2t: List[List[int]] = [] # _v2t (Vertex to Triangles)
        self._v2v: List[List[int]] = [] # _v2v (Vertex to Vertices)
        self._t2t: List[List[int]] = [] # _t2t (Triangle to Triangles)
        
        self.bounding_box: BoundingBox = BoundingBox() # boundingBox

        if _vertices is not None and _tris is not None:
            self.create(_vertices, _tris)
            
    # ===============================================
    # C++ PUBLIC Core Methods
    # ===============================================

    def clear(self):
        """Trimesh의 모든 내부 데이터를 비웁니다."""
        self._vertices = np.array([], dtype=np.float64)
        self._tris = np.array([], dtype=np.int32)
        self._vertices_norm = np.array([], dtype=np.float64)
        self._tris_norm = np.array([], dtype=np.float64)
        self._v2t.clear()
        self._v2v.clear()
        self._t2t.clear()
        self.bounding_box.clear()

    def create(self, _vertices: List[float], _tris: List[int]) -> bool:
        self.clear()
        
        self._vertices = np.array(_vertices, dtype=np.float64)
        self._tris = np.array(_tris, dtype=np.int32)
        
        self._init()
        
        return True

    # --- Accessors (Getters/Setters) ---
    
    # Non-const/Const Getter 통합 (Python @property)
    @property
    def vertices(self) -> np.ndarray:
        """getVerticesVector()"""
        return self._vertices
    
    @vertices.setter
    def vertices(self, _vertices: List[float]):
        """setVerticesVector()"""
        # C++: setVerticesVector(const std::vector<double> & _vertices)
        self._vertices = np.array(_vertices, dtype=np.float64)
        # TODO: C++ 주석에 따라, set 후 init() 호출 여부를 결정해야 함.
        
    @property
    def triangles(self) -> np.ndarray:
        """getTrianglesVector()"""
        return self._tris
    
    @triangles.setter
    def triangles(self, _tris: List[int]):
        """setTrianglesVector()"""
        # C++: setTrianglesVector(const std::vector<int> & _tris)
        self._tris = np.array(_tris, dtype=np.int32)
        # TODO: C++ 주석에 따라, set 후 init() 호출 여부를 결정해야 함.

    @property
    def num_vertices(self) -> int:
        """getNumVertices()"""
        return len(self._vertices) // 3
    
    @property
    def num_triangles(self) -> int:
        """getNumTriangles()"""
        return len(self._tris) // 3

    def v2t(self, v_id: int) -> List[int]:
        """v2t(unsigned long vId)"""
        return self._v2t[v_id]
        
    def v2v(self, v_id: int) -> List[int]:
        """v2v(unsigned long vId)"""
        return self._v2v[v_id]
        
    def t2t(self, t_id: int) -> List[int]:
        """t2t(unsigned long tId)"""
        return self._t2t[t_id]
        
    def get_bounding_box(self) -> BoundingBox:
        """getBoundingBox()"""
        return self.bounding_box

    # --- Vertex/Normal Accessors ---
    
    def get_vertex(self, v_id: int) -> np.ndarray:
        """getVertex(unsigned long vId) -> cg3::Vec3<double>"""
        v_id_ptr = v_id * 3
        # CHECK_BOUNDS는 생략하고 3D 벡터(x, y, z)를 반환
        return self._vertices[v_id_ptr : v_id_ptr + 3]

    def set_vertex(self, v_id: int, new_position: np.ndarray):
        """setVertex(unsigned long vId, cg3::Vec3d newPosition)"""
        v_id_ptr = v_id * 3
        # NumPy를 사용하여 3개 좌표를 한 번에 설정
        self._vertices[v_id_ptr : v_id_ptr + 3] = new_position
        
    def get_triangle_normal(self, t_id: int) -> np.ndarray:
        """getTriangleNormal(unsigned long tId)"""
        t_id_ptr = t_id * 3
        return self._tris_norm[t_id_ptr : t_id_ptr + 3]

    def get_vertex_normal(self, v_id: int) -> np.ndarray:
        """getVertexNormal(unsigned long vId)"""
        v_id_ptr = v_id * 3
        return self._vertices_norm[v_id_ptr : v_id_ptr + 3]
    
    # --- Update Methods ---
    
    def update_normals(self):
        """updateNormals()"""
        self._update_tris_normals()
        self._update_vertices_normals()
        
    def update_bounding_box(self):
        """updateBoundingBox()"""
        self.bounding_box.clear()
        
        # getVertex()를 호출하는 C++ 로직을 NumPy 슬라이싱으로 대체
        for v_id in range(self.num_vertices):
            v = self.get_vertex(v_id) # 3-요소 벡터
            
            # C++: boundingBox.min = boundingBox.min.min(v);
            self.bounding_box.min = np.minimum(self.bounding_box.min, v)
            self.bounding_box.max = np.maximum(self.bounding_box.max, v)

    # ===============================================
    # C++ PROTECTED / PRIVATE Implementation
    # ===============================================
    
    def _init(self):
        """init()"""
        # C++: trisNorm.resize(...) / verticesNorm.resize(...)
        self._tris_norm = np.zeros(self.num_triangles * 3, dtype=np.float64)
        self._vertices_norm = np.zeros(self.num_vertices * 3, dtype=np.float64)
        
        self._build_adjacency()
        self.update_normals()
        self.update_bounding_box()

    def _build_adjacency(self):
        """buildAdjacency() - C++ 원본 로직 유지"""
        self._v2v.clear()
        self._v2t.clear()
        self._t2t.clear()

        self._v2v.extend([[] for _ in range(self.num_vertices)])
        self._v2t.extend([[] for _ in range(self.num_vertices)])
        self._t2t.extend([[] for _ in range(self.num_triangles)])

        edges: Set[Edge] = set()
        edge2tri: Dict[Edge, int] = {}

        for t_id in range(self.num_triangles):
            t_id_ptr = t_id * 3
            for i in range(3):
                v_id = self._tris[t_id_ptr + i]
                self._v2t[v_id].append(t_id)

                adj_id = self._tris[t_id_ptr + (i + 1) % 3]
                e = unique_edge(int(v_id), int(adj_id))
                edges.add(e)

                if e not in edge2tri:
                    edge2tri[e] = t_id
                else:
                    nbr_tri = edge2tri[e]
                    self._t2t[t_id].append(nbr_tri)
                    self._t2t[nbr_tri].append(t_id)

        # C++ OLD loop (V2V for non-boundary edges)
        for e in edges:
            self._v2v[e[0]].append(e[1])
            self._v2v[e[1]].append(e[0])

    def _update_tris_normals(self):
        """updateTrisNormals() - NumPy 벡터화 연산 대신 C++의 OMP 루프 로직을 유지"""
        
        # C++ 코드는 OMP를 사용했으므로, Python에서 병렬 처리를 생략하고 NumPy 슬라이싱을 사용해 효율을 높입니다.
        
        num_triangles = self.num_triangles
        if num_triangles == 0:
            return

        # V0, V1, V2 인덱스를 미리 추출
        v_indices = self._tris.reshape((-1, 3)) # (num_triangles, 3)
        
        # 정점 좌표를 추출하여 (N, 3) 행렬로 재구성
        vertices_coords = self._vertices.reshape((-1, 3))
        
        # V0, V1, V2 정점 추출 (Fancy Indexing)
        v0 = vertices_coords[v_indices[:, 0]]
        v1 = vertices_coords[v_indices[:, 1]]
        v2 = vertices_coords[v_indices[:, 2]]

        # 엣지 벡터 계산 (u = V1 - V0, v = V2 - V0)
        u = v1 - v0
        v = v2 - v0

        # 외적 (Cross Product) 계산: n = u x v
        normals = np.cross(u, v)

        # 길이 (Length) 계산: lengthN
        length_n = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # 정규화 (Normalization): nxN = nx / lengthN
        # 길이가 0인 경우(퇴화 삼각형)를 처리하기 위해 나누기 전에 마스크 적용
        # length_n이 0인 경우 1로 치환하여 NaN 방지
        safe_length_n = np.where(length_n == 0, 1.0, length_n)
        
        normalized_normals = normals / safe_length_n
        
        # 결과를 trisNorm에 저장 (1차원 배열로 평탄화)
        self._tris_norm = normalized_normals.flatten()
        
    def _update_vertices_normals(self):
        """updateVerticesNormals() - C++의 루프 기반 로직을 유지"""
        
        num_vertices = self.num_vertices
        if num_vertices == 0:
            return
        
        self._vertices_norm = np.zeros(num_vertices * 3, dtype=np.float64)
        
        # C++의 for 루프 로직을 유지하면서 NumPy를 사용하여 연산 속도를 높임
        for v_id in range(num_vertices):
            # v2t(vId)로 인접 삼각형 인덱스 가져오기
            neigh_t_ids = self.v2t(v_id) 

            if not neigh_t_ids:
                # 인접 삼각형이 없으면 건너뜁니다.
                continue

            # 인접 삼각형의 노멀을 추출 (Fancy Indexing)
            # 노멀은 self._tris_norm에 3개 좌표 단위로 저장되어 있음
            neigh_norm_indices = np.array(neigh_t_ids) * 3
            
            # 인접 노멀 벡터 (N, 3)
            # 여기서는 C++의 sumx, sumy, sumz 계산을 위해 배열 슬라이싱을 사용해야 함
            
            sum_norm = np.zeros(3)
            for t_id in neigh_t_ids:
                t_id_ptr = t_id * 3
                sum_norm += self._tris_norm[t_id_ptr : t_id_ptr + 3]
            
            # 평균 계산
            sum_norm /= len(neigh_t_ids)
            
            # 정규화 (Normalization)
            length_sum = np.linalg.norm(sum_norm)
            
            if length_sum > 1e-6: # 0으로 나누는 것을 방지
                normalized_sum = sum_norm / length_sum
            else:
                normalized_sum = np.array([0.0, 0.0, 0.0]) # 0 벡터

            # 결과를 verticesNorm에 저장
            v_id_ptr = v_id * 3
            self._vertices_norm[v_id_ptr : v_id_ptr + 3] = normalized_sum

    # --- Copy method (for Cage) ---
    def copy(self) -> 'Trimesh':
        """
        Cage 클래스에서 요구하는 깊은 복사를 수행합니다.
        """
        new_mesh = Trimesh()
        new_mesh._vertices = self._vertices.copy()
        new_mesh._tris = self._tris.copy()
        new_mesh._vertices_norm = self._vertices_norm.copy()
        new_mesh._tris_norm = self._tris_norm.copy()
        
        # 인접 정보도 깊은 복사
        new_mesh._v2t = [list(l) for l in self._v2t]
        new_mesh._v2v = [list(l) for l in self._v2v]
        new_mesh._t2t = [list(l) for l in self._t2t]
        
        # BoundingBox 복사 (BoundingBox에 copy()가 있다면 더 좋음, 없으면 재계산 필요)
        new_mesh.bounding_box = BoundingBox() 
        new_mesh.bounding_box.min = self.bounding_box.min.copy()
        new_mesh.bounding_box.max = self.bounding_box.max.copy()

        return new_mesh