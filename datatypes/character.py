import numpy as np
from typing import List, Union
from .trimesh import Trimesh

class Character(Trimesh):
    def __init__(self, vertices=None, tris=None):
        super().__init__(vertices, tris)
        self._rest_pose_vertices: np.ndarray = np.array([], dtype=np.float64)
        self._rest_pose_vertices_normals: np.ndarray = np.array([], dtype=np.float64)
        self._rest_pose_triangles_normals: np.ndarray = np.array([], dtype=np.float64)

        if vertices is not None and tris is not None:
            self.create(vertices, tris)

    def create(self, vertices, tris):
        self.clear()
        super().create(vertices, tris)
        self._rest_pose_vertices = np.array(vertices, dtype=np.float64)
        self.init()
        return True

    def init(self):
        self.update_normals()
        if self._vertices_norm.size > 0:
            self._rest_pose_vertices_normals = self._vertices_norm.copy()
        if self._tris_norm.size > 0:
            self._rest_pose_triangles_normals = self._tris_norm.copy()

    def clear(self):
        self._rest_pose_vertices = np.array([], dtype=np.float64)
        self._rest_pose_vertices_normals = np.array([], dtype=np.float64)
        self._rest_pose_triangles_normals = np.array([], dtype=np.float64)
        super().clear()

    @property
    def rest_pose_vertices(self) -> np.ndarray:
        return self._rest_pose_vertices

    @rest_pose_vertices.setter
    def rest_pose_vertices(self, vertices: Union[List[float], np.ndarray]):
        self._rest_pose_vertices = np.array(vertices, dtype=np.float64)

    def get_rest_pose_vertex(self, v_id: int) -> np.ndarray:
        v_id_ptr = v_id * 3
        # NumPy 슬라이싱을 사용하면 더 효율적입니다.
        return self._rest_pose_vertices[v_id_ptr : v_id_ptr + 3]

    def set_rest_pose_vertex(self, v_id: int, new_position: np.ndarray):
        v_id_ptr = v_id * 3
        # NumPy 슬라이싱으로 한 번에 할당합니다.
        self._rest_pose_vertices[v_id_ptr : v_id_ptr + 3] = new_position

    @property
    def current_pose_vertices(self):
        return self.vertices

    @current_pose_vertices.setter
    def current_pose_vertices(self, vertices):
        self.vertices = vertices

    def get_actual_pose_vertex(self, v_id):
        return self.get_vertex(v_id)