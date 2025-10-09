import numpy as np
from typing import List, Union
from .trimesh import Trimesh

class Cage:
    def __init__(self, vertices: List[float] = None, tris: List[int] = None):
        # C++ 원본의 protected 멤버 변수들
        self._original_rest_pose: Trimesh = Trimesh()
        self._rest_pose: Trimesh = Trimesh()
        self._current_pose: Trimesh = Trimesh()
        
        # lastTranslations는 C++ 구현에 따라 빈 리스트로 초기화 후 init()에서 크기 조정
        self._last_translations: np.ndarray = np.array([], dtype=np.float64)
        
        if vertices is not None and tris is not None:
            self.create(vertices, tris)

    def create(self, vertices: List[float], tris: List[int]) -> bool:
        self.clear()
        
        self._original_rest_pose.create(vertices, tris)
        
        # C++의 'restPose = originalRestPose' (값 복사) 의도를 반영
        # Trimesh에 .copy()가 있다고 가정하고 안전하게 깊은 복사
        self._rest_pose = self._original_rest_pose.copy()
        self._current_pose = self._original_rest_pose.copy()

        self.init() 

        return True

    def init(self):
        """lastTranslations 벡터를 초기화하고 0으로 채웁니다."""
        # C++: lastTranslations.resize(originalRestPose.getNumVertices()*3, 0.0);
        num_coords = self.num_vertices * 3
        self._last_translations = np.zeros(num_coords, dtype=np.float64)

    def clear(self):
        """모든 내부 데이터와 Trimesh 객체들을 비웁니다."""
        # C++ 원본 구현과 동일하게 clear() 호출
        self._original_rest_pose.clear()
        self._rest_pose.clear()
        self._current_pose.clear()
        self._last_translations = np.array([], dtype=np.float64)
        # NOTE: C++ 원본은 clear() 후 Trimesh 객체 자체를 재할당하지 않고 내부 데이터만 비웁니다.

    def on_current_pose_vertices_updated(self):
        pass

    # --- Accessors (C++ Getters/Setters) ---
    
    @property
    def num_vertices(self) -> int:
        return self._original_rest_pose.num_vertices

    @property
    def num_triangles(self) -> int:
        return self._original_rest_pose.num_triangles

    # Current Pose
    @property
    def current_pose_vertices(self) -> List[float]:
        return self._current_pose.vertices
    @current_pose_vertices.setter
    def current_pose_vertices(self, vertices: List[float]):
        self._current_pose.vertices = vertices
        self.on_current_pose_vertices_updated()

    def get_current_pose_vertex(self, v_id: int) -> np.ndarray:
        return self._current_pose.get_vertex(v_id)

    def set_current_pose_vertex(self, v_id: int, new_position: np.ndarray):
        self._current_pose.set_vertex(v_id, new_position)

    # Rest Pose
    @property
    def rest_pose_vertices(self) -> List[float]:
        return self._rest_pose.vertices
    @rest_pose_vertices.setter
    def rest_pose_vertices(self, vertices: List[float]):
        self._rest_pose.vertices = vertices

    def get_rest_pose_vertex(self, v_id: int) -> np.ndarray:
        return self._rest_pose.get_vertex(v_id)

    def set_rest_pose_vertex(self, v_id: int, new_position: np.ndarray):
        self._rest_pose.set_vertex(v_id, new_position)

    # Original Rest Pose
    @property
    def original_rest_pose_vertices(self) -> List[float]:
        return self._original_rest_pose.vertices

    @property
    def original_rest_pose_triangles(self) -> List[int]:
        #! C++ 원본은 currentPose의 트라이앵글을 반환했지만, 
        #! 이는 원본 포즈의 트라이앵글을 반환하는 것이 논리적이므로 수정합니다.
        return self._original_rest_pose.triangles
    
    @property
    def current_pose_triangles(self) -> List[int]:
        """
        NOTE: original_rest_pose_triangles를 사용하는 구간인
              MVC 코드가 의도한 대로 동작하지 않는다면 이 프로퍼티를 대신해서
              사용
        """
        return self._current_pose.triangles


    @property
    def last_translations(self) -> np.ndarray:
        return self._last_translations

    def set_keyframe(self, keyframe: List[float]):
        """
        C++ 로직: keyframe(변위 벡터)을 Original Rest Pose에 더하여 Rest Pose를 설정합니다.
        """
        original_vertices = np.array(self.original_rest_pose_vertices)
        keyframe_np = np.array(keyframe)
        
        # Rest Pose = Original Rest Pose + Keyframe (Displacement)
        new_rest_vertices = original_vertices + keyframe_np
        
        self.rest_pose_vertices = new_rest_vertices.tolist()


    def interpolate_keyframes(self, keyframe_low: List[float], keyframe_top: List[float], a: float):
        """
        C++ 로직: 두 키프레임(변위 벡터)을 보간하여 Original Rest Pose에 더해 Rest Pose를 설정합니다.
        """
        original_vertices = np.array(self.original_rest_pose_vertices)
        keyframe_low_np = np.array(keyframe_low)
        keyframe_top_np = np.array(keyframe_top)
        
        # Interpolated Keyframe = (Low * (1.0-a)) + (Top * a)
        interpolated_keyframe = (keyframe_low_np * (1.0 - a)) + (keyframe_top_np * a)
        
        # Rest Pose = Original Rest Pose + Interpolated Keyframe
        self.rest_pose_vertices = (original_vertices + interpolated_keyframe).tolist()