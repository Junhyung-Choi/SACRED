import numpy as np
from typing import Union
from .trimesh import Trimesh

class Cage:
    def __init__(self, vertices: np.ndarray = None, tris: np.ndarray = None, render_object = None):
        # C++ 원본의 protected 멤버 변수들
        self._original_rest_pose: Trimesh = Trimesh()
        self._rest_pose: Trimesh = Trimesh()
        self._current_pose: Trimesh = Trimesh()
        self._render_object = render_object
        
        # lastTranslations는 C++ 구현에 따라 빈 리스트로 초기화 후 init()에서 크기 조정
        self._last_translations: np.ndarray = np.array([], dtype=np.float64)
        
        if vertices is not None and tris is not None:
            self.create(vertices, tris)

    def create(self, vertices: np.ndarray, tris: np.ndarray) -> bool:
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
        tmp = self.current_pose_vertices
        # tmp = tmp.reshape(-1,3)
        # tmp[:,2] *= -1.0 # Z축 뒤집기
        # tmp = tmp.flatten()
        self._render_object._vao.set_vertex_position_batch(tmp)
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
    def current_pose_vertices(self) -> np.ndarray:
        return self._current_pose.vertices
    @current_pose_vertices.setter
    def current_pose_vertices(self, vertices: np.ndarray):
        self._current_pose.vertices = vertices
        self.on_current_pose_vertices_updated()

    def get_current_pose_vertex(self, v_id: int) -> np.ndarray:
        return self._current_pose.get_vertex(v_id)

    def set_current_pose_vertex(self, v_id: int, new_position: np.ndarray):
        # 1. 정점이 실제로 업데이트되기 전의 '이전' 위치를 가져옵니다.
        old_position = self._current_pose.get_vertex(v_id)
        
        # 2. 이 편집으로 인한 변위(delta)를 계산합니다.
        delta = new_position - old_position
        
        # 3. 이 delta를 _last_translations 배열의 올바른 위치에 누적합니다.
        #    (v_id*3)부터 3개의 요소(dx, dy, dz)에 덮어쓰는 것이 아니라 더합니다(+=).
        self._last_translations[v_id * 3 : v_id * 3 + 3] += delta
        
        # 4. 이제 _current_pose의 정점 위치를 실제로 업데이트합니다.
        self._current_pose.set_vertex(v_id, new_position)

    # Rest Pose
    @property
    def rest_pose_vertices(self) -> np.ndarray:
        return self._rest_pose.vertices
    @rest_pose_vertices.setter
    def rest_pose_vertices(self, vertices: np.ndarray):
        self._rest_pose.vertices = vertices

    def get_rest_pose_vertex(self, v_id: int) -> np.ndarray:
        return self._rest_pose.get_vertex(v_id)

    def set_rest_pose_vertex(self, v_id: int, new_position: np.ndarray):
        self._rest_pose.set_vertex(v_id, new_position)

    # Original Rest Pose
    @property
    def original_rest_pose_vertices(self) -> np.ndarray:
        return self._original_rest_pose.vertices

    @property
    def original_rest_pose_triangles(self) -> np.ndarray:
        #! C++ 원본은 currentPose의 트라이앵글을 반환했지만, 
        #! 이는 원본 포즈의 트라이앵글을 반환하는 것이 논리적이므로 수정합니다.
        return self._original_rest_pose.triangles
    
    @property
    def current_pose_triangles(self) -> np.ndarray:
        """
        NOTE: original_rest_pose_triangles를 사용하는 구간인
              MVC 코드가 의도한 대로 동작하지 않는다면 이 프로퍼티를 대신해서
              사용
        """
        return self._current_pose.triangles


    @property
    def last_translations(self) -> np.ndarray:
        return self._last_translations
    @last_translations.setter
    def last_translations(self, val):
        self._last_translations = val

    def set_keyframe(self, keyframe: np.ndarray):
        """
        C++ 로직: keyframe(변위 벡터)을 Original Rest Pose에 더하여 Rest Pose를 설정합니다.
        """
        original_vertices = self.original_rest_pose_vertices
        
        # Rest Pose = Original Rest Pose + Keyframe (Displacement)
        new_rest_vertices = original_vertices + keyframe
        
        self.rest_pose_vertices = new_rest_vertices


    def interpolate_keyframes(self, keyframe_low: np.ndarray, keyframe_top: np.ndarray, a: float):
        """
        C++ 로직: 두 키프레임(변위 벡터)을 보간하여 Original Rest Pose에 더해 Rest Pose를 설정합니다.
        """
        original_vertices = self.original_rest_pose_vertices
        
        # Interpolated Keyframe = (Low * (1.0-a)) + (Top * a)
        interpolated_keyframe = (keyframe_low * (1.0 - a)) + (keyframe_top * a)
        
        # Rest Pose = Original Rest Pose + Interpolated Keyframe
        self.rest_pose_vertices = original_vertices + interpolated_keyframe