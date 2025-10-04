import numpy as np
from scipy.spatial.transform import Rotation as R
from .quaternion import Quaternion

class Transform:
    def __init__(self, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0, rx: float = 0.0, ry: float = 0.0, rz: float = 0.0, mat: list[float] = None, quat: Quaternion = None, col1: np.ndarray = None, col2: np.ndarray = None, col3: np.ndarray = None):
        if mat is not None:
            self.matrix = np.array(mat).reshape((4, 4))
        elif quat is not None:
            self.matrix = np.identity(4)
            self.matrix[:3, :3] = quat.to_rotation_matrix()
        elif col1 is not None and col2 is not None and col3 is not None:
            self.matrix = np.identity(4)
            self.matrix[:3, 0] = col1
            self.matrix[:3, 1] = col2
            self.matrix[:3, 2] = col3
        else:
            # Initalize Transform Matrix from Euler angles and translation
            self.matrix = np.identity(4)

            # Calculate Rotation Matrix
            if rx != 0.0 or ry != 0.0 or rz != 0.0:
                grad_to_rad = np.pi / 180.0
                q_x = Quaternion.from_axis_angle(np.array([1, 0, 0]), rx * grad_to_rad)
                q_y = Quaternion.from_axis_angle(np.array([0, 1, 0]), ry * grad_to_rad)
                q_z = Quaternion.from_axis_angle(np.array([0, 0, 1]), rz * grad_to_rad)

                # Rotation Order : X -> Y -> Z
                q_rotation = q_z * q_y * q_x 
                self.matrix[:3, :3] = q_rotation.to_rotation_matrix()
            
            # Add Translation
            self.matrix[0, 3] = tx
            self.matrix[1, 3] = ty
            self.matrix[2, 3] = tz

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3,3]

    def get_rotation(self) -> Quaternion:
        """4x4 변환 행렬에서 3x3 회전 행렬을 추출하여 Quaternion으로 변환합니다."""
        
        # 3x3 회전 행렬 추출
        R_matrix = self.matrix[:3, :3]
        
        # SciPy를 사용하여 행렬에서 쿼터니언 성분 [x, y, z, w] 추출
        # SciPy는 쿼터니언을 [x, y, z, w] 순서로 반환합니다.
        quat_xyzw = R.from_matrix(R_matrix).as_quat()
        
        # 직접 만든 Quaternion 객체로 변환하여 반환
        return Quaternion(quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3])

    def set_to_zero(self):
        self.matrix = np.zeros((4, 4))

    def set_translation(self, t: np.ndarray):
        self.matrix[:3, 3] = t

    def copy(self) -> 'Transform':
        return Transform(mat=self.matrix.copy().flatten().tolist())

    def cumulate_with(self, t: 'Transform') -> 'Transform':
        return self * t

    def inverse(self) -> 'Transform':
        return Transform(mat=np.linalg.inv(self.matrix).flatten().tolist())

    def apply_to_point(self, p: np.ndarray) -> np.ndarray:
        p_homogeneous = np.append(p, 1)
        p_transformed = np.dot(self.matrix, p_homogeneous)
        return p_transformed[:3]

    def transform_towards_origin(self, t: 'Transform') -> 'Transform':
        """
        C++의 transformTowardsOrigin 로직을 복제합니다.
        현재 변환(self)의 위치를 피벗으로 사용하여, 새로운 변환(t)을 적용합니다.
        
        로직 순서:
        1. 현재 이동 성분 (T_current)을 백업합니다.
        2. 현재 변환 행렬에서 이동 성분을 0으로 만듭니다 (순수 회전 R_current만 남김).
        3. 새로운 변환 행렬 (t)을 R_current에 왼쪽에서 곱합니다: t * R_current
        4. 백업했던 T_current를 결과 행렬에 다시 더합니다.
        """
        # 1. 현재 변환 행렬을 복사하여 새로운 Transform 객체 생성
        result_transform = self.copy()

        # 2. 현재 이동 성분 (tx, ty, tz) 백업
        # self.matrix[:3, 3]는 translation 벡터입니다.
        vec_t = self.matrix[:3, 3].copy() 

        # 3. 결과 행렬에서 이동 성분을 0으로 설정 (순수 회전 행렬만 남김)
        result_transform.matrix[:3, 3] = 0

        # 4. 새로운 변환 t를 현재 회전 행렬에 왼쪽에서 곱함
        # C++: thisTransform.transformation = t.transformation * thisTransform.transformation;
        result_transform.matrix = np.dot(t.matrix, result_transform.matrix)

        # 5. 백업했던 이동 성분을 결과 행렬의 이동 성분 위치에 다시 더함
        # C++: thisTransform.transformation(x,3) += vecT[x];
        result_transform.matrix[:3, 3] += vec_t

        return result_transform

    def data(self) -> list[float]:
        return self.matrix.flatten().tolist()

    def __add__(self, other: 'Transform') -> 'Transform':
        return Transform(mat=(self.matrix + other.matrix).flatten().tolist())

    def __mul__(self, other: 'Transform') -> 'Transform':
        return Transform(mat=np.dot(self.matrix, other.matrix).flatten().tolist())

    def interpolate(self, other: 'Transform', a: float) -> 'Transform':
        t1 = self.get_translation()
        t2 = other.get_translation()
        t = (t1 * (1 - a)) + (t2 * a)

        q1 = self.get_rotation()
        q2 = other.get_rotation()
        q = q1.slerp(q2, a)

        new_transform = Transform(quat=q)
        new_transform.set_translation(t)
        return new_transform