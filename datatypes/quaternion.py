import numpy as np

class Quaternion:
    # --- Initializer ---
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        self._q = np.array([x,y,z,w], dtype=np.float64)

    def __repr__(self):
        # self._x 대신 self.x (property getter)를 사용합니다.
        return f"Quaternion(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, w={self.w:.4f})"
    
    @classmethod
    def from_axis_angle(cls, axis, angle):
        """ 
        축과 각도로부터 쿼터니온을 생성합니다 
        
        :param axis: The rotation axis (will be normalized internally).
        :param angle: The rotation angle in radians.
        """
        quat = cls()
        quat.set_rotation(axis, angle)
        return quat
    
    @classmethod
    def zero(cls):
        return Quaternion(0,0,0,0)

    # --- Property ---
    @property
    def x(self): return self._q[0]
    @x.setter
    def x(self, val): self._q[0] = val

    @property
    def y(self): return self._q[1]
    @y.setter
    def y(self, val): self._q[1] = val

    @property
    def z(self): return self._q[2]
    @z.setter
    def z(self, val): self._q[2] = val

    @property
    def w(self): return self._q[3]
    @w.setter
    def w(self, val): self._q[3] = val

    @property
    def v(self):
        """
        Quaternion은 scalar + vector의 형태로 표기됩니다 (s,v)
        이때 vector 부분만을 추출해서 리턴하는 함수입니다
        """
        return self._q[:3]

    # --- Methods ---
    def set_rotation(self, axis: np.ndarray, angle: float):
        # * The rotation axis must be a unit vector (magnitude of one) for the formula.
        # * Note : If you want to optimize, comment out and let your code put normalized axis 
        axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
        
        half_angle = angle / 2.0
        sin_half_angle = np.sin(half_angle)

        self.x=axis[0] * sin_half_angle
        self.y=axis[1] * sin_half_angle
        self.z=axis[2] * sin_half_angle
        self.w=np.cos(half_angle)

        # * Normalize the resulting quaternion to counteract floating-point errors.
        self.normalize()

    def length_squared(self):
        """크기의 제곱(노름)을 반환합니다."""
        return np.dot(self._q, self._q)
    # alias
    norm = length_squared

    def length(self):
        return np.linalg.norm(self._q)
    # alias 
    magnitude = length

    def normalize(self, eps=1e-10):
        length = self.length()
        
        # * To prevent Floating Point Error (NaN)
        # * Ignore if length is smaller than eps
        if length > eps:
            self._q /= length
        
        return length

    def dot(self, other):
        """다른 쿼터니언과의 내적을 계산합니다. (NumPy dot product 사용)"""
        return np.dot(self._q, other._q)

    def conjugate(self):
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def inverse(self):
        """inv_q = conjugate_q / length_sq"""
        len_sqr = self.length_squared()
    
        # * To Prevent Floating Point Error
        if len_sqr < 1e-10:
            return Quaternion.zero()
        
        inv_len = 1.0 / len_sqr
        conjuate = self.conjugate()
        return conjuate * inv_len

    def apply_rotation(self, v):
        """
        Rotate vector by Quaternion
        q * v_q * q_conj
        """
        v_q = Quaternion(*np.r_[v, [0.0]])
        
        result = self * v_q * self.conjugate()
        return result.v
    
    def to_euler(self, is_degree: bool = True) -> np.ndarray:
        """
        쿼터니언을 오일러 각(Roll, Pitch, Yaw)으로 변환합니다. (ZYX 순서)
        NumPy 배열 형태로 압축된 최적화된 구현입니다.
        
        :param is_degree: True이면 각도를 도(degree)로 반환합니다.
        :return: 3-요소 numpy 배열 [Roll, Pitch, Yaw].
        """
        x, y, z, w = self.x, self.y, self.z, self.w
        
        # Pitch 계산을 위한 sinp
        sinp = 2 * (w * y - z * x)

        angles = np.array([
            # Roll (x-축): atan2(sinr_cosp, cosr_cosp)
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2)),

            # Pitch (y-축): Gimbal Lock 처리 (수정됨: sinp의 부호를 따라야 함)
            np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp),
            
            # Yaw (z-축): atan2(siny_cosp, cosy_cosp)
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)),
        ], dtype=np.float64)

        if is_degree:
            return angles * (180.0 / np.pi)
        
        return angles
    
    # Operator Overload
    def __add__(self, other):
        """쿼터니언 덧셈."""
        if isinstance(other, Quaternion):
            return Quaternion(*(self._q + other._q))
        else:
            return NotImplemented
    
    def __mul__(self, other):
        """
        쿼터니언 곱셈 (q * p) 또는 스칼라 곱셈 (q * s).
        """
        if isinstance(other, Quaternion):
            # Hamilton Product
            # self.w와 self.v 프로퍼티를 사용합니다.
            s1, v1 = self.w, self.v
            s2, v2 = other.w, other.v

            s_new = (s1 * s2) - np.dot(v1, v2)
            v_new = (s1 * v2) + (s2 * v1) + np.cross(v1, v2)
            
            return Quaternion(v_new[0], v_new[1], v_new[2], s_new)
        
        elif isinstance(other, (int, float, np.floating)):
            # 쿼터니언 * 스칼라: NumPy 배열 연산으로 깔끔하게 처리
            return Quaternion(*(self._q * other))
        
        else:
            return NotImplemented

    def __rmul__(self, other):
        """스칼라 곱셈 (s * q)."""
        if isinstance(other, (int, float, np.floating)):
            # 스칼라 * 쿼터니언
            return self * other
        else:
            return NotImplemented

    # TODO : Move below functions to appropriate places
    def to_rotation_matrix(self) -> np.ndarray:
        raise DeprecationWarning("Do not use rotation matrix. use operator")
        pass 
        xx, yy, zz = self.x * self.x, self.y * self.y, self.z * self.z
        xy, xz, yz = self.x * self.y, self.x * self.z, self.y * self.z
        wx, wy, wz = self.w * self.x, self.w * self.y, self.w * self.z

        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
            [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)]
        ])

    def slerp(self, other: 'Quaternion', t: float) -> 'Quaternion':
        dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

        if dot < 0.0:
            other.x = -other.x
            other.y = -other.y
            other.z = -other.z
            other.w = -other.w
            dot = -dot

        if dot > 0.9995:
            result = Quaternion(
                x=self.x + t * (other.x - self.x),
                y=self.y + t * (other.y - self.y),
                z=self.z + t * (other.z - self.z),
                w=self.w + t * (other.w - self.w),
            )
            return result

        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return Quaternion(
            x=(s0 * self.x) + (s1 * other.x),
            y=(s0 * self.y) + (s1 * other.y),
            z=(s0 * self.z) + (s1 * other.z),
            w=(s0 * self.w) + (s1 * other.w),
        )

