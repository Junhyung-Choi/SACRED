import numpy as np

from typing import List, Union

from ..datatypes.weights import Weights
from ..datatypes.character import Character
from ..datatypes.cage import Cage

class CageUpdater:
    """
    Cage Updater (V' -> C')

    변형된 캐릭터 메시(V')를 기반으로, 이를 가장 잘 표현하는 케이지(C')의
    새로운 위치를 최소제곱법(Least-Squares)으로 역산합니다.

    C++ 원본: `SuperCages/operators/cageUpdater.cpp`
    """
    def __init__(self):
        self.w_cage: Union[Weights, None] = None
        self.character: Union[Character , None] = None
        self.cage: Union[Cage,None] = None
        self.selected_vertices: Union[List[int] ,None] = None
        self.clear()

    def create(
        self,
        w_cage: Weights,
        character: Character,
        cage: Cage,
        selected_vertices: Union[List[int] , None] = None
    ) -> bool:
        """
        CageUpdater를 초기화합니다.

        Args:
            w_cage (Weights): 케이지 가중치 (캐릭터 정점을 케이지 정점의 선형 결합으로 표현).
            character (Character): 캐릭터 객체.
            cage (Cage): 케이지 객체.
            selected_vertices (List[int] | None): 계산에 사용할 특정 캐릭터 정점의 인덱스 리스트.
                                                   None이면 모든 정점을 사용합니다.
        """
        self.clear()

        self.w_cage = w_cage
        self.character = character
        self.cage = cage
        self.selected_vertices = selected_vertices

        return True

    def clear(self):
        self.w_cage = None
        self.character = None
        self.cage = None
        self.selected_vertices = None

    def update_position(self):
        """
        변형된 캐릭터 정점 위치를 기반으로 케이지 정점의 새 위치를 계산합니다.
        선형 시스템 `W_cage * C' = V'`를 최소제곱법으로 풀어 C'를 구합니다.
        """
        if not all([self.character, self.cage, self.w_cage]):
            return

        num_cage_vertices = self.cage.num_vertices
        
        # W_cage: (num_cage_vertices, num_char_vertices)
        # NumPy lstsq는 (M, N) 형태의 행렬을 기대하므로 전치(transpose)
        # A (M, N) @ x (N) = b (M)
        # W_cage.T (num_char_vertices, num_cage_vertices) @ C' (num_cage_vertices) = V' (num_char_vertices)
        W_T = self.w_cage.weights.T

        # 1. 변형된 캐릭터 정점(V') 가져오기
        V_prime = self.character.current_pose_vertices.reshape(-1, 3)

        # 2. 선택된 정점이 있는 경우, 가중치 행렬과 정점 벡터를 필터링
        if self.selected_vertices and len(self.selected_vertices) > 0:
            W_T = W_T[self.selected_vertices, :]
            V_prime = V_prime[self.selected_vertices, :]

        # 3. 각 좌표(x, y, z)에 대해 최소제곱법 시스템 풀기
        new_cage_x = np.linalg.lstsq(W_T, V_prime[:, 0], rcond=None)[0]
        new_cage_y = np.linalg.lstsq(W_T, V_prime[:, 1], rcond=None)[0]
        new_cage_z = np.linalg.lstsq(W_T, V_prime[:, 2], rcond=None)[0]

        # 4. 계산된 좌표를 합쳐 새로운 케이지 정점 위치(C') 생성
        C_prime = np.stack([new_cage_x, new_cage_y, new_cage_z], axis=1)

        # 5. 케이지의 현재 포즈 업데이트
        self.cage.current_pose_vertices = C_prime.flatten()
