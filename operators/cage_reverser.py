import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, eye, kron
from scipy.sparse.linalg import inv, lsqr
from typing import List, Union

from ..datatypes.character import Character
from ..datatypes.cage import Cage
from ..datatypes.skeleton import Skeleton
from ..datatypes.weights import Weights

class CageReverser:
    """
    Cage Reverser (C' -> C)

    케이지의 현재 포즈(C')에 적용된 편집을 휴식 포즈(C)에 역으로 전파합니다.
    논문의 핵심 로직으로, 거대한 선형 시스템을 풀어 휴식 포즈의 변위를 계산합니다.

    C++ 원본: `SuperCages/operators/cageReverser.cpp`
    """
    def __init__(self):
        self.character: Union[Character, None] = None
        self.cage: Union[Cage, None] = None
        self.skel: Union[Skeleton, None] = None
        self.psi: Union[Weights, None] = None   # ψ (skeleton updater weights)
        self.phi: Union[Weights, None] = None   # φ (cage weights)
        self.omega: Union[Weights, None] = None # ω (skeleton weights)
        self.selected_vertices_for_inversion: Union[List[int], None] = None
        self.refresh_matrices: bool = True

        # Sparse Matrices (SciPy)
        self.PSI: Union[csc_matrix, None] = None
        self.PHI: Union[csc_matrix, None] = None
        self.PHI_transpose: Union[csc_matrix, None] = None
        self.OMEGA: Union[csc_matrix, None] = None
        self.Ar: Union[csc_matrix, None] = None
        self.B_topo_inverse: Union[csc_matrix, None] = None
        self.R: Union[csc_matrix, None] = None
        self.A: Union[csc_matrix, None] = None

    def create(
        self,
        character: Character,
        cage: Cage,
        skel: Skeleton,
        psi: Weights,
        phi: Weights,
        omega: np.ndarray,
        selected_vertices: Union[List[int], None] = None
    ) -> bool:
        """
        # psi : skeleton_updater_weights
        # phi : cage_weights
        # omega : skeleton_weights
        """
        self.character = character
        self.cage = cage
        self.skel = skel
        self.psi = psi
        self.phi = phi
        self.omega = omega
        self.selected_vertices_for_inversion = selected_vertices

        # 1. 가중치 행렬로부터 Kronecker Product을 사용하여 희소 행렬 생성
        self.PSI = self._identity_kronecker_product(self.psi.matrix)
        
        phi_weights = self.phi.matrix
        omega_weights = self.omega
        if self.selected_vertices_for_inversion:
            phi_weights = phi_weights[:, self.selected_vertices_for_inversion]
            omega_weights = omega_weights[:, self.selected_vertices_for_inversion]

        self.PHI = self._identity_kronecker_product(phi_weights)
        self.PHI_transpose = self.PHI.transpose().tocsc()
        self.OMEGA = self._identity_kronecker_product(omega_weights).transpose()

        # 2. 시스템 구성에 필요한 행렬들 계산
        self.B_topo_inverse = self._compute_B_topo_inverse()
        self.update_Ar()
        self.update_R()
        self._update_solver_matrix()

        self.refresh_matrices = False
        return True

    def skeleton_edited(self):
        self.refresh_matrices = True

    def _identity_kronecker_product(self, W: np.ndarray) -> csc_matrix:
        """ W와 3x3 단위 행렬의 크로네커 곱을 계산 (W_ij * I_3) """
        return csc_matrix(kron(W, eye(3)))

    def _compute_B_topo_inverse(self) -> csc_matrix:
        """ 스켈레톤의 계층 구조를 나타내는 B_topo 행렬의 역행렬을 계산 """
        num_nodes = self.skel.num_nodes
        dim = 3 * num_nodes
        B_topo = lil_matrix((dim, dim))

        for j in range(num_nodes):
            node = self.skel[j]
            father = node.father
            
            if father == -1: # Root node
                for d in range(3):
                    B_topo[3*j + d, 3*j + d] = 1.0
            else: # Non-root node
                for d in range(3):
                    B_topo[3*j + d, 3*j + d] = -1.0
                    B_topo[3*j + d, 3*father + d] = 1.0
        
        B_topo_csc = B_topo.tocsc()
        return inv(B_topo_csc)

    def update_Ar(self):
        """ 스켈레톤 변형의 회전 성분을 나타내는 Ar 행렬 업데이트 """
        num_nodes = self.skel.num_nodes
        dim = 3 * num_nodes
        Ar_lil = lil_matrix((dim, dim))

        rotations = {}
        for j in range(num_nodes):
            node = self.skel[j]
            T_rest_inv = np.linalg.inv(node.global_t_rest.matrix)
            Rj = node.global_t_current.matrix @ T_rest_inv
            rotations[j] = Rj[:3, :3]

        for j in range(num_nodes):
            father = self.skel[j].father
            Rj = rotations[j]

            if father == -1: # Root node
                block = np.eye(3) - Rj
            else: # Non-root node
                Rf = rotations[father]
                block = Rj - Rf
            
            Ar_lil[3*j:3*j+3, 3*j:3*j+3] = block

        self.Ar = Ar_lil.tocsc()

    def update_R(self):
        """ 각 캐릭터 정점의 블렌딩된 회전을 나타내는 R 행렬 업데이트 """
        num_nodes = self.skel.num_nodes
        
        vert_indices = self.selected_vertices_for_inversion
        if not vert_indices:
            vert_indices = range(self.character.num_vertices)
        
        num_verts = len(vert_indices)
        dim = 3 * num_verts
        R_lil = lil_matrix((dim, dim))

        # 각 관절의 회전 행렬 미리 계산
        joint_rotations = []
        for j in range(num_nodes):
            node = self.skel[j]
            T_rest_inv = np.linalg.inv(node.global_t_rest.matrix)
            Rj = (node.global_t_current.matrix @ T_rest_inv)[:3, :3]
            joint_rotations.append(Rj)

        # 각 정점에 대해 블렌딩된 회전 계산
        for i_idx, i in enumerate(vert_indices):
            Ri = np.zeros((3, 3))
            for j in range(num_nodes):
                w = self.omega[j, i]
                Ri += w * joint_rotations[j]
            
            R_lil[3*i_idx:3*i_idx+3, 3*i_idx:3*i_idx+3] = Ri

        self.R = R_lil.tocsc()

    def _update_solver_matrix(self):
        """ 최소제곱법으로 풀 시스템 행렬 A를 구성 """
        # A = PHI.T * ((R * PHI) + (OMEGA * B_topo_inverse * Ar * PSI))
        print("\t Computation of A : ")
        term1 = self.R @ self.PHI
        print(f"OMGEA: {self.OMEGA.shape}, B_topo_inverse: {self.B_topo_inverse.shape}, Ar: {self.Ar.shape}, PSI: {self.PSI.shape}")
        term2 = self.OMEGA @ self.B_topo_inverse @ self.Ar @ self.PSI
        self.A = self.PHI_transpose @ (term1 + term2)
        self.A = self.A.tocsc()
        print("\t Computation of A done")

    def propagate_to_rest(self):
        """
        케이지의 현재 포즈(C') 변경을 휴식 포즈(C)로 전파합니다.
        선형 시스템 A * dC = b 를 풀어 dC를 구합니다.
        """
        if self.refresh_matrices:
            print("Updating CageReverser matrices...")
            self.update_Ar()
            self.update_R()
            self._update_solver_matrix()
            self.refresh_matrices = False
            print("CageReverser matrices updated.")

        # 1. 케이지의 마지막 변위(dC')를 가져옴
        # C++에서는 lastTranslations를 사용하지만, 여기서는 rest와 current의 차이로 계산
        num_cage_verts = self.cage.num_vertices
        c_prime = self.cage.current_pose_vertices.reshape(num_cage_verts, 3)
        c_rest = self.cage.rest_pose_vertices.reshape(num_cage_verts, 3)
        
        # 이전 상태를 저장하는 로직이 없으므로, dC'는 (C'_new - C'_old)가 되어야 함.
        # 여기서는 간단히 (C'_current - C_rest)를 변위로 가정.
        # 실제 애플리케이션에서는 편집 전후의 C' 차이를 사용해야 함.
        # dC_prime = (c_prime - c_prime_old).flatten()
        # 여기서는 C++ 코드의 lastTranslations와 유사한 cage.last_translations가 있다고 가정
        if not hasattr(self.cage, 'last_translations') or self.cage.last_translations is None:
             # last_translations가 없으면 변위가 없다고 가정
            dC_prime = np.zeros(num_cage_verts * 3)
        else:
            dC_prime = self.cage.last_translations

        if np.linalg.norm(dC_prime) < 1e-9:
            print("No cage editing has detected")
            return # 변위가 없으면 계산할 필요 없음

        # 2. 우변 b 계산
        # b = PHI.T * (PHI * dC')
        b = self.PHI_transpose @ (self.PHI @ dC_prime)

        # 3. 최소제곱법으로 dC 풀기
        # A * dC = b
        # lsqr은 (A, b)를 받아 x를 반환 (Ax=b)
        # print("\t Solving for dC...")
        result = lsqr(self.A, b)
        dC = result[0]
        # print("\t Solving for dC done.")

        # 4. 계산된 변위 dC를 휴식 포즈에 적용
        old_rest_vertices = self.cage.rest_pose_vertices
        new_rest_vertices = old_rest_vertices + dC
        self.cage.rest_pose_vertices = new_rest_vertices

        # 변위 적용 후 last_translations 초기화 (다음 편집을 위해)
        self.cage.last_translations = np.zeros_like(dC_prime)
