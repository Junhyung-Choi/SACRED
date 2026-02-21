import numpy as np
from ..datatypes.weights import Weights
from ..datatypes.character import Character
from ..datatypes.skeleton import Skeleton
from ..datatypes.cage import Cage

from .mvc import compute_mvc_coordinates_3d, compute_mvc_weight_matrix
from .mec import compute_mec_coordinates

import torch

class SkeletonUpdater:
    """
    Skeleton Updater (C -> S)

    Rest Pose에서 변경된 Cage(C)를 Skeleton(S)에 적용하여 업데이트합니다.
    단순히 관절 위치를 보간하는 대신, Joint Localization 함수(L)와
    MEC(Maximum Entropy Coordinates)를 사용하여 Cage Weight를 계산하고,
    이를 기반으로 변형된 Cage에 따라 Skeleton을 보간합니다.

    C++ 원본: `SuperCages/operators/skeletonUpdater.cpp`
    """
    def __init__(self):
        self.w_skel: Weights | None = None
        self.w_cage: Weights | None = None
        self.character: Character | None = None
        self.skeleton: Skeleton | None = None
        self.cage: Cage | None = None
        self.skeleton_updater_weights: Weights | None = None
        self.original_node_positions: np.ndarray = np.array([], dtype=np.float64)
        self.clear()

    def create(self, w_skeleton: np.ndarray, w_cage: Weights, character: Character, skeleton: Skeleton, cage: Cage) -> bool:
        """
        Param:
        - w_skeleton #(N_Bone, N_CharV)
        """

        self.clear()

        self.w_skel = w_skeleton
        self.w_cage = w_cage
        self.character = character
        self.skeleton = skeleton
        self.cage = cage

        self.skeleton_updater_weights = self.generate_skeleton_updater_weights(
            w_skeleton, w_cage, character, skeleton, cage
        )

        num_nodes = skeleton.num_nodes
        self.original_node_positions = np.zeros((num_nodes, 3), dtype=np.float64)
        for i in range(num_nodes):
            self.original_node_positions[i] = skeleton[i].local_t_current.translation

        return True

    def clear(self):
        self.w_skel = None
        self.w_cage = None
        self.character = None
        self.skeleton = None
        self.cage = None
        self.skeleton_updater_weights = None
        self.original_node_positions = np.array([], dtype=np.float64)

    @staticmethod
    def generate_skeleton_updater_weights(
        skeleton_weights: np.ndarray, cage_weights: torch.Tensor, character: Character, skeleton: Skeleton, cage: Cage
    ) -> Weights:
        num_nodes = skeleton.num_nodes
        num_char_vertices = character.num_vertices
        num_cage_vertices = cage.num_vertices

        mesh_vertices = character.rest_pose_vertices.reshape(-1, 3)
        mesh_triangles = character.triangles.reshape(-1, 3)
        cage_vertices = cage.rest_pose_vertices.reshape(-1, 3)

        updater_weights = Weights(num_nodes, num_cage_vertices)

        # 1. 모든 관절의 MVC 가중치를 미리 계산 (캐릭터 메시 기준)
        joint_positions = np.array([node.global_t_rest.translation for node in skeleton.nodes])
        bone_char_mvc_weights = compute_mvc_weight_matrix(
            src_vertices=joint_positions,
            cage_vertices=mesh_vertices,
            cage_triangles=mesh_triangles,
        ) 
        
        fallback_count = 0

        for j in range(num_nodes):
            joint_j_pos = skeleton[j].global_t_rest.translation
            father = skeleton[j].father
            bones_adj = [father] if father != -1 else []
            bones_adj.append(j)

            # [DEBUG] 첫 번째 관절 혹은 특정 주기마다 유닛/위치 검증
            if j == 0:
                print(f"\n--- [Unit/Position Check for Joint {j}] ---")
                print(f"  Joint Pos: {joint_j_pos}")
                cage_min, cage_max = cage_vertices.min(axis=0), cage_vertices.max(axis=0)
                print(f"  Cage BBox: {cage_min} ~ {cage_max}")
                is_inside = np.all((joint_j_pos >= cage_min) & (joint_j_pos <= cage_max))
                print(f"  Is Joint inside Cage BBox?: {is_inside}")
                min_dist = np.min(np.linalg.norm(cage_vertices - joint_j_pos, axis=1))
                print(f"  Distance to nearest cage vertex: {min_dist:.4f}m")

            # 1. MVC 좌표 가져오기 (Mesh 공간)
            mvcoords = bone_char_mvc_weights[j]

            # 2. 순수 기하학적 투영 (Fallback에서 100% 성공하는 안전한 Base Prior)
            pure_prior = mvcoords @ cage_weights.numpy()
            pure_prior = np.maximum(pure_prior, 0.0)

            # 3. LBS Locality를 Mesh에서 Cage 공간으로 변환
            w_vb = skeleton_weights[bones_adj, :] 
            mesh_locality = np.sum(w_vb, axis=0) 
            
            # [핵심] LBS 가중치 자체를 케이지의 영향력으로 투영합니다
            cage_locality = mesh_locality @ cage_weights.numpy() 
            
            # Cage Locality 정규화 (가장 영향력이 큰 케이지 정점이 1.0이 되도록)
            if cage_locality.max() > 0:
                cage_locality /= cage_locality.max()

            # 4. Cage 레벨에서 Soft Blending
            base_ratio = 0.2 # 기하학적 구조 유지를 위한 최소 비율 (20%)
            soft_locality = base_ratio + (1.0 - base_ratio) * cage_locality
            
            # 최종 Prior 계산 (안전한 Base에 조절된 Locality 곱하기)
            joint_weights_invalid = pure_prior * soft_locality

            # [STABILIZATION] 최소값 보장 및 정규화
            joint_weights_invalid = np.maximum(joint_weights_invalid, 1e-6)
            prior_sum = joint_weights_invalid.sum()
            if prior_sum > 0:
                joint_weights_invalid /= prior_sum
            
            # 5. MEC 계산
            try:
                joint_weights, mec_stats = compute_mec_coordinates(
                    joint_j_pos, cage_vertices, joint_weights_invalid,
                    max_iterations=100, line_search_steps=50, max_dw=0.001
                )

                # 정밀도가 낮을 경우(수렴 실패)의 Safe Guard
                if mec_stats.linear_precision_error > 1e-5:
                    print(f"  ⚠️ Joint {j}: MEC failed with Prior. Retrying with Mesh-to-Cage Fallback...")
                    fallback_count += 1
                    fallback_prior = np.maximum(mvcoords @ cage_weights.numpy(), 1e-6)
                    fallback_prior /= fallback_prior.sum()
                    
                    joint_weights, _ = compute_mec_coordinates(
                        joint_j_pos, cage_vertices, fallback_prior,
                        max_iterations=100, line_search_steps=50, max_dw=0.001
                    )

            except Exception as e:
                print(f"  ❌ Joint {j} MEC Error: {e}")
                joint_weights = np.zeros(num_cage_vertices)

            updater_weights[j, :] = joint_weights

        print(f"\n\t find_weights_for_articulations done (fallback_count={fallback_count})")
        return updater_weights

    def update_position(self):
        """
        케이지 변형에 따라 스켈레톤의 위치를 업데이트합니다.
        """
        if not all([self.skeleton, self.cage, self.skeleton_updater_weights]):
            return
        
        before_update = np.array([node.local_t_current.translation for node in self.skeleton.nodes])

        skeleton_nodes = self.skeleton.nodes
        num_nodes = len(skeleton_nodes)
        
        # 1. 루트 모션 계산
        # * NOTE: original codes looks like supporting multiple root_motion
        # * However, it only uses last one, so we can also use last root_node_index
        root_motion = np.zeros(3)
        root_node_idx = self.skeleton.root_indexes[-1]
        if root_node_idx != -1:
            root_node = skeleton_nodes[root_node_idx]
            root_motion = root_node.local_t_current.translation - root_node.local_t_rest.translation

        # 2. 케이지 정점과 가중치를 사용하여 새로운 Rest Pose의 전역 위치 계산
        # NumPy를 사용하여 모든 노드에 대해 한 번에 계산
        # self.skeleton_updater_weights.weights: (num_cage_vertices, num_nodes)
        # self.cage.rest_pose_vertices: (num_cage_vertices * 3)
        
        W = self.skeleton_updater_weights.matrix # (65, 24) # (N_Bone, N_CageV)
        C_rest = self.cage.rest_pose_vertices.reshape(-1, 3) # (N_CageV, 3)
        # Z_diff = self.cage.original_rest_pose_vertices.reshape(-1, 3) - C_rest #(N_CageV, 3)
        # C_rest += Z_diff * 2  # 원본 C++ 코드의 Z축 뒤집기 보정

        # pRest = W @ C_rest  (num_nodes, 3)
        new_global_rest_positions = W @ C_rest #(N_Bone, 3)

        for j in range(num_nodes):
            skeleton_nodes[j].global_t_rest.set_translation(new_global_rest_positions[j])

        # 3. 전역 Rest Pose로부터 지역 Rest Pose 업데이트
        self.skeleton.update_local_from_global_rest()

        # 4. 새로운 지역 Rest Pose를 현재 포즈에 적용 (루트 모션 포함)
        for j in range(num_nodes):
            t = skeleton_nodes[j].local_t_rest.translation
            if skeleton_nodes[j].father == -1:
                t += root_motion
            skeleton_nodes[j].local_t_current.set_translation(t)

        # 5. 지역 현재 포즈로부터 전역 현재 포즈 업데이트
        self.skeleton.update_global_from_local_current()

        after_update = np.array([node.local_t_current.translation for node in self.skeleton.nodes])
        
        diff = after_update - before_update
        
        # print(f"스켈레톤 변경 사항:")
        # for i, d in enumerate(diff):
        #     if np.sum(np.abs(d)) > 1e-3:
        #             print(f"  - 관절 {i} ({self.skeleton.nodes[i].name}): 위치 변경됨 {before_update[i]} -> {after_update[i]} (차이: {d})")

    def get_weights(self) -> Weights:
        return self.skeleton_updater_weights
