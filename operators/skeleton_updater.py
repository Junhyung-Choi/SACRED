import numpy as np
from ..datatypes.weights import Weights
from ..datatypes.character import Character
from ..datatypes.skeleton import Skeleton
from ..datatypes.cage import Cage

from .mvc import compute_mvc_coordinates_3d, compute_mvc_weight_matrix
from .mec import compute_mec_coordinates

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

    def create(self, w_skeleton: Weights, w_cage: Weights, character: Character, skeleton: Skeleton, cage: Cage) -> bool:
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
        skeleton_weights: Weights, cage_weights: Weights, character: Character, skeleton: Skeleton, cage: Cage
    ) -> Weights:
        """
        스켈레톤-케이지 결합을 위한 가중치(Skeleton Updater Weights)를 생성합니다.
        각 스켈레톤 관절을 케이지 정점의 선형 결합으로 표현하는 가중치를 계산합니다.

        NOTE: AVWeights matrix is not used, so it is removed in this conversion
        """
        print("\t find_weights_for_articulations")
        num_nodes = skeleton.num_nodes # N_SkelNode
        num_char_vertices = character.num_vertices # N_CharV
        num_cage_vertices = cage.num_vertices # N_CageV

        mesh_vertices = character.rest_pose_vertices.reshape(-1, 3) # (N_CharV, 3)
        mesh_triangles = character.triangles.reshape(-1, 3) # (N_CharF, 3)
        cage_vertices = cage.rest_pose_vertices.reshape(-1, 3) #(N_CageV, 3)

        updater_weights = Weights(num_nodes, num_cage_vertices) # (N_SkelNode, N_CageV)

        s_exponent = 0.05  # 파인튜닝된 파라미터
        epsilon = 0.01

        joint_positions = np.array([node.global_t_rest.translation for node in skeleton.nodes])

        bone_char_mvc_weights = compute_mvc_weight_matrix( # (N_SkelNode, N_CharV)
            src_vertices=joint_positions,
            cage_vertices=mesh_vertices,
            cage_triangles=mesh_triangles,
        ) 

        # C++의 OMP 병렬 루프를 순차적으로 변환. 필요 시 joblib 등으로 병렬화 가능
        for j in range(num_nodes):
            father = skeleton[j].father
            bones_adj = [father] if father != -1 else []
            bones_adj.append(j)

            joint_j_pos = skeleton[j].global_t_rest.translation

            # 1. Joint j에 대한 MVC 계산 (vs. Character)
            # mvcoords = compute_mvc_coordinates_3d(
            #     joint_j_pos, mesh_triangles, mesh_vertices
            # )
            mvcoords = bone_char_mvc_weights[j] # (N_CharV, )

            # 2. Locality Factor를 적용하여 Weight Prior 계산
            weight_prior = np.zeros(num_char_vertices)

            # skeleton_weights.weights는 (N_Bone, N_CharV) 형태라고 가정
            w_vb = skeleton_weights[bones_adj, :] # (len(bones_adj), N_CharV)
            locality_factor = np.power(np.maximum(epsilon, np.abs(w_vb)), s_exponent).sum(axis=0) - 1 # (N_CharV, )
            locality_factor = np.maximum(locality_factor, epsilon)

            weight_prior = mvcoords * locality_factor # (N_CharV, ) = (N_CharV, ) * (N_CharV, ) 

            # 3. Weight Prior를 케이지에 투영
            joint_weights_invalid = weight_prior @ cage_weights.matrix # (N_CageV, ) = (N_CharV, ) * (N_CharV, N_CageV) 

            # 4. MEC(Maximum Entropy Coordinates) 계산
            try:
                # C++ 원본의 파라미터(100, 50, 0.001)를 적용
                joint_weights, mec_stats = compute_mec_coordinates(
                    joint_j_pos, cage_vertices, joint_weights_invalid,
                    max_iterations=100, line_search_steps=50, max_dw=0.001
                )

                # MEC 실패 시 예외 처리 (AWFUL safe guard)
                if mec_stats.linear_precision_error > 1e-6:
                    print("MEC has failed with the LBS derived prior. Re-computing without it.")
                    # mvcoords와 cage_weights를 사용한 대체 prior 계산
                    joint_weights_invalid_fallback = mvcoords @ cage_weights.T
                    joint_weights, mec_stats = compute_mec_coordinates(
                        joint_j_pos, cage_vertices, joint_weights_invalid_fallback,
                        max_iterations=100, line_search_steps=50, max_dw=0.001
                    )

            except Exception as e:
                print(f"MEC computation failed: {e}")
                joint_weights = np.zeros(num_cage_vertices)

            # breakpoint()
            updater_weights[j, :] = joint_weights

        print("\t find_weights_for_articulations done")
        return updater_weights

    def update_position(self):
        """
        케이지 변형에 따라 스켈레톤의 위치를 업데이트합니다.
        """
        if not all([self.skeleton, self.cage, self.skeleton_updater_weights]):
            return

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

    def get_weights(self) -> Weights:
        return self.skeleton_updater_weights
