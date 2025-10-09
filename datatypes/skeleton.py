import numpy as np
from typing import List, Union
from .skeleton_node import SkeletonNode
from .transform import Transform
from .bbox import BoundingBox

class Skeleton:
    def __init__(self, joints: List[np.ndarray] = None, joints_rotations: List[np.ndarray] = None, 
                 fathers: List[int] = None, names: List[str] = None):
        self._root_indexes: List[int] = []
        self._nodes: List['SkeletonNode'] = []
        self.bounding_box: BoundingBox = BoundingBox()
        self._root_motion: 'Transform' = Transform() # Transform 객체로 초기화

        if joints is not None and joints_rotations is not None and fathers is not None and names is not None:
            self.create(joints, joints_rotations, fathers, names)

    # * Properties 
    @property
    def root_indexes(self) -> List[int]:
        return self._root_indexes

    @property
    def nodes(self) -> List['SkeletonNode']:
        return self._nodes
    
    @property
    def num_roots(self):
        return len(self._root_indexes)

    @property
    def num_nodes(self):
        return len(self._nodes)

    def __getitem__(self, index: Union[int, slice]) -> Union['SkeletonNode', List['SkeletonNode']]:
        """
        C++의 getNode(ulong index)에 대응하며, 인덱스 또는 슬라이스 접근을 지원합니다.
        """
        if isinstance(index, int):
            return self._nodes[index]
        
        elif isinstance(index, slice):
            return self._nodes[index]

        raise TypeError("Index must be an integer or a slice.")

    @property
    def root_motion(self) -> 'Transform':
        return self._root_motion
        
    # * Constructor

    def create(self, joints: List[np.ndarray], joints_rotations: List[np.ndarray], 
               fathers: List[int], names: List[str]) -> bool:
        self.clear()

        for i in range(len(joints)):
            # joints_rotations[i]는 (rx, ry, rz), joints[i]는 (x, y, z)
            rx, ry, rz = joints_rotations[i].tolist()
            tx, ty, tz = joints[i].tolist()
            
            # C++의 Transform(rx, ry, rz, x, y, z)에 대응
            transform = Transform(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz)

            self._add_node(names[i], fathers[i], Transform(), transform)

        self.update_local_from_global_rest()
        self.update_local_from_global_current()

        return True

    def clear(self):
        self._root_indexes.clear()
        self._nodes.clear()
        self.bounding_box.clear()
        self._root_motion = Transform()

    def _add_node(self, node_name: str, father: int, 
                 local_transformation: 'Transform', model_transformation: 'Transform') -> int:
        
        node = SkeletonNode(node_name, father, local_transformation, model_transformation)
        node_id = len(self._nodes)
        self._nodes.append(node)

        if father != -1:
            # 부모 노드의 next에 자식 ID 추가
            self._nodes[father].next.append(node_id)
        else:
            self._root_indexes.append(node_id)

        self.update_bounding_box()

        return node_id

    # --- Update Transformations ---
    def update_bounding_box(self):
        self.bounding_box.clear()

        # 모든 노드의 Current Global Transform Translation을 사용하여 바운딩 박스 업데이트
        for node in self.nodes:
            pos = node.global_t_current.translation
            
            self.bounding_box.min = np.minimum(self.bounding_box.min, pos)
            self.bounding_box.max = np.maximum(self.bounding_box.max, pos)

    def update_local_from_global_rest(self):
        """
        Rest Pose의 Global Transform에서 Local Transform을 계산합니다.
        """
        stack = list(self.root_indexes)
        while stack:
            n = stack.pop()
            node = self.nodes[n]
            father_index = node.father

            if father_index != -1:
                # * 부모가 있는 경우: L_rest = G_rest * G_father_rest^(-1) 로 Local Transform을 계산합니다.
                # * 이는 Forward Kinematics (G_rest = G_father_rest * L_rest)의 역연산입니다.
                rest_father_transformation = self.nodes[father_index].global_t_rest
                node.local_t_rest = node.global_t_rest * rest_father_transformation.inverse()
            else:
                # * 루트 노드인 경우: Local Transform은 Global Transform과 동일합니다.
                node.local_t_rest = node.global_t_rest.copy()
            
            stack.extend(node.next) 
    
    def update_local_from_global_current(self):
        """
        Current Pose의 Global Transform에서 Local Transform을 계산합니다.
        """
        stack = list(self.root_indexes)
        while stack:
            n = stack.pop()
            node = self.nodes[n]
            father_index = node.father

            if father_index != -1:
                father_transformation = self.nodes[father_index].global_t_current
                node.local_t_current = node.global_t_current * father_transformation.inverse()
            else:
                node.local_t_current = node.global_t_current.copy()

            stack.extend(node.next)

    def update_global_from_local_rest(self):
        """
        Rest Pose의 Local Transform에서 Global Transform을 계산합니다.
        """
        stack = list(self.root_indexes)
        while stack:
            n = stack.pop()
            node = self.nodes[n]
            father_index = node.father

            if father_index != -1:
                rest_father_transformation = self.nodes[father_index].global_t_rest
                node.global_t_rest = rest_father_transformation * node.local_t_rest
            else:
                node.global_t_rest = node.local_t_rest.copy()

            stack.extend(node.next)

    def update_global_from_local_current(self):
        """
        Current Pose의 Global Transform에서 Local Transform을 계산합니다.
        """
        stack = list(self.root_indexes)
        while stack:
            n = stack.pop()
            node = self.nodes[n]
            father_index = node.father

            if father_index != -1:
                father_transformation = self.nodes[father_index].global_t_current
                node.global_t_current = father_transformation * node.local_t_current
            else:
                node.global_t_current = node.local_t_current.copy()

            stack.extend(node.next)

    def update_global_t(self):
        """
        Current Pose = Transform * Rest Pose
        -> Transform(변화량) = Current Pose * Rest Pose^(-1)
        """
        for node in self.nodes:
            node.global_t = node.global_t_current * node.global_t_rest.inverse()

    # --- Animation/Keyframe Methods ---

    def set_keyframe(self, keyframe: List['Transform']):
        """
        C++ 코드에서 rootMotion은 반복분 내내 덮어씌워지고
        root_index가 여러개인 상황을 고려하지 않기 때문에
        해당 반복문을 제거하고, 루트 노드의 모션만 사용하도록 수정했습니다.
        """
        for i in range(len(self.nodes)):
            self.nodes[i].local_t_current = keyframe[i]
            
        self.update_global_from_local_current()
        
        if self.root_indexes:
            root_node = self.nodes[self.root_indexes[0]]
            
            t_current = root_node.global_t_current.translation
            t_rest = root_node.global_t_rest.translation
            t = t_current - t_rest
            
            self._root_motion.set_translation(t)

    def interpolate_keyframes(self, keyframe_low: List['Transform'], keyframe_top: List['Transform'], a: float):
        """
        C++ 코드에서 rootMotion은 반복분 내내 덮어씌워지고
        root_index가 여러개인 상황을 고려하지 않기 때문에
        해당 반복문을 제거하고, 루트 노드의 모션만 사용하도록 수정했습니다.
        """
        for i in range(len(self.nodes)):
            self.nodes[i].local_t_current = keyframe_low[i].interpolate(keyframe_top[i], a)
            
        self.update_global_from_local_current()
        
        if self.root_indexes:
            root_node = self.nodes[self.root_indexes[0]]
            
            t_current = root_node.global_t_current.translation
            t_rest = root_node.global_t_rest.translation
            t = t_current - t_rest
            
            self._root_motion.set_translation(t)
            
    # --- Specific Transformation Methods ---

    # ! On C++ Code's Comment : "It's buggy!"
    def add_global_transformation(self, node_index: int, transformation: 'Transform'):
        """
        주어진 변환을 특정 노드의 현재 글로벌 변환에 적용하고
        로컬 변환 및 모든 자식 노드의 글로벌 변환을 업데이트합니다.

        Note:
        C++ 원본 코드는 스켈레톤 전체를 업데이트하고
        Rest Pose를 건드리는 비효율적이고 잠재적으로 버그가 있는 로직을 포함하고 있었기에, 이 구현은
        해당 노드와 그 **자손들(descendants)만** 업데이트하여 효율성을 높였으며, 휴식 포즈(Rest Pose)는
        변경하지 않도록 수정되었습니다
        """
        if node_index >= len(self.nodes):
            return

        node = self.nodes[node_index]
    
        node.global_t_current = node.global_t_current.transform_towards_origin(transformation)

        # 1. 로컬 변환 (local_t_current) 업데이트
        father_index = node.father
        if father_index != -1:
            father_transformation = self.nodes[father_index].global_t_current
            node.local_t_current = father_transformation.inverse() * node.global_t_current
        else:
            node.local_t_current = node.global_t_current.copy() # 루트 노드의 local은 global과 동일

        # 2. 자식 노드의 글로벌 변환 재계산 (스택 기반)
        # C++ : updateGlobalFromLocalCurrent()
        stack = list(node.next)
        while stack:
            n = stack.pop()
            child_node = self.nodes[n]
            father_transformation = self.nodes[child_node.father].global_t_current
            child_node.global_t_current = father_transformation * child_node.local_t_current
            
            stack.extend(child_node.next)
            
        # ! C++ 코드에 있던 updateGlobalFromLocalRest() 호출은 제거됨 (rest 포즈는 변경되지 않아야 하므로)
        self.update_bounding_box()