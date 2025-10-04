import numpy as np
from .skeleton_node import SkeletonNode
from .transform import Transform

# TODO: Implement the cg3 library functionalities.

class BoundingBox:
    """
    For 3D BoundingBox
    """
    def __init__(self):
        self.clear()
    
    def clear(self):
        """
        Intialize min and max as opposite infinite value
        """
        self.__somePrivate__ = ""
        self._min = np.array([np.inf] * 3, dtype=np.float64) # <--- _min으로 변경
        self._max = np.array([-np.inf] * 3, dtype=np.float64) # <--- _max로 변경
    
    @property
    def min(self):
        return self._min 
    @min.setter
    def min(self, value):
        self._min = value
    
    @property
    def max(self):
        return self._max
    @max.setter
    def max(self, value):
        self._max = value

    @property
    def center(self):
        return (self.min + self.max) * 0.5

    @property
    def diagonal(self):
        return np.linalg.norm(self.min - self.max)


class Skeleton:
    def __init__(self, joints=None, joints_rotations=None, fathers=None, names=None):
        self._root_indexes = []
        self._nodes = []
        self.bounding_box = BoundingBox()
        self._root_motion = Transform()

        if joints is not None and joints_rotations is not None and fathers is not None and names is not None:
            self.create(joints, joints_rotations, fathers, names)

    @property
    def root_indexes(self):
        return self._root_indexes

    @property
    def nodes(self):
        return self._nodes

    @property
    def root_motion(self):
        return self._root_motion

    def create(self, joints, joints_rotations, fathers, names):
        self.clear()

        for i in range(len(joints)):
            # TODO: Use a proper Transform class
            # For now, we assume joints_rotations are Euler angles and joints are translations
            # and create a transformation matrix from them.
            rx, ry, rz = joints_rotations[i]
            x, y, z = joints[i]
            transform = Transform(tx=x, ty=y, tz=z, rx=rx, ry=ry, rz=rz)

            self.add_node(names[i], fathers[i], Transform(), transform)

        self.update_local_from_global_rest()
        self.update_local_from_global_current()

        return True

    def clear(self):
        self._root_indexes.clear()
        self._nodes.clear()
        self.bounding_box.clear()
        self._root_motion = Transform()

    def add_node(self, node_name, father, local_transformation, model_transformation):
        node = SkeletonNode(node_name, father, local_transformation, model_transformation)
        node_id = len(self.nodes)
        self._nodes.append(node)

        if father != -1:
            self._nodes[father].next.append(node_id)
        else:
            self._root_indexes.append(node_id)

        self.update_bounding_box()

        return node_id

    def update_local_from_global_rest(self):
        stack = list(self.root_indexes)
        while stack:
            n = stack.pop()
            node = self.nodes[n]
            father_index = node.father

            if father_index != -1:
                rest_father_transformation = self.nodes[father_index].global_t_rest
                node.local_t_rest = node.global_t_rest * rest_father_transformation.inverse()
            else:
                node.local_t_rest = node.global_t_rest.copy()

            stack.extend(node.next)

    def update_local_from_global_current(self):
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
        for i in range(len(self.nodes)):
            t_rest = self.nodes[i].global_t_rest
            t_current = self.nodes[i].global_t_current
            self.nodes[i].global_t = t_current * t_rest.inverse()

    def set_keyframe(self, keyframe: list[Transform]):
        for i in range(len(self.nodes)):
            self.nodes[i].local_t_current = keyframe[i]
        self.update_global_from_local_current()
        for i in self.root_indexes:
            t = self.nodes[i].global_t_current.get_translation() - self.nodes[i].global_t_rest.get_translation()
            self.root_motion.set_translation(t)

    def interpolate_keyframes(self, keyframe_low: list[Transform], keyframe_top: list[Transform], a: float):
        for i in range(len(self.nodes)):
            self.nodes[i].local_t_current = keyframe_low[i].interpolate(keyframe_top[i], a)
        self.update_global_from_local_current()
        for i in self.root_indexes:
            t = self.nodes[i].global_t_current.get_translation() - self.nodes[i].global_t_rest.get_translation()
            self.root_motion.set_translation(t)