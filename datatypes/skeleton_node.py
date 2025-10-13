from .transform import Transform

class SkeletonNode:
    def __init__(self, name, father, local_t: Transform, global_t: Transform):
        self._name = name
        self._father = father  # Index of the father node
        self.next = []  # Index of the attached nodes

        # Transformations
        self._local_t_current = local_t
        self._global_t_current = global_t

        self._local_t_rest = local_t.copy()
        self._global_t_rest = global_t.copy()

        self._global_t = Transform()

    @property
    def name(self):
        return self._name

    @property
    def father(self):
        return self._father

    @property
    def local_t_current(self):
        return self._local_t_current
    @local_t_current.setter
    def local_t_current(self, value):
        self._local_t_current = value

    @property
    def global_t_current(self):
        return self._global_t_current
    @global_t_current.setter
    def global_t_current(self, value):
        self._global_t_current = value

    @property
    def local_t_rest(self):
        return self._local_t_rest
    @local_t_rest.setter
    def local_t_rest(self, value):
        self._local_t_rest = value

    @property
    def global_t_rest(self):
        return self._global_t_rest
    @global_t_rest.setter
    def global_t_rest(self, value):
        self._global_t_rest = value

    @property
    def global_t_offset(self):
        return self._global_t
    @global_t_offset.setter
    def global_t_offset(self, value):
        self._global_t = value