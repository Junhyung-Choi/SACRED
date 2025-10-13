import numpy as np
from typing import List, Union
from scipy.sparse import csr_matrix, kron

class Weights:
    def __init__(self, vertex_number: int = 0, handle_number: int = 0, data : np.ndarray = None):
        """
        Initializes a Weights object that behaves like a NumPy array.
        This class stores a matrix of weights, typically for skinning, where each row corresponds to a vertex
        and each column to a handle (e.g., a bone).
        You can access and modify weights using standard NumPy slicing:
        - w[vertex_id, handle_id]
        - w[vertex_id]
        - w[vertex_id, handle_id] = 0.5
        Args:
            vertex_number (int): The number of vertices.
            handle_number (int): The number of handles.
            data (np.ndarray, optional) : A Numpy array to initialize the weights with.
                                          If provided, vertex number and handle number are ignored.
        """
        self._matrix: np.ndarray = np.zeros((0, 0))
        self.create(vertex_number=vertex_number, handle_number=handle_number, data=data)

    @classmethod
    def from_ndarray(cls, data: np.ndarray) -> 'Weights':
        """
        Create a Weights object directly from a Numpy Array.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data must be a 2D NumPy array.")

        rows, cols = data.shape
        weights_obj = cls(rows,cols)
        weights_obj._matrix = data.copy() # Use a copy to prevent external modifications
        return weights_obj


    def create(self, vertex_number: int = 0, handle_number: int = 0, data:np.ndarray=None):
        """
        Resizes the weights matrix, initializing all weights to zero.
        Args:
            vertex_number (int): The new number of vertices.
            handle_number (int): The new number of handles.
        """
        if data is not None:
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                raise ValueError("Input data must be a 2D NumPy array.")
            self._matrix = data.copy()
        else:
            self._matrix = np.zeros((vertex_number, handle_number))

    def clear(self):
        """Clears all weights and resets the matrix to 0x0."""
        self.create(0, 0)

    def __getitem__(self, key):
        """
        Gets weights using standard NumPy array indexing.
        Examples:
            weight = weights[vertex_id, handle_id]
            weights_for_vertex = weights[vertex_id]
        """
        return self._matrix[key]

    def __setitem__(self, key, value):
        """
        Sets weights using standard NumPy array indexing.
        Example:
            weights[vertex_id, handle_id] = new_weight
        """
        self._matrix[key] = value

    @property
    def matrix(self) -> np.ndarray:
        """Returns the entire weight matrix as a numpy array."""
        return self._matrix

    def get_non_zeros(self, vertex_id: int) -> List[int]:
        """
        Gets the indices of handles with non-zero weights for a given vertex.
        Args:
            vertex_id (int): The index of the vertex.
        Returns:
            list[int]: A list of handle indices with non-zero weights.
        """
        if not (0 <= vertex_id < self.num_vertices):
            raise IndexError("Vertex index out of bounds.")
        return np.nonzero(self._matrix[vertex_id])[0].tolist()
    
    @property
    def shape(self) -> int:
        return self._matrix.shape

    @property
    def num_vertices(self) -> int:
        """Returns the number of vertices."""
        return self._matrix.shape[0]

    @property
    def num_handles(self) -> int:
        """Returns the number of handles."""
        return self._matrix.shape[1]

    def __mul__(self, other: 'Weights') -> 'Weights':
        """
        Multiplies this weights matrix with another.
        This corresponds to standard matrix multiplication self @ other.
        The number of handles in self must equal the number of vertices in other.
        Args:
            other (Weights): The weights object to multiply with.
        Returns:
            Weights: A new Weights object containing the result.
        """
        if self.num_handles != other.num_vertices:
            raise ValueError(f"Incompatible shapes for multiplication: "
                             f"{self._matrix.shape} and {other._matrix.shape}")
        
        result_matrix = self._matrix @ other._matrix
        result = Weights(result_matrix.shape[0], result_matrix.shape[1])
        result._matrix = result_matrix
        return result

    def to_sparse(self, rows_indices: Union[List[int], None] = None) -> csr_matrix:
        """
        Exports the weights to a SciPy sparse matrix (CSR format).
        Original C++ Method : exportWeightsToEigen
        Args:
            rows_indices (list[int], optional): A list of vertex indices to include.
                                                If None, all vertices are included. Defaults to None.
        Returns:
            csr_matrix: The sparse matrix representation of the weights.
        """
        if rows_indices is not None:
            return csr_matrix(self._matrix[rows_indices, :])
        return csr_matrix(self._matrix)

    def identity_kronecker_product(self, rows_indices: Union[List[int], None] = None) -> csr_matrix:
        """
        Computes the Kronecker product of the weights matrix with a 3x3 identity matrix.
        This is useful for applying transformations to 3D vertices.
        Args:
            rows_indices (list[int], optional): A list of vertex indices to include.
                                                If None, all vertices are included. Defaults to None.
        Returns:
            csr_matrix: The resulting sparse matrix.
        """
        source_matrix = self._matrix
        if rows_indices is not None:
            source_matrix = self._matrix[rows_indices, :]
        
        # The C++ code builds this from triplets, which is equivalent to a Kronecker product
        # with the identity matrix.
        return kron(source_matrix, np.eye(3), format='csr')

    @property
    def T(self):
        return self._matrix.T