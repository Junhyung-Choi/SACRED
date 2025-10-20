import numpy as np

def compute_mvc_coordinates_3d(
    eta: np.ndarray,
    cage_triangles: np.ndarray,
    cage_vertices: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    3D 닫힌 삼각형 메시에 대한 평균값 좌표(Mean Value Coordinates)를 계산합니다.

    이 함수는 "Mean Value Coordinates for Closed Triangular Meshes" (Ju et al., 2005) 논문과
    C++ 원본 코드 `MVCoordinates::MVC3D::computeCoordinatesOriginalCode`를 기반으로 구현되었습니다.

    Args:
        eta (np.ndarray): 좌표를 계산할 3D 지점 (shape: (3,)).
        cage_triangles (np.ndarray): 케이지의 삼각형 면을 구성하는 정점 인덱스 배열 (shape: (n_triangles, 3)).
        cage_vertices (np.ndarray): 케이지의 정점 좌표 배열 (shape: (n_vertices, 3)).
        epsilon (float): 부동소수점 연산을 위한 작은 값.

    Returns:
        np.ndarray: 계산된 가중치 배열 (shape: (n_vertices,)).
    """
    num_vertices = cage_vertices.shape[0]
    num_triangles = cage_triangles.shape[0]

    weights = np.zeros(num_vertices, dtype=np.float64)

    # 1. eta에서 각 정점까지의 거리와 단위 벡터 계산
    d = np.linalg.norm(cage_vertices - eta, axis=1)

    # eta가 정점과 매우 가까운 경우 처리
    if np.any(d < epsilon):
        closest_vertex_idx = np.argmin(d)
        weights[closest_vertex_idx] = 1.0
        return weights

    u = (cage_vertices - eta) / d[:, np.newaxis]

    w_weights = np.zeros(num_vertices, dtype=np.float64)
    total_weight_sum = 0.0

    # 2. 각 삼각형을 순회하며 가중치 계산
    for t in range(num_triangles):
        vid = cage_triangles[t]
        
        # 3. 구면 삼각형의 변 길이와 각도 계산
        l = np.array([
            np.linalg.norm(u[vid[1]] - u[vid[2]]),
            np.linalg.norm(u[vid[2]] - u[vid[0]]),
            np.linalg.norm(u[vid[0]] - u[vid[1]])
        ])

        theta = 2.0 * np.arcsin(l / 2.0)

        h = np.sum(theta) / 2.0

        # 4. eta가 삼각형 위에 있는지 확인 (barycentric 좌표 사용)
        if np.pi - h < epsilon:
            w = np.sin(theta) * l[[2, 0, 1]] * l[[1, 2, 0]]
            sum_w = np.sum(w)
            if sum_w > epsilon:
                weights[vid] = w / sum_w
            return weights

        # 5. 일반적인 경우의 가중치 계산
        c = (2.0 * np.sin(h) * np.sin(h - theta)) / (np.sin(theta[[1, 2, 0]]) * np.sin(theta[[2, 0, 1]])) - 1.0

        # u[vid[0]], u[vid[1]], u[vid[2]] 기저의 부호 계산
        sign_basis = np.sign(np.dot(np.cross(u[vid[0]], u[vid[1]]), u[vid[2]]))
        if sign_basis == 0: sign_basis = 1.0

        s = sign_basis * np.sqrt(np.maximum(0.0, 1.0 - c * c))

        # eta가 삼각형과 동일 평면상에 있지만 외부에 있는 경우, 이 삼각형은 무시
        if np.any(np.abs(s) < epsilon):
            continue

        w = (theta - c[[1, 2, 0]] * theta[[2, 0, 1]] - c[[2, 0, 1]] * theta[[1, 2, 0]]) / \
            (2.0 * d[vid] * np.sin(theta[[1, 2, 0]]) * s[[2, 0, 1]])

        total_weight_sum += np.sum(w)
        w_weights[vid] += w

    # 6. 최종 가중치 정규화
    if abs(total_weight_sum) > epsilon:
        weights = w_weights / total_weight_sum

    return weights


def compute_mvc_weight_matrix(src_vertices, cage_vertices, cage_triangles, eps=1e-8, only_positive = False):
    """
    A corrected and more robust vectorized implementation of Mean Value Coordinates.

    This version fixes several issues from the original `MVC_weight_batch`:
    1.  Adds correct sign calculation based on the determinant.
    2.  Adds a check for coplanar-but-outside points to avoid division by zero.
    3.  Adds a pre-check for vertices that are on or very close to a cage vertex.

    Note: The logic for points exactly on a face is still a simplification. The original
    iterative code would `break` after finding the first face, but a true vectorized 
    equivalent is extremely complex. This version handles it more robustly but does not 
    replicate the `break` logic.

    Param:
    - src_vertices : (V, 3) --> [(v_x, v_y, v_z), ...]
    - cage_vertices : (N_cv, 3) --> [(v_x, v_y, v_z), ...]
    - cage_triangles : (N_cf, 3) -->  [(fv_i, fv_i+1, fv_i+2), ...] #fv: face_vertex
    """
    V = src_vertices.shape[0]
    Nc = cage_vertices.shape[0]
    F = cage_triangles.shape[0]

    weights_final = np.zeros((V, Nc))

    # --- 1. Pre-calculation for vertices on/near a cage vertex ---
    dists_to_cage_verts = np.linalg.norm(src_vertices[:, None, :] - cage_vertices[None, :, :], axis=2)
    min_dists = np.min(dists_to_cage_verts, axis=1)
    on_vertex_mask = min_dists < eps

    if np.any(on_vertex_mask):
        closest_cage_indices = np.argmin(dists_to_cage_verts[on_vertex_mask], axis=1)
        # Create a mapping from the index in the masked array to the original index
        original_indices = np.where(on_vertex_mask)[0]
        weights_final[original_indices, closest_cage_indices] = 1.0

    # --- Process only vertices that are not on a cage vertex ---
    processing_mask = ~on_vertex_mask
    if not np.any(processing_mask):
        return weights_final

    # Filter source vertices to only those that need full calculation
    src_v_proc = src_vertices[processing_mask]
    V_proc = src_v_proc.shape[0]

    # --- 2. Main MVC Calculation (Vectorized) ---
    i0, i1, i2 = cage_triangles[:, 0], cage_triangles[:, 1], cage_triangles[:, 2]
    P0, P1, P2 = cage_vertices[i0], cage_vertices[i1], cage_vertices[i2]

    X = src_v_proc[:, None, :]  # (V_proc, 1, 3)
    D0 = P0[None, :, :] - X      # (V_proc, F, 3)
    D1 = P1[None, :, :] - X
    D2 = P2[None, :, :] - X

    d0 = np.linalg.norm(D0, axis=-1)
    d1 = np.linalg.norm(D1, axis=-1)
    d2 = np.linalg.norm(D2, axis=-1)

    # Add epsilon here to prevent division by zero for unit vectors
    U0 = D0 / (d0[..., None] + eps)
    U1 = D1 / (d1[..., None] + eps)
    U2 = D2 / (d2[..., None] + eps)

    L0 = np.linalg.norm(U1 - U2, axis=-1)
    L1 = np.linalg.norm(U2 - U0, axis=-1)
    L2 = np.linalg.norm(U0 - U1, axis=-1)

    theta0 = 2 * np.arcsin(np.clip(L0 / 2.0, -1.0, 1.0))
    theta1 = 2 * np.arcsin(np.clip(L1 / 2.0, -1.0, 1.0))
    theta2 = 2 * np.arcsin(np.clip(L2 / 2.0, -1.0, 1.0))

    h = (theta0 + theta1 + theta2) / 2.0

    s_theta0 = np.sin(theta0)
    s_theta1 = np.sin(theta1)
    s_theta2 = np.sin(theta2)

    # --- 3. Fix: Sign calculation --- 
    # This is critical for correctness and was missing from the original batch version.
    mats_for_det = np.stack([U0, U1, U2], axis=3) # Shape: (V_proc, F, 3, 3)
    dets = np.linalg.det(mats_for_det)
    sign = np.sign(dets)
    sign[sign == 0] = 1.0 # Avoid sign being zero

    # --- 4. General Case and Coplanar/On-Face Checks ---
    W0 = np.zeros((V_proc, F))
    W1 = np.zeros((V_proc, F))
    W2 = np.zeros((V_proc, F))

    # Case A: Point is on the triangle face (barycentric-like weights)
    on_face_mask = np.abs(np.pi - h) < eps
    if np.any(on_face_mask):
        # Using a simplified barycentric-like weight for on-face cases
        w_face0 = s_theta0 * L1 * L2
        w_face1 = s_theta1 * L2 * L0
        w_face2 = s_theta2 * L0 * L1
        total_w = w_face0 + w_face1 + w_face2
        
        # Normalize and apply
        W0[on_face_mask] = w_face0[on_face_mask] / (total_w[on_face_mask] + eps)
        W1[on_face_mask] = w_face1[on_face_mask] / (total_w[on_face_mask] + eps)
        W2[on_face_mask] = w_face2[on_face_mask] / (total_w[on_face_mask] + eps)

    # Case B: General case
    general_mask = ~on_face_mask
    if np.any(general_mask):
        # Select only the data for the general case to avoid unnecessary calculations
        _h, _theta0, _theta1, _theta2 = h[general_mask], theta0[general_mask], theta1[general_mask], theta2[general_mask]
        _s_theta0, _s_theta1, _s_theta2 = s_theta0[general_mask], s_theta1[general_mask], s_theta2[general_mask]
        _sign = sign[general_mask]

        # Check for valid denominators before calculating c
        den_c0 = _s_theta1 * _s_theta2
        den_c1 = _s_theta2 * _s_theta0
        den_c2 = _s_theta0 * _s_theta1
        valid_c_mask = (den_c0 > eps) & (den_c1 > eps) & (den_c2 > eps)

        # Proceed only where c is valid
        if np.any(valid_c_mask):
            _h, _theta0, _theta1, _theta2 = _h[valid_c_mask], _theta0[valid_c_mask], _theta1[valid_c_mask], _theta2[valid_c_mask]
            _s_theta0, _s_theta1, _s_theta2 = _s_theta0[valid_c_mask], _s_theta1[valid_c_mask], _s_theta2[valid_c_mask]
            _sign = _sign[valid_c_mask]
            den_c0, den_c1, den_c2 = den_c0[valid_c_mask], den_c1[valid_c_mask], den_c2[valid_c_mask]

            c0 = (2 * np.sin(_h) * np.sin(_h - _theta0)) / den_c0 - 1
            c1 = (2 * np.sin(_h) * np.sin(_h - _theta1)) / den_c1 - 1
            c2 = (2 * np.sin(_h) * np.sin(_h - _theta2)) / den_c2 - 1

            s0 = _sign * np.sqrt(np.clip(1 - c0**2, 0, 1))
            s1 = _sign * np.sqrt(np.clip(1 - c1**2, 0, 1))
            s2 = _sign * np.sqrt(np.clip(1 - c2**2, 0, 1))

            # --- 5. Fix: Coplanar check ---
            # Ignore contributions if point is coplanar but outside triangle
            coplanar_mask = (np.abs(s0) < eps) | (np.abs(s1) < eps) | (np.abs(s2) < eps)
            valid_s_mask = ~coplanar_mask

            if np.any(valid_s_mask):
                # Filter everything again by the valid s mask
                _d0, _d1, _d2 = d0[general_mask][valid_c_mask][valid_s_mask], d1[general_mask][valid_c_mask][valid_s_mask], d2[general_mask][valid_c_mask][valid_s_mask]
                _theta0, _theta1, _theta2 = _theta0[valid_s_mask], _theta1[valid_s_mask], _theta2[valid_s_mask]
                _s_theta0, _s_theta1, _s_theta2 = _s_theta0[valid_s_mask], _s_theta1[valid_s_mask], _s_theta2[valid_s_mask]
                c0, c1, c2 = c0[valid_s_mask], c1[valid_s_mask], c2[valid_s_mask]
                s0, s1, s2 = s0[valid_s_mask], s1[valid_s_mask], s2[valid_s_mask]

                w_gen0 = (_theta0 - c1 * _theta2 - c2 * _theta1) / (2 * _d0 * s1 * _s_theta2 + eps)
                w_gen1 = (_theta1 - c2 * _theta0 - c0 * _theta2) / (2 * _d1 * s2 * _s_theta0 + eps)
                w_gen2 = (_theta2 - c0 * _theta1 - c1 * _theta0) / (2 * _d2 * s0 * _s_theta1 + eps)

                # Create a final mask to place results correctly
                final_mask = np.zeros_like(general_mask, dtype=bool)
                temp_mask = np.zeros_like(valid_c_mask, dtype=bool)
                temp_mask[valid_c_mask] = valid_s_mask
                final_mask[general_mask] = temp_mask

                W0[final_mask] = w_gen0
                W1[final_mask] = w_gen1
                W2[final_mask] = w_gen2

    # --- 6. Accumulate and Normalize ---
    W_vert = np.zeros((V_proc, Nc))
    np.add.at(W_vert, (slice(None), i0), W0)
    np.add.at(W_vert, (slice(None), i1), W1)
    np.add.at(W_vert, (slice(None), i2), W2)

    sum_along_rows = W_vert.sum(axis=1, keepdims=True)
    W_vert_normalized = W_vert / (sum_along_rows + eps)

    # Place the calculated weights back into the final matrix
    weights_final[processing_mask] = W_vert_normalized

    if only_positive:
        weights_final[weights_final < 0] = 0
        row_sums = weights_final.sum(axis=1, keepdims=True)
        np.divide(weights_final, row_sums, out=weights_final, where=row_sums != 0)

    return weights_final # (V, Nc)

