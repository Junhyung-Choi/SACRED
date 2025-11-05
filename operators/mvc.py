import numpy as np
import torch

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


def compute_mvc_weight_matrix(src_vertices, cage_vertices, cage_triangles, eps=1e-8):
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

    return weights_final # (V, Nc)

def compute_mvc_weight_matrix_torch(src_vertices: torch.Tensor, cage_vertices: torch.Tensor, cage_triangles: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean Value Coordinates (MVC) vectorized implementation using PyTorch.

    This version is a direct translation of the robust NumPy implementation
    to leverage PyTorch's tensor operations and potential GPU acceleration.

    Param:
    - src_vertices : (V, 3) --> torch.Tensor
    - cage_vertices : (Nc, 3) --> torch.Tensor
    - cage_triangles : (F, 3) --> torch.Tensor (indices, must be LongTensor)
    - eps : float
    """
    V = src_vertices.shape[0]
    Nc = cage_vertices.shape[0]
    F = cage_triangles.shape[0]
    dtype = src_vertices.dtype
    device = src_vertices.device

    weights_final = torch.zeros((V, Nc), dtype=dtype, device=device)

    # --- 1. Pre-calculation for vertices on/near a cage vertex ---
    # torch.linalg.norm(A - B, dim=-1)
    dists_to_cage_verts = torch.linalg.norm(src_vertices[:, None, :] - cage_vertices[None, :, :], dim=2)
    min_dists, _ = torch.min(dists_to_cage_verts, dim=1)
    on_vertex_mask = min_dists < eps

    if torch.any(on_vertex_mask):
        # argmin은 이미 계산된 min_dists를 사용
        closest_cage_indices = torch.argmin(dists_to_cage_verts[on_vertex_mask], dim=1)
        original_indices = torch.where(on_vertex_mask)[0]

        # PyTorch index_put_
        weights_final[original_indices, closest_cage_indices] = 1.0

    # --- Process only vertices that are not on a cage vertex ---
    processing_mask = ~on_vertex_mask
    if not torch.any(processing_mask):
        return weights_final

    # Filter source vertices
    src_v_proc = src_vertices[processing_mask]
    V_proc = src_v_proc.shape[0]
    
    # --- 2. Main MVC Calculation (Vectorized) ---
    i0, i1, i2 = cage_triangles[:, 0], cage_triangles[:, 1], cage_triangles[:, 2]
    
    # cage_vertices 인덱싱은 텐서로 변환
    P0, P1, P2 = cage_vertices[i0], cage_vertices[i1], cage_vertices[i2]

    X = src_v_proc[:, None, :]  # (V_proc, 1, 3)
    D0 = P0[None, :, :] - X      # (V_proc, F, 3)
    D1 = P1[None, :, :] - X
    D2 = P2[None, :, :] - X

    # torch.linalg.norm
    d0 = torch.linalg.norm(D0, dim=-1)
    d1 = torch.linalg.norm(D1, dim=-1)
    d2 = torch.linalg.norm(D2, dim=-1)

    # Add epsilon and expand dim for division
    U0 = D0 / (d0.unsqueeze(-1) + eps)
    U1 = D1 / (d1.unsqueeze(-1) + eps)
    U2 = D2 / (d2.unsqueeze(-1) + eps)

    L0 = torch.linalg.norm(U1 - U2, dim=-1)
    L1 = torch.linalg.norm(U2 - U0, dim=-1)
    L2 = torch.linalg.norm(U0 - U1, dim=-1)

    # torch.arcsin, torch.clamp (equivalent to np.clip)
    theta0 = 2 * torch.arcsin(torch.clamp(L0 / 2.0, -1.0, 1.0))
    theta1 = 2 * torch.arcsin(torch.clamp(L1 / 2.0, -1.0, 1.0))
    theta2 = 2 * torch.arcsin(torch.clamp(L2 / 2.0, -1.0, 1.0))

    h = (theta0 + theta1 + theta2) / 2.0

    s_theta0 = torch.sin(theta0)
    s_theta1 = torch.sin(theta1)
    s_theta2 = torch.sin(theta2)

    # --- 3. Fix: Sign calculation --- 
    mats_for_det = torch.stack([U0, U1, U2], dim=3) # Shape: (V_proc, F, 3, 3)
    # torch.linalg.det
    dets = torch.linalg.det(mats_for_det)
    sign = torch.sign(dets)
    # torch.where(condition, x, y)
    sign = torch.where(sign == 0, torch.tensor(1.0, dtype=dtype, device=device), sign) 

    # --- 4. General Case and Coplanar/On-Face Checks ---
    W0 = torch.zeros((V_proc, F), dtype=dtype, device=device)
    W1 = torch.zeros((V_proc, F), dtype=dtype, device=device)
    W2 = torch.zeros((V_proc, F), dtype=dtype, device=device)

    # Case A: Point is on the triangle face (barycentric-like weights)
    # torch.abs
    on_face_mask = torch.abs(torch.pi - h) < eps
    if torch.any(on_face_mask):
        w_face0 = s_theta0 * L1 * L2
        w_face1 = s_theta1 * L2 * L0
        w_face2 = s_theta2 * L0 * L1
        total_w = w_face0 + w_face1 + w_face2
        
        # Normalize and apply (마스크된 부분만 계산하고 할당)
        mask_indices = torch.where(on_face_mask)
        norm_w = total_w[mask_indices] + eps
        
        W0[mask_indices] = w_face0[mask_indices] / norm_w
        W1[mask_indices] = w_face1[mask_indices] / norm_w
        W2[mask_indices] = w_face2[mask_indices] / norm_w

    # Case B: General case
    general_mask = ~on_face_mask
    if torch.any(general_mask):
        # 마스킹된 데이터만 선택
        _h = h[general_mask]
        _theta0, _theta1, _theta2 = theta0[general_mask], theta1[general_mask], theta2[general_mask]
        _s_theta0, _s_theta1, _s_theta2 = s_theta0[general_mask], s_theta1[general_mask], s_theta2[general_mask]
        _sign = sign[general_mask]
        _d0, _d1, _d2 = d0[general_mask], d1[general_mask], d2[general_mask]

        # Check for valid denominators before calculating c
        den_c0 = _s_theta1 * _s_theta2
        den_c1 = _s_theta2 * _s_theta0
        den_c2 = _s_theta0 * _s_theta1
        valid_c_mask = (den_c0 > eps) & (den_c1 > eps) & (den_c2 > eps)

        # Proceed only where c is valid
        if torch.any(valid_c_mask):
            # c가 유효한 부분만 다시 필터링
            __h, __theta0, __theta1, __theta2 = _h[valid_c_mask], _theta0[valid_c_mask], _theta1[valid_c_mask], _theta2[valid_c_mask]
            __s_theta0, __s_theta1, __s_theta2 = _s_theta0[valid_c_mask], _s_theta1[valid_c_mask], _s_theta2[valid_c_mask]
            __d0, __d1, __d2 = _d0[valid_c_mask], _d1[valid_c_mask], _d2[valid_c_mask]
            __sign = _sign[valid_c_mask]
            den_c0, den_c1, den_c2 = den_c0[valid_c_mask], den_c1[valid_c_mask], den_c2[valid_c_mask]

            # c 계산
            c0 = (2 * torch.sin(__h) * torch.sin(__h - __theta0)) / den_c0 - 1
            c1 = (2 * torch.sin(__h) * torch.sin(__h - __theta1)) / den_c1 - 1
            c2 = (2 * torch.sin(__h) * torch.sin(__h - __theta2)) / den_c2 - 1

            # s 계산 (np.sqrt -> torch.sqrt, np.clip -> torch.clamp)
            s0 = __sign * torch.sqrt(torch.clamp(1 - c0**2, 0, 1))
            s1 = __sign * torch.sqrt(torch.clamp(1 - c1**2, 0, 1))
            s2 = __sign * torch.sqrt(torch.clamp(1 - c2**2, 0, 1))

            # --- 5. Fix: Coplanar check ---
            coplanar_mask = (torch.abs(s0) < eps) | (torch.abs(s1) < eps) | (torch.abs(s2) < eps)
            valid_s_mask = ~coplanar_mask

            if torch.any(valid_s_mask):
                # 최종적으로 유효한 부분만 필터링
                ___d0, ___d1, ___d2 = __d0[valid_s_mask], __d1[valid_s_mask], __d2[valid_s_mask]
                ___theta0, ___theta1, ___theta2 = __theta0[valid_s_mask], __theta1[valid_s_mask], __theta2[valid_s_mask]
                ___s_theta0, ___s_theta1, ___s_theta2 = __s_theta0[valid_s_mask], __s_theta1[valid_s_mask], __s_theta2[valid_s_mask]
                c0, c1, c2 = c0[valid_s_mask], c1[valid_s_mask], c2[valid_s_mask]
                s0, s1, s2 = s0[valid_s_mask], s1[valid_s_mask], s2[valid_s_mask]

                # w_gen 계산
                w_gen0 = (___theta0 - c1 * ___theta2 - c2 * ___theta1) / (2 * ___d0 * s1 * ___s_theta2 + eps)
                w_gen1 = (___theta1 - c2 * ___theta0 - c0 * ___theta2) / (2 * ___d1 * s2 * ___s_theta0 + eps)
                w_gen2 = (___theta2 - c0 * ___theta1 - c1 * ___theta0) / (2 * ___d2 * s0 * ___s_theta1 + eps)

                # 최종 결과 마스크 생성 및 할당
                # general_mask & valid_c_mask & valid_s_mask 에 해당하는 인덱스를 추출
                final_indices_tuple = torch.where(general_mask)[0], torch.where(general_mask)[1]
                
                # valid_c_mask와 valid_s_mask를 결합하여 최종 위치를 찾습니다.
                final_mask_in_full = torch.zeros_like(general_mask, dtype=torch.bool, device=device)
                temp_mask = torch.zeros_like(valid_c_mask, dtype=torch.bool, device=device)
                temp_mask[valid_c_mask] = valid_s_mask
                
                # 1. general_mask가 True인 모든 인덱스 (V_proc 차원, F 차원)를 튜플로 가져옵니다.
                general_indices_V = torch.where(general_mask)[0] # (General_True_Count,)
                general_indices_F = torch.where(general_mask)[1] # (General_True_Count,)
                final_mask_indices = (general_indices_V[temp_mask], general_indices_F[temp_mask])
                                
                W0[final_mask_indices] = w_gen0
                W1[final_mask_indices] = w_gen1
                W2[final_mask_indices] = w_gen2


    # --- 6. Accumulate and Normalize (torch.scatter_add_) ---
    W_vert = torch.zeros((V_proc, Nc), dtype=dtype, device=device)
    
    # i0, i1, i2 (F,)를 (V_proc, F)로 확장하여 scatter_add의 index로 사용
    i0_exp = i0[None, :].expand(V_proc, F)
    i1_exp = i1[None, :].expand(V_proc, F)
    i2_exp = i2[None, :].expand(V_proc, F)

    # scatter_add_(dim=1, index, src)
    W_vert.scatter_add_(1, i0_exp, W0)
    W_vert.scatter_add_(1, i1_exp, W1)
    W_vert.scatter_add_(1, i2_exp, W2)

    # Normalize
    sum_along_rows = W_vert.sum(dim=1, keepdim=True)
    W_vert_normalized = W_vert / (sum_along_rows + eps)

    # Place the calculated weights back into the final matrix
    # torch.where(processing_mask)로 원본 인덱스 추출
    original_proc_indices = torch.where(processing_mask)[0]
    weights_final[original_proc_indices, :] = W_vert_normalized

    return weights_final