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