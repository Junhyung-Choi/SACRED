import numpy as np
from dataclasses import dataclass

@dataclass
class MecStats:
    iterations: int = 0
    is_nan: bool = False
    linear_precision_error: float = 0.0

def compute_mec_coordinates(
    eta: np.ndarray,
    cage_vertices: np.ndarray,
    prior_masses: np.ndarray,
    max_iterations: int = 20,
    line_search_steps: int = 100,
    max_dw: float = 0.001,
    epsilon_termination: float = 1e-7,
    error_triggering_line_search: float = 1e-4
) -> tuple[np.ndarray, MecStats]:
    """
    임의의 폴리토프에 대한 최대 엔트로피 좌표(Maximum Entropy Coordinates)를 계산합니다.

    이 함수는 "Maximum Entropy Coordinates for Arbitrary Polytopes" (Hormann & Sukumar, 2008) 논문과
    C++ 원본 코드 `MEC::computeCoordinates`를 기반으로 구현되었습니다.

    Args:
        eta (np.ndarray): 좌표를 계산할 3D 지점 (shape: (3,)).
        cage_vertices (np.ndarray): 케이지의 정점 좌표 배열 (shape: (n_vertices, 3)).
        prior_masses (np.ndarray): 각 케이지 정점에 대한 사전 질량(prior) 배열 (shape: (n_vertices,)).
        max_iterations (int): 뉴턴-랩슨법의 최대 반복 횟수.
        line_search_steps (int): 라인 서치를 위한 스텝 수.
        max_dw (float): dLambda의 최대 크기를 제한하는 값.
        epsilon_termination (float): 수렴 판정을 위한 작은 값.
        error_triggering_line_search (float): 라인 서치를 트리거하는 오차 임계값.

    Returns:
        tuple[np.ndarray, MecStats]:
            - 계산된 가중치 배열 (shape: (n_vertices,)).
            - 계산 통계 정보를 담은 MecStats 객체.
    """
    num_vertices = cage_vertices.shape[0]
    assert num_vertices == prior_masses.shape[0], "정점 수와 사전 질량의 수가 일치해야 합니다."

    stats = MecStats()
    lambda_vec = np.zeros(3, dtype=np.float64)
    vi_bar = cage_vertices - eta

    for i in range(max_iterations):
        stats.iterations = i + 1

        # 1. Z, 그래디언트(gZ), 헤시안(HZ) 계산
        exp_term = np.exp(-np.dot(vi_bar, lambda_vec))
        Zi = prior_masses * exp_term
        Z = np.sum(Zi)

        if Z == 0 or np.isnan(Z) or np.isinf(Z):
            stats.is_nan = True
            break

        gZ = -np.sum(Zi[:, np.newaxis] * vi_bar, axis=0)
        HZ = np.einsum('i,ij,ik->jk', Zi, vi_bar, vi_bar)

        # 2. 목적 함수 F의 그래디언트(gF)와 헤시안(HF) 계산
        gF = gZ / Z
        HF = HZ / Z - np.outer(gF, gF)

        # 3. 뉴턴 탐색 방향(dLambda) 계산
        try:
            dLambda = -np.linalg.solve(HF, gF)
        except np.linalg.LinAlgError:
            dLambda = -np.linalg.pinv(HF) @ gF

        # 4. dLambda 크기 제한 (안정성)
        dw_term = prior_masses**2 * np.exp(-2.0 * np.dot(vi_bar, lambda_vec)) * \
                  (np.exp(-np.dot(vi_bar, dLambda)) - 1)**2
        DW = np.sqrt(np.sum(dw_term))
        if DW > max_dw:
            dLambda *= (max_dw / DW)

        # 5. 라인 서치 (필요 시)
        next_lambda = lambda_vec + dLambda
        error_current = np.abs(np.log(np.sum(prior_masses * np.exp(-np.dot(vi_bar, next_lambda)))))

        if np.isnan(error_current) or error_current > error_triggering_line_search:
            s = np.linspace(0, 1, line_search_steps)[1:]
            test_lambdas = lambda_vec + s[:, np.newaxis] * dLambda
            errors = np.abs(np.log(np.sum(prior_masses[:, np.newaxis] * np.exp(-vi_bar @ test_lambdas.T), axis=0)))
            
            valid_errors = errors[~np.isnan(errors)]
            if len(valid_errors) > 0:
                best_s_idx = np.nanargmin(errors)
                lambda_vec += s[best_s_idx] * dLambda
        else:
            lambda_vec = next_lambda

        if np.linalg.norm(gF) < epsilon_termination:
            break

    # 6. 최종 가중치 계산 및 정규화
    weights = prior_masses * np.exp(-np.dot(vi_bar, lambda_vec))
    sum_w = np.sum(weights)
    weights = weights / sum_w if sum_w != 0 else np.zeros(num_vertices)
    
    stats.linear_precision_error = np.linalg.norm(weights @ vi_bar)
    stats.is_nan = np.any(np.isnan(weights))
    if stats.is_nan:
        weights = np.nan_to_num(weights)

    return weights, stats