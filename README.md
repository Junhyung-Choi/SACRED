# SACRED: Skeleton And Cage Real-time Editor/Deformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

"Real-Time Deformation with Coupled Cages and Skeletons" 논문의 Python 구현 프로젝트입니다. 스켈레톤과 케이지를 함께 사용하여 3D 모델을 실시간으로 편집하고 변형하는 라이브러리 개발을 목표로 합니다.

## 📖 소개 (Introduction)

이 프로젝트는 "Real-Time Deformation with Coupled Cages and Skeletons" 논문을 Python으로 구현하는 것을 목표로 합니다. 원본 C++ 코드를 기반으로 하며, aPyOpenGL 플랫폼에서 작동하도록 개발될 예정입니다.

## ✨ 주요 기능 (Features)

> ⚠️ 아직 개발 초기 단계이며, 아래는 구현 예정인 기능 목록입니다.

본 라이브러리는 실시간 변형에 필요한 핵심 로직, 즉 **전처리 및 행렬 연산**에 집중합니다.

*   모델 및 케이지 데이터 전처리
*   스켈레톤-케이지 결합을 위한 행렬 계산
*   실시간 변형을 위한 행렬 연산 파이프라인

> 편집 UI는 타겟 플랫폼인 [aPyOpenGL](https://github.com/seokhyeonhong/aPyOpenGL)에서 제공하는 기능을 활용할 예정입니다.

## 🚀 사용법 (Usage)

라이브러리 사용 예제는 추후 이 섹션에 추가될 예정입니다.

```python
# 예시
import sacred

# ...
```

## 🤝 기여 (Contributing)

기여는 언제나 환영입니다! Pull Request를 보내주시거나 이슈를 등록해주세요.

## 📝 라이선스 (License)

본 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## 📚 참고 자료 (References)

### 원본 논문

Corda, F., Thiery, J. M., Livesu, M., Puppo, E., Boubekeur, T., & Scateni, R. (2020). Real-Time Deformation with Coupled Cages and Skeletons. *Computer Graphics Forum*, *39*(6), 19–32. https://doi.org/10.1111/cgf.13900

```bibtex
@article{doi:10.1111/cgf.13900,
  author  = {Corda, Fabrizio and Thiery, Jean Marc and Livesu, Marco and Puppo, Enrico and Boubekeur, Tamy and Scateni, Riccardo},
  title   = {Real-Time Deformation with Coupled Cages and Skeletons},
  journal = {Computer Graphics Forum},
  volume  = {39},
  number  = {6},
  pages   = {19-32},
  doi     = {10.1111/cgf.13900},
  year    = {2020}
}
```

### 관련 링크

*   **Original Paper:** [https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13900](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13900)
*   **Original Code (C++):** [https://github.com/cordafab/SuperCages/tree/master](https://github.com/cordafab/SuperCages/tree/master)
*   **Target Platform:** [https://github.com/seokhyeonhong/aPyOpenGL](https://github.com/seokhyeonhong/aPyOpenGL)
