# Transformer (ViT-Lite) Subsection — 진행 상황 (2026-07-12)

## 실험 세팅 (이번 파일럿)
- **Model**: ViT-Lite (patch=4, embed_dim=192, depth=6, heads=3, CIFAR 32×32 대응) — `vit_cifar.py`
- **Dataset**: CIFAR10-LT
- **IR**: 100만 (가장 극단적인 케이스, 파일럿 우선순위)
- **Epoch**: **40** (표준 200에서 축소 — 시간 제약 때문, 아래 TODO 참고)
- **Seed**: 42, 43 (n=2, 표준 5에서 축소)
- **Loss**: ce, wce, cb, lwce, plwce, es_ace(=ES-LWCE), focal (7종, pwce/sqce는 논문 미채택 baseline이라 제외)

⚠️ **위 축소 설정은 ResNet32 baseline(200 epoch, 5 seed, 승호 실험)과 학습 예산이 다르므로,
절대 수치로 두 아키텍처를 직접 비교하지 않음. 어디까지나 파일럿(pilot) 결과.**

## 결과 (IR=100, ViT-Lite, 40 epoch, n=2)

| Loss | F1-Macro (mean±std) | Few-Acc | 순위(F1) |
|---|---|---|---|
| lwce | 0.3843±0.0038 | 0.1894 | 1위 |
| wce | 0.3791±0.0033 | 0.2616 | 2위 |
| plwce | 0.3783±0.0048 | 0.1762 | 3위 |
| focal | 0.3767±0.0058 | 0.1797 | 4위 |
| ce | 0.3752±0.0000 | 0.1684 | 5위 |
| es_ace (ES-LWCE) | 0.3739±0.0038 | 0.1750 | 6위 |
| cb | 0.3652±0.0005 | 0.1953 | 7위 |

## 정직한 해석
- **LWCE가 이 파일럿에서 최고 성능** — 제안 방법 계열이 우수하다는 논문 스토리와 부합.
- **WCE 붕괴 패턴 미재현**: ResNet32(200 epoch)에서 IR=100 기준 WCE가 8개 중 꼴찌(F1=0.3935)였던 것과 달리,
  이번 ViT(40 epoch) 파일럿에서는 WCE가 2위, Few-Acc는 오히려 1위. **"weight explosion이 Transformer에서도
  동일하게 재현된다"는 주장은 이 데이터로 뒷받침되지 않음.**
- **ES-LWCE(es_ace)가 기대만큼 우위를 보이지 않음** — F1 기준 하위권(6위). 40 epoch라는 짧은 학습
  예산에서는 아직 그 효과가 충분히 드러나지 않았을 가능성.
- 가능한 원인: (1) 학습 예산 부족 — weight explosion 효과가 누적되려면 더 많은 epoch 필요할 수 있음,
  (2) Transformer의 학습 동역학(normalization, gradient flow)이 CNN과 달라 같은 현상이 다르게 나타날 수 있음.

## TODO (다음 단계 — 우선순위 순)
1. **[최우선] IR=100을 표준 조건(epoch=200, seed=5)으로 재실행** — ResNet32 baseline과 완전히
   동일한 학습 예산으로 맞춰서, 지금 나온 "패턴 미재현" 결과가 진짜인지 재검증
2. IR=10, IR=50도 ViT로 확장 (현재 IR=100만 완료)
3. ResNet32 + es_ace(ES-LWCE)를 200 epoch/5 seed로 별도 실행 — 기존 40개 baseline엔 es_ace가
   없어서, 같은 조건으로 채워 넣어야 ResNet32 vs ViT 비교표 완성 가능
4. **이름 통일**: 코드/노트북에서는 `es_ace`, 논문 Section 3.2에서는 **ES-LWCE**로 표기됨 —
   최종 논문/코드 어디서든 하나로 통일 필요 

## 참고 — 2차 논문 피드백 반영 (별도 트랙, 아직 미착수)
Transformer 실험과 별개로, 2026-07-05 피드백의 Major Comments 3가지는 아직 반영 안 됨:
- LWCE "bounded" 표현을 개별 weight/ratio로 구분해서 정밀화
- ES-LWCE의 존재 이유를 "boundedness"가 아니라 "희귀 클래스 안정성 + 사용자 조절 가능한 cap"으로 재설명
- PLWCE의 hyperparameter 여부 명확히 구분 (LWCE=parameter-free, PLWCE=1-parameter)

## 논문 추가 내용 (Section 4)
\begin{table}[t]
\centering
\caption{ViT-Lite pilot results on CIFAR-10-LT, IR=100 (40 epochs, n=2 seeds).}
\label{tab:vit_ir100}
\begin{tabular}{lccc}
\toprule
Loss & F1-Macro & Few-Acc & Rank (F1) \\
\midrule
LWCE   & 0.3843 $\pm$ 0.0038 & 0.1894 & 1 \\
WCE    & 0.3791 $\pm$ 0.0033 & \textbf{0.2616} & 2 \\
PLWCE  & 0.3783 $\pm$ 0.0048 & 0.1762 & 3 \\
Focal  & 0.3767 $\pm$ 0.0058 & 0.1797 & 4 \\
CE     & 0.3752 $\pm$ 0.0000 & 0.1684 & 5 \\
ES-LWCE& 0.3739 $\pm$ 0.0038 & 0.1750 & 6 \\
CB     & 0.3652 $\pm$ 0.0005 & 0.1953 & 7 \\
\bottomrule
\end{tabular}
\end{table}
