import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors

# 고해상도 설정
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 1. Confusion Matrix 데이터 정의
cm_data = np.array([[990, 0],
                    [10,  0]])

# 2. 레이블 및 주석(annotation) 텍스트 정의
labels = ['Healthy', 'Cancer']
annot_data = [
    [f"True Negative\n(TN)\n\n990", f"False Positive\n(FP)\n\n0"],
    [f"False Negative\n(FN)\n\n10", f"True Positive\n(TP)\n\n0"]
]

# 3. 각 셀의 의미에 따른 색상 인덱스
# 0 = Correct (TN, TP) -> Green
# 1 = Incorrect (FP, Type I) -> Yellow
# 2 = Critical Incorrect (FN, Type II) -> Red
color_indices = np.array([[0, 1],  # TN(0), FP(1)
                        [2, 0]]) # FN(2), TP(0)

# 4. 커스텀 색상 맵 생성 (조금 더 차분한 학술적 색상)
cmap = matplotlib.colors.ListedColormap(['#E8F5E9', '#FFF9C4', '#FFCDD2'])

# 5. 시각화
plt.figure(figsize=(10, 8))

# 5-1. Seaborn Heatmap을 사용하여 셀의 배경색과 그리드를 그립니다.
ax = sns.heatmap(color_indices,
                cmap=cmap,
                cbar=False,
                xticklabels=labels,
                yticklabels=labels,
                linewidths=3,
                linecolor='white',
                annot=False)

# 5-2. 수동으로 텍스트 주석을 추가합니다.
# FN (False Negative) 셀만 붉은색 텍스트로 강조
text_colors = [['#2E7D32', '#424242'],       # TN (진한 초록), FP (검정)
             ['#C62828', '#424242']]         # FN (진한 빨강), TP (검정)

for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.5, annot_data[i][j],
                ha='center', va='center',
                fontsize=18,
                fontweight='bold',
                color=text_colors[i][j])

# 6. 제목 및 레이블 설정
ax.set_title('Confusion Matrix: "99% Accurate" Model', fontsize=20, pad=20, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15, rotation=0, va='center')

# 7. 파일로 저장
output_filename = 'confusion_matrix_v3.pdf'
plt.savefig(output_filename, bbox_inches='tight', dpi=300)

print(f"학술용 Confusion matrix가 '{output_filename}' 파일로 저장되었습니다.")
plt.close()