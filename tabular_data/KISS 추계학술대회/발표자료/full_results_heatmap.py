import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

# 고해상도 설정
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

try:
    df = pd.read_csv("xgboost_gblinear_full_results_100.csv")

    # -------------------------------------------------
    # 1. Borda-Count 포인트 & 평균 랭킹 계산
    # -------------------------------------------------
    df['F1_Rank'] = df.groupby('Dataset')['F1-Score'].rank(method='min', ascending=False)
    df['Num_Losses'] = df.groupby('Dataset')['Loss'].transform('nunique')
    df['F1_Points'] = (df['Num_Losses'] - df['F1_Rank']) + 1

    f1_agg = df.groupby('Loss').agg(
        F1_Total_Points=('F1_Points', 'sum'),
        F1_Avg_Rank=('F1_Rank', 'mean')
    ).reset_index()

    df['PR_AUC_Rank'] = df.groupby('Dataset')['PR_AUC'].rank(method='min', ascending=False)
    df['PR_AUC_Points'] = (df['Num_Losses'] - df['PR_AUC_Rank']) + 1

    pr_agg = df.groupby('Loss').agg(
        PR_Total_Points=('PR_AUC_Points', 'sum'),
        PR_Avg_Rank=('PR_AUC_Rank', 'mean')
    ).reset_index()

    rank_summary = pd.merge(f1_agg, pr_agg, on='Loss')   # 여기서 Loss는 CSV 그대로

    # -------------------------------------------------
    # 2. 데이터셋 IR 순서
    # -------------------------------------------------
    dataset_ir_order = {
        'glass': 5, 'yeast': 9, 'steel_faults': 6, 'german_credit': 1,
        'telco_churn': 2, 'page_blocks': 10, 'credit_card_default': 3,
        'bank_marketing': 4, 'secom': 7, 'aps_failure': 8,
        'credit_card_fraud': 11
    }

    # -------------------------------------------------
    # 3. 히트맵용 피벗 테이블 (행 순서도 PLWCE 우선 정렬!)
    # -------------------------------------------------
    def create_heatmap_data(metric_col):
        pivot = df.pivot_table(
            index='Loss',
            columns='Dataset',
            values=metric_col,
            aggfunc='first'
        )
        ordered_cols = sorted(pivot.columns, key=lambda x: dataset_ir_order.get(x, 999))
        pivot = pivot[ordered_cols]

        rank_col = 'F1_Avg_Rank' if 'F1' in metric_col else 'PR_Avg_Rank'

        # === 여기부터 수정: 동일한 정렬 기준 사용 ===
        def sort_key(row):
            loss_name = row.name.upper()  # index가 Loss
            avg_rank = row[rank_col]
            if 'PLWCE' in loss_name:
                return (avg_rank, 0)
            else:
                return (avg_rank, 1)

        # rank_summary에서 정렬 순서 추출
        sorted_index = rank_summary.set_index('Loss').apply(sort_key, axis=1).sort_values().index
        pivot = pivot.loc[sorted_index]
        # === 여기까지 수정 끝 ===

        # 보기 좋게 _ → ' '
        pivot.index = pivot.index.str.replace('_', ' ')
        pivot.columns = pivot.columns.str.replace('_', ' ').str.title()
        return pivot

    f1_pivot = create_heatmap_data('F1_Rank')
    pr_pivot = create_heatmap_data('PR_AUC_Rank')

    # -------------------------------------------------
    # 4. y축 라벨 + 색상 (동일한 정렬 기준으로 색상 적용)
    # -------------------------------------------------
    def get_ylabels_and_colors(pivot, metric):
        labels = []
        colors = []

        rank_col = 'F1_Avg_Rank' if metric == 'F1' else 'PR_Avg_Rank'
        pts_col  = 'F1_Total_Points' if metric == 'F1' else 'PR_Total_Points'

        # 위에서 정의한 정렬 기준과 동일하게 순위 추출
        def sort_key(row):
            loss_name = row.name.upper()
            avg_rank = row[rank_col]
            return (avg_rank, 0 if 'PLWCE' in loss_name else 1)

        sorted_index = rank_summary.set_index('Loss').apply(sort_key, axis=1).sort_values().index

        top1_2_loss = [name.replace('_', ' ') for name in sorted_index[:2]]
        top3_4_loss = [name.replace('_', ' ') for name in sorted_index[2:4]]

        for loss_raw in pivot.index:
            original_loss = loss_raw.replace(' ', '_')
            row = rank_summary[rank_summary['Loss'] == original_loss].iloc[0]

            avg = row[rank_col]
            pts = row[pts_col]
            label = f"{loss_raw}\n(Avg Rank: {avg:.2f}, {int(pts)} pt)"
            labels.append(label)

            # 색상: 현재 표시되는 이름 기준
            if loss_raw in top1_2_loss:
                colors.append('#D32F2F')    # 1~2등 빨강
            elif loss_raw in top3_4_loss:
                colors.append('#1976D2')    # 3~4등 파랑
            else:
                colors.append('black')

        return labels, colors
    
    f1_ylabels, f1_ycolors = get_ylabels_and_colors(f1_pivot, 'F1')
    pr_ylabels, pr_ycolors = get_ylabels_and_colors(pr_pivot, 'PR')

    # -------------------------------------------------
    # 5. 컬러맵
    # -------------------------------------------------
    colors_gradient = [
        '#1B5E20', '#2E7D32', '#43A047', '#66BB6A', '#81C784',
        '#FFEB3B', '#FFC107', '#FF9800', '#FF5722', '#D32F2F', '#B71C1C'
    ]
    cmap = LinearSegmentedColormap.from_list('custom_strong', colors_gradient, N=100)

    # -------------------------------------------------
    # 6. 히트맵 그리기
    # -------------------------------------------------
    def draw_heatmap(pivot, ylabels, ycolors, metric_name, pdf):
        fig, ax = plt.subplots(figsize=(14, 10))

        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                       vmin=1, vmax=8, interpolation='bilinear')

        cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label('Rank (Lower is Better)', rotation=270, labelpad=25,
                       fontsize=13, fontweight='bold')
        cbar.ax.tick_params(labelsize=11)

        # 셀 안 숫자
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                text_color = 'white' if val <= 2 or val >= 7 else 'black'
                bg_color = 'black' if val <= 2 else 'darkred' if val >= 7 else 'white'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=text_color,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color,
                                  alpha=0.4, edgecolor='none'))

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(ylabels, fontsize=9.5, fontweight='bold')

        # y축 색상 적용
        for tick, color in zip(ax.get_yticklabels(), ycolors):
            tick.set_color(color)

        ax.set_title(f'Loss Function Rankings: {metric_name} Performance\n'
                     '(Datasets ordered by Imbalance Ratio: Low → High)',
                     fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Dataset (Increasing Imbalance →)', fontsize=13,
                      fontweight='bold', labelpad=10)
        ax.set_ylabel('Loss Function', fontsize=13, fontweight='bold')

        # 그리드
        ax.set_xticks(np.arange(len(pivot.columns)) - .5, minor=True)
        ax.set_yticks(np.arange(len(pivot)) - .5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5, alpha=0.5)
        ax.tick_params(which="minor", size=0)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # -------------------------------------------------
    # 7. PDF 저장
    # -------------------------------------------------
    with PdfPages('f1_heatmap.pdf') as pdf:
        draw_heatmap(f1_pivot, f1_ylabels, f1_ycolors, 'F1-Score', pdf)
        print("F1-Score 히트맵 저장: f1_heatmap.pdf")

    with PdfPages('pr_auc_heatmap.pdf') as pdf:
        draw_heatmap(pr_pivot, pr_ylabels, pr_ycolors, 'PR-AUC', pdf)
        print("PR-AUC 히트맵 저장: pr_auc_heatmap.pdf")

    print("\n모든 작업 완료! PDF 2개 생성됨")

except FileNotFoundError:
    print("Error: CSV 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"Error: {e}")