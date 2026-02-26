import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

try:
    # 1. 데이터 로드
    df = pd.read_csv("xgboost_gblinear_full_results_80.csv")

    # --- F1-Score 분석 ---
    df['F1_Rank'] = df.groupby('Dataset')['F1-Score'].rank(method='min', ascending=False)
    df['Num_Losses'] = df.groupby('Dataset')['Loss'].transform('nunique')
    df['F1_Points'] = (df['Num_Losses'] - df['F1_Rank']) + 1

    f1_agg = df.groupby('Loss').agg(
        Total_Points=('F1_Points', 'sum'),
        Average_Rank=('F1_Rank', 'mean')
    ).reset_index()

    f1_sorted = f1_agg.sort_values(by='Total_Points', ascending=False).reset_index(drop=True)
    print("--- F1-Score Analysis (Higher Score is Better) ---")
    print(f1_sorted.to_string(index=False))

    # --- matplotlib로 그리기 (F1) ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # 색상 (viridis_r 흉내)
    colors = sns.color_palette("viridis_r", len(f1_sorted))

    # 수평 막대 그리기
    bars = ax.barh(
        y=np.arange(len(f1_sorted)),
        width=f1_sorted['Total_Points'],
        color=colors,
        height=0.8
    )

    # y축 라벨 설정 (정렬된 순서)
    ax.set_yticks(np.arange(len(f1_sorted)))
    ax.set_yticklabels(f1_sorted['Loss'])

    # x축 범위 고정
    max_points = f1_sorted['Total_Points'].max()
    ax.set_xlim(0, max_points + 10)

    # 제목 및 라벨
    ax.set_title('Total Points by Loss Function (F1-Score)', fontsize=16, pad=20)
    ax.set_xlabel('Total Points (from F1-Score Ranks)', fontsize=12)
    ax.set_ylabel('Loss Function', fontsize=12)

    # 텍스트 라벨 (막대 끝 + 0.5)
    for idx, (bar, row) in enumerate(zip(bars, f1_sorted.itertuples())):
        value = row.Total_Points
        avg_rank = row.Average_Rank
        text = f"{value:.0f} Pts (Avg Rank: {avg_rank:.2f})"
        ax.text(
            x=value + 0.5,
            y=bar.get_y() + bar.get_height()/2,
            s=text,
            va='center',
            ha='left',
            fontsize=10,
            color='black'
        )

    # 여백 조정
    plt.tight_layout(pad=3.0)
    plt.savefig("loss_function_f1_total_points_80.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\nF1-Score Chart saved: loss_function_f1_total_points_80.png")


    # --- PR_AUC 분석 ---
    df['PR_AUC_Rank'] = df.groupby('Dataset')['PR_AUC'].rank(method='min', ascending=False)
    df['PR_AUC_Points'] = (df['Num_Losses'] - df['PR_AUC_Rank']) + 1

    pr_agg = df.groupby('Loss').agg(
        Total_Points=('PR_AUC_Points', 'sum'),
        Average_Rank=('PR_AUC_Rank', 'mean')
    ).reset_index()

    pr_sorted = pr_agg.sort_values(by='Total_Points', ascending=False).reset_index(drop=True)
    print("\n--- PR_AUC Analysis (Higher Score is Better) ---")
    print(pr_sorted.to_string(index=False))

    # --- matplotlib로 그리기 (PR_AUC) ---
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = sns.color_palette("plasma_r", len(pr_sorted))

    bars = ax.barh(
        y=np.arange(len(pr_sorted)),
        width=pr_sorted['Total_Points'],
        color=colors,
        height=0.8
    )

    ax.set_yticks(np.arange(len(pr_sorted)))
    ax.set_yticklabels(pr_sorted['Loss'])

    max_points = pr_sorted['Total_Points'].max()
    ax.set_xlim(0, max_points + 10)

    ax.set_title('Total Points by Loss Function (PR_AUC)', fontsize=16, pad=20)
    ax.set_xlabel('Total Points (from PR_AUC Ranks)', fontsize=12)
    ax.set_ylabel('Loss Function', fontsize=12)

    for idx, (bar, row) in enumerate(zip(bars, pr_sorted.itertuples())):
        value = row.Total_Points
        avg_rank = row.Average_Rank
        text = f"{value:.0f} Pts (Avg Rank: {avg_rank:.2f})"
        ax.text(
            x=value + 0.5,
            y=bar.get_y() + bar.get_height()/2,
            s=text,
            va='center',
            ha='left',
            fontsize=10,
            color='black'
        )

    plt.tight_layout(pad=3.0)
    plt.savefig("loss_function_pr_auc_total_points_80.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("PR_AUC Chart saved: loss_function_pr_auc_total_points_80.png")

except FileNotFoundError:
    print("Error: CSV file not found.")
except Exception as e:
    print(f"Error: {e}")