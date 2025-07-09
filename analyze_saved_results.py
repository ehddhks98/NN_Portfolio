import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def load_saved_results():
    """저장된 테스트 결과 로드"""
    results_path = '../../checkpoints/test_results.npy'
    
    if not os.path.exists(results_path):
        print(f"결과 파일을 찾을 수 없습니다: {results_path}")
        return None
    
    results = np.load(results_path, allow_pickle=True).item()
    
    print("=== 저장된 결과 정보 ===")
    print(f"테스트 손실: {results['test_loss']:.4f}")
    print(f"테스트 샤프 지수: {results['test_sharpe']:.4f}")
    print(f"예측 베타 형태: {results['test_betas'].shape}")
    print(f"포트폴리오 가중치 형태: {results['test_weights'].shape}")
    print(f"샤프 지수 비율 형태: {results['test_sharpe_ratios'].shape}")
    print(f"훈련 에포크 수: {len(results['train_losses'])}")
    
    return results

def plot_training_history(results, save_path='./saved_results_analysis'):
    """훈련 히스토리 시각화"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 손실 그래프
    epochs = range(1, len(results['train_losses']) + 1)
    ax1.plot(epochs, results['train_losses'], label='Train Loss', color='blue')
    ax1.plot(epochs, results['val_losses'], label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 샤프 지수 그래프
    ax2.plot(epochs, results['train_sharpe'], label='Train Sharpe', color='blue')
    ax2.plot(epochs, results['val_sharpe'], label='Val Sharpe', color='red')
    ax2.set_title('Training and Validation Sharpe Ratio')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()  # 메모리 절약을 위해 plt.show() 대신 plt.close() 사용

def analyze_beta_predictions(results, save_path='./saved_results_analysis'):
    """베타 예측 분석"""
    os.makedirs(save_path, exist_ok=True)
    
    betas = results['test_betas']
    asset_names = ['AAPL', 'MSFT', 'IBM', 'INTC', 'ORCL']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 각 자산별 베타 분포
    for i in range(5):
        ax = axes[i]
        sns.histplot(betas[:, i], bins=50, kde=True, ax=ax)
        ax.set_title(f'{asset_names[i]} Beta Distribution')
        ax.set_xlabel('Beta')
        ax.set_ylabel('Frequency')
        ax.axvline(betas[:, i].mean(), color='red', linestyle='--', 
                  label=f'Mean: {betas[:, i].mean():.3f}')
        ax.legend()
    
    # 베타 상관관계 히트맵
    ax = axes[5]
    beta_corr = np.corrcoef(betas.T)
    sns.heatmap(beta_corr, annot=True, cmap='coolwarm', center=0,
               xticklabels=asset_names, yticklabels=asset_names, ax=ax)
    ax.set_title('Beta Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'beta_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 베타 통계 출력
    print("\n=== 베타 예측 통계 ===")
    for i, name in enumerate(asset_names):
        print(f"{name}: 평균={betas[:, i].mean():.3f}, 표준편차={betas[:, i].std():.3f}")

def analyze_portfolio_weights(results, save_path='./saved_results_analysis'):
    """포트폴리오 가중치 분석"""
    os.makedirs(save_path, exist_ok=True)
    
    weights = results['test_weights']
    asset_names = ['AAPL', 'MSFT', 'IBM', 'INTC', 'ORCL']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 가중치 분포
    ax1 = axes[0, 0]
    weight_data = []
    for i in range(5):
        weight_data.extend([(asset_names[i], w) for w in weights[:, i]])
    
    weight_df = pd.DataFrame(weight_data, columns=['Asset', 'Weight'])
    sns.boxplot(data=weight_df, x='Asset', y='Weight', ax=ax1)
    ax1.set_title('Portfolio Weight Distribution')
    ax1.set_ylabel('Weight')
    
    # 가중치 시계열 (첫 100개 샘플)
    ax2 = axes[0, 1]
    samples_to_show = min(100, len(weights))
    for i in range(5):
        ax2.plot(weights[:samples_to_show, i], label=asset_names[i], alpha=0.7)
    ax2.set_title(f'Portfolio Weights Over Time (First {samples_to_show} samples)')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Weight')
    ax2.legend()
    ax2.grid(True)
    
    # 평균 가중치
    ax3 = axes[1, 0]
    mean_weights = np.mean(weights, axis=0)
    ax3.bar(asset_names, mean_weights)
    ax3.set_title('Average Portfolio Weights')
    ax3.set_ylabel('Average Weight')
    
    # 가중치 변동성
    ax4 = axes[1, 1]
    weight_std = np.std(weights, axis=0)
    ax4.bar(asset_names, weight_std)
    ax4.set_title('Weight Volatility (Standard Deviation)')
    ax4.set_ylabel('Standard Deviation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'weight_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 가중치 통계 출력
    print("\n=== 포트폴리오 가중치 통계 ===")
    for i, name in enumerate(asset_names):
        print(f"{name}: 평균={mean_weights[i]:.3f}, 표준편차={weight_std[i]:.3f}")

def analyze_sharpe_ratios(results, save_path='./saved_results_analysis'):
    """샤프 지수 분석"""
    os.makedirs(save_path, exist_ok=True)
    
    sharpe_ratios = results['test_sharpe_ratios']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 샤프 지수 분포
    ax1.hist(sharpe_ratios, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_title('Sharpe Ratio Distribution')
    ax1.set_xlabel('Sharpe Ratio')
    ax1.set_ylabel('Frequency')
    ax1.axvline(sharpe_ratios.mean(), color='red', linestyle='--',
               label=f'Mean: {sharpe_ratios.mean():.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 샤프 지수 시계열
    ax2.plot(sharpe_ratios[:100] if len(sharpe_ratios) > 100 else sharpe_ratios)
    ax2.set_title('Sharpe Ratio Over Time')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sharpe_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 샤프 지수 통계
    print("\n=== 샤프 지수 통계 ===")
    print(f"평균 샤프 지수: {sharpe_ratios.mean():.4f}")
    print(f"샤프 지수 표준편차: {sharpe_ratios.std():.4f}")
    print(f"최대 샤프 지수: {sharpe_ratios.max():.4f}")
    print(f"최소 샤프 지수: {sharpe_ratios.min():.4f}")
    print(f"양수 샤프 지수 비율: {100 * (sharpe_ratios > 0).mean():.1f}%")
    print(f"1 이상 샤프 지수 비율: {100 * (sharpe_ratios > 1).mean():.1f}%")

def generate_summary_report(results, save_path='./saved_results_analysis'):
    """종합 요약 보고서 생성"""
    os.makedirs(save_path, exist_ok=True)
    
    betas = results['test_betas']
    weights = results['test_weights']
    sharpe_ratios = results['test_sharpe_ratios']
    asset_names = ['AAPL', 'MSFT', 'IBM', 'INTC', 'ORCL']
    
    report = f"""
# Dual-Encoder TCN 포트폴리오 최적화 모델 - 저장된 결과 분석 보고서

## 모델 성능 요약
- **최종 테스트 손실**: {results['test_loss']:.4f}
- **최종 테스트 샤프 지수**: {results['test_sharpe']:.4f}
- **훈련 에포크 수**: {len(results['train_losses'])}
- **테스트 샘플 수**: {len(sharpe_ratios)}

## 베타 예측 결과
"""
    
    for i, name in enumerate(asset_names):
        report += f"- **{name}**: 평균 {betas[:, i].mean():.3f} ± {betas[:, i].std():.3f}\n"
    
    report += f"""
## 포트폴리오 가중치 분석
"""
    
    mean_weights = np.mean(weights, axis=0)
    for i, name in enumerate(asset_names):
        report += f"- **{name}**: 평균 {mean_weights[i]:.3f} ± {np.std(weights[:, i]):.3f}\n"
    
    report += f"""
## 성능 지표
- **평균 샤프 지수**: {sharpe_ratios.mean():.4f} ± {sharpe_ratios.std():.4f}
- **최대 샤프 지수**: {sharpe_ratios.max():.4f}
- **최소 샤프 지수**: {sharpe_ratios.min():.4f}
- **양수 샤프 지수 비율**: {100 * (sharpe_ratios > 0).mean():.1f}%
- **1 이상 샤프 지수 비율**: {100 * (sharpe_ratios > 1).mean():.1f}%

## 훈련 과정
- **최종 훈련 손실**: {results['train_losses'][-1]:.4f}
- **최종 검증 손실**: {results['val_losses'][-1]:.4f}
- **최종 훈련 샤프**: {results['train_sharpe'][-1]:.4f}
- **최종 검증 샤프**: {results['val_sharpe'][-1]:.4f}
"""
    
    # 보고서 저장
    with open(os.path.join(save_path, 'saved_results_summary.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n종합 보고서가 {os.path.join(save_path, 'saved_results_summary.md')}에 저장되었습니다.")

def main():
    """메인 함수"""
    print("저장된 결과 분석을 시작합니다...")
    
    # 결과 로드
    results = load_saved_results()
    if results is None:
        return
    
    # 분석 수행
    print("\n1. 훈련 히스토리 시각화...")
    plot_training_history(results)
    
    print("\n2. 베타 예측 분석...")
    analyze_beta_predictions(results)
    
    print("\n3. 포트폴리오 가중치 분석...")
    analyze_portfolio_weights(results)
    
    print("\n4. 샤프 지수 분석...")
    analyze_sharpe_ratios(results)
    
    print("\n5. 종합 보고서 생성...")
    generate_summary_report(results)
    
    print("\n✅ 모든 분석이 완료되었습니다!")
    print("결과는 './saved_results_analysis/' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 