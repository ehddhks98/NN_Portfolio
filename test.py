import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model import AdaptivePortfolioOptimizer
from dataloader import create_dataloaders
from trainer import PortfolioTrainer

class PortfolioEvaluator:
    """TCN 기반 적응형 포트폴리오 최적화 모델 평가기"""
    
    def __init__(self, model_path, device='auto'):
        """
        Args:
            model_path: 훈련된 모델 체크포인트 경로
            device: 계산 디바이스
        """
        self.device = self._setup_device(device)
        self.model_path = Path(model_path)
        self.model = None
        self.checkpoint_info = None
        
    def _setup_device(self, device_arg):
        """디바이스 설정"""
        if device_arg == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_arg)
        return device
        
    def load_model(self, num_assets=5, hidden_size=64, num_channels=[32, 64, 128], 
                   kernel_size=3, dropout=0.2):
        """훈련된 모델 로드"""
        
        # 모델 생성
        self.model = AdaptivePortfolioOptimizer(
            num_assets=num_assets,
            hidden_size=hidden_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 체크포인트 로드
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 체크포인트 정보 저장
        self.checkpoint_info = checkpoint
        
        print(f"✅ 모델 로드 완료: {self.model_path}")
        print(f"훈련 에포크: {checkpoint.get('epoch', 'N/A')}")
        print(f"훈련 손실: {checkpoint.get('train_loss', 'N/A'):.4f}")
        print(f"검증 손실: {checkpoint.get('val_loss', 'N/A'):.4f}")
        print(f"훈련 샤프 비율: {checkpoint.get('train_sharpe', 'N/A'):.4f}")
        print(f"검증 샤프 비율: {checkpoint.get('val_sharpe', 'N/A'):.4f}")
        
        return self.model
    
    def evaluate_on_test_set(self, test_loader):
        """테스트 세트에서 모델 평가"""
        if self.model is None:
            raise ValueError("먼저 모델을 로드해야 합니다.")
            
        self.model.eval()
        
        # 결과 수집을 위한 리스트
        all_results = {
            'betas': [],
            'weights': [],
            'sharpe_ratios': [],  # 실현 샤프 비율
            'realized_returns': [],
            'expected_returns': [],
            'cov_matrices': [],
            'losses': []
        }
        
        total_loss = 0
        total_sharpe = 0
        total_returns = 0
        num_batches = 0
        
        print("테스트 세트 평가 중...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="테스트 진행"):
                # 데이터를 디바이스로 이동
                asset_data = batch['asset_data'].to(self.device)
                common_data = batch['common_data'].to(self.device)
                future_returns = batch['future_returns'].to(self.device)
                
                # 모델 순전파
                results = self.model(asset_data, common_data, future_returns)
                
                # 결과 저장
                all_results['betas'].append(results['betas'].cpu())
                all_results['weights'].append(results['weights'].cpu())
                all_results['sharpe_ratios'].append(results['realized_sharpe_ratio'].cpu())
                all_results['realized_returns'].append(results['realized_return'].cpu())
                all_results['expected_returns'].append(results['expected_returns'].cpu())
                all_results['cov_matrices'].append(results['cov_matrix'].cpu())
                all_results['losses'].append(results['loss'].cpu())
                
                # 통계 업데이트
                total_loss += results['loss'].item()
                total_sharpe += results['realized_sharpe_ratio'].mean().item()
                total_returns += results['realized_return'].mean().item()
                num_batches += 1
        
        # 결과 결합
        for key in all_results:
            if key != 'cov_matrices':  # 공분산 행렬은 3D이므로 별도 처리
                all_results[key] = torch.cat(all_results[key], dim=0).numpy()
            else:
                all_results[key] = torch.cat(all_results[key], dim=0).numpy()
        
        # 평균 성능 계산
        performance = {
            'avg_loss': total_loss / num_batches,
            'avg_sharpe': total_sharpe / num_batches,
            'avg_returns': total_returns / num_batches,
            'total_samples': all_results['betas'].shape[0]
        }
        
        print(f"\n=== 테스트 결과 요약 ===")
        print(f"총 샘플 수: {performance['total_samples']:,}")
        print(f"평균 손실: {performance['avg_loss']:.4f}")
        print(f"평균 샤프 비율: {performance['avg_sharpe']:.4f}")
        print(f"평균 실현 수익률: {performance['avg_returns']:.4f}")
        
        return all_results, performance
    
    def analyze_portfolio_weights(self, weights, save_path):
        """포트폴리오 가중치 분석"""
        asset_names = ['AAPL', 'MSFT', 'IBM', 'INTC', 'ORCL']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # 1. 가중치 분포 (박스플롯)
        ax = axes[0]
        weight_data = []
        for i, asset in enumerate(asset_names):
            weight_data.extend([(asset, w) for w in weights[:, i]])
        
        weight_df = pd.DataFrame(weight_data, columns=['자산', '가중치'])
        sns.boxplot(data=weight_df, x='자산', y='가중치', ax=ax)
        ax.set_title('포트폴리오 가중치 분포', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. 가중치 시계열 (처음 200개 샘플)
        ax = axes[1]
        n_samples = min(200, weights.shape[0])
        for i, asset in enumerate(asset_names):
            ax.plot(weights[:n_samples, i], label=asset, alpha=0.8, linewidth=2)
        ax.set_title(f'포트폴리오 가중치 시계열 (처음 {n_samples}개 샘플)', fontsize=14, fontweight='bold')
        ax.set_xlabel('샘플')
        ax.set_ylabel('가중치')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 가중치 상관관계 히트맵
        ax = axes[2]
        weight_corr = np.corrcoef(weights.T)
        sns.heatmap(weight_corr, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=asset_names, yticklabels=asset_names, ax=ax)
        ax.set_title('가중치 상관관계 행렬', fontsize=14, fontweight='bold')
        
        # 4. 가중치 변동성
        ax = axes[3]
        weight_std = np.std(weights, axis=0)
        bars = ax.bar(asset_names, weight_std, color='steelblue', alpha=0.7)
        ax.set_title('가중치 변동성 (표준편차)', fontsize=14, fontweight='bold')
        ax.set_ylabel('표준편차')
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, std in zip(bars, weight_std):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{std:.3f}', ha='center', va='bottom')
        
        # 5. 가중치 합 검증
        ax = axes[4]
        weight_sums = np.sum(weights, axis=1)
        ax.hist(weight_sums, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='목표값 (1.0)')
        ax.set_title('가중치 합 분포', fontsize=14, fontweight='bold')
        ax.set_xlabel('가중치 합')
        ax.set_ylabel('빈도')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 평균 가중치
        ax = axes[5]
        mean_weights = np.mean(weights, axis=0)
        bars = ax.bar(asset_names, mean_weights, color='orange', alpha=0.7)
        ax.set_title('평균 포트폴리오 가중치', fontsize=14, fontweight='bold')
        ax.set_ylabel('평균 가중치')
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, weight in zip(bars, mean_weights):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{weight:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path / 'portfolio_weights_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return weight_df, weight_corr, weight_std
    
    def analyze_beta_estimates(self, betas, save_path):
        """베타 추정값 분석"""
        asset_names = ['AAPL', 'MSFT', 'IBM', 'INTC', 'ORCL']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # 1. 베타 분포 (각 자산별)
        for i in range(5):
            ax = axes[i]
            sns.histplot(betas[:, i], bins=50, kde=True, ax=ax, alpha=0.7)
            ax.set_title(f'{asset_names[i]} 베타 분포', fontsize=14, fontweight='bold')
            ax.set_xlabel('베타')
            ax.set_ylabel('빈도')
            ax.axvline(betas[:, i].mean(), color='red', linestyle='--', 
                      label=f'평균: {betas[:, i].mean():.3f}')
            ax.axvline(1.0, color='gray', linestyle='-', alpha=0.5, label='시장 베타 (1.0)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. 베타 상관관계 히트맵
        ax = axes[5]
        beta_corr = np.corrcoef(betas.T)
        sns.heatmap(beta_corr, annot=True, cmap='coolwarm', center=0,
                   xticklabels=asset_names, yticklabels=asset_names, ax=ax)
        ax.set_title('베타 상관관계 행렬', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path / 'beta_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return beta_corr
    
    def analyze_performance(self, sharpe_ratios, realized_returns, expected_returns, save_path):
        """성능 분석"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 샤프 비율 분포
        ax = axes[0, 0]
        sns.histplot(sharpe_ratios, bins=50, kde=True, ax=ax, alpha=0.7, color='blue')
        ax.set_title('샤프 비율 분포', fontsize=14, fontweight='bold')
        ax.set_xlabel('샤프 비율')
        ax.set_ylabel('빈도')
        ax.axvline(sharpe_ratios.mean(), color='red', linestyle='--',
                  label=f'평균: {sharpe_ratios.mean():.3f}')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5, label='기준선 (0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 실현 수익률 분포
        ax = axes[0, 1]
        sns.histplot(realized_returns, bins=50, kde=True, ax=ax, alpha=0.7, color='green')
        ax.set_title('실현 수익률 분포', fontsize=14, fontweight='bold')
        ax.set_xlabel('실현 수익률')
        ax.set_ylabel('빈도')
        ax.axvline(realized_returns.mean(), color='red', linestyle='--',
                  label=f'평균: {realized_returns.mean():.4f}')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5, label='기준선 (0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 기대 수익률 vs 실현 수익률
        ax = axes[1, 0]
        expected_portfolio_returns = np.mean(expected_returns, axis=1)
        ax.scatter(expected_portfolio_returns, realized_returns, alpha=0.6, color='purple')
        ax.plot([min(expected_portfolio_returns), max(expected_portfolio_returns)],
               [min(expected_portfolio_returns), max(expected_portfolio_returns)],
               'r--', label='완벽한 예측')
        ax.set_xlabel('기대 수익률')
        ax.set_ylabel('실현 수익률')
        ax.set_title('기대 수익률 vs 실현 수익률', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 상관계수 계산
        correlation = np.corrcoef(expected_portfolio_returns, realized_returns)[0, 1]
        ax.text(0.05, 0.95, f'상관계수: {correlation:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        # 4. 성능 시계열 (처음 500개 샘플)
        ax = axes[1, 1]
        n_samples = min(500, len(sharpe_ratios))
        ax.plot(sharpe_ratios[:n_samples], label='샤프 비율', alpha=0.8, color='blue')
        ax.plot(realized_returns[:n_samples] * 10, label='실현 수익률 (×10)', alpha=0.8, color='green')
        ax.set_title(f'성과 지표 시계열 (처음 {n_samples}개 샘플)', fontsize=14, fontweight='bold')
        ax.set_xlabel('샘플')
        ax.set_ylabel('값')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return correlation
    
    def generate_comprehensive_report(self, results, performance, save_path):
        """종합 보고서 생성"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        print(f"\n=== 종합 분석 시작 ===")
        
        # 1. 포트폴리오 가중치 분석
        print("1. 포트폴리오 가중치 분석 중...")
        weight_df, weight_corr, weight_std = self.analyze_portfolio_weights(results['weights'], save_path)
        
        # 2. 베타 추정값 분석
        print("2. 베타 추정값 분석 중...")
        beta_corr = self.analyze_beta_estimates(results['betas'], save_path)
        
        # 3. 성능 분석
        print("3. 성능 분석 중...")
        correlation = self.analyze_performance(
            results['sharpe_ratios'], results['realized_returns'], 
            results['expected_returns'], save_path
        )
        
        # 4. 요약 통계 저장
        summary_stats = {
            'performance': performance,
            'weight_stats': {
                'mean': np.mean(results['weights'], axis=0).tolist(),
                'std': weight_std.tolist(),
                'correlation': weight_corr.tolist()
            },
            'beta_stats': {
                'mean': np.mean(results['betas'], axis=0).tolist(),
                'std': np.std(results['betas'], axis=0).tolist(),
                'correlation': beta_corr.tolist()
            },
            'sharpe_stats': {
                'mean': float(np.mean(results['sharpe_ratios'])),
                'std': float(np.std(results['sharpe_ratios'])),
                'min': float(np.min(results['sharpe_ratios'])),
                'max': float(np.max(results['sharpe_ratios']))
            },
            'return_correlation': float(correlation),
            'checkpoint_info': self.checkpoint_info
        }
        
        # JSON으로 저장
        import json
        with open(save_path / 'summary_report.json', 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        # 전체 결과 저장
        np.savez_compressed(save_path / 'detailed_results.npz', **results)
        
        print(f"\n✅ 종합 분석 완료! 결과가 {save_path}에 저장되었습니다.")
        print(f"   - 포트폴리오 가중치 분석: portfolio_weights_analysis.png")
        print(f"   - 베타 분석: beta_analysis.png")
        print(f"   - 성능 분석: performance_analysis.png")
        print(f"   - 요약 보고서: summary_report.json")
        print(f"   - 상세 결과: detailed_results.npz")
        
        return summary_stats

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='TCN 포트폴리오 모델 테스트 및 평가')
    
    # 스크립트 위치 기준으로 기본 데이터 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, 'data')
    
    # 모델 관련
    parser.add_argument('--model_path', type=str, required=True,
                       help='훈련된 모델 체크포인트 경로')
    parser.add_argument('--num_assets', type=int, default=5,
                       help='자산 수')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='은닉층 크기')
    parser.add_argument('--num_channels', nargs='+', type=int, default=[32, 64, 128],
                       help='TCN 채널 수 리스트')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='컨볼루션 커널 크기')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='드롭아웃 비율')
    
    # 데이터 관련
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                       help='데이터 디렉토리 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='테스트 배치 크기')
    parser.add_argument('--lookback', type=int, default=252,
                       help='과거 데이터 길이 (일)')
    parser.add_argument('--pred_horizon', type=int, default=21,
                       help='예측 기간 (일)')
    
    # 기타
    parser.add_argument('--device', type=str, default='auto',
                       help='계산 디바이스 (auto, cuda, cpu)')
    parser.add_argument('--save_path', type=str, default='results',
                       help='결과 저장 경로')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    try:
        # 인수 파싱
        args = parse_arguments()
        
        print("=" * 60)
        print("TCN 기반 적응형 포트폴리오 최적화 모델 테스트")
        print("=" * 60)
        
        # 평가기 생성
        evaluator = PortfolioEvaluator(args.model_path, args.device)
        
        # 모델 로드
        evaluator.load_model(
            num_assets=args.num_assets,
            hidden_size=args.hidden_size,
            num_channels=args.num_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout
        )
        
        # 데이터 로더 생성
        print(f"\n=== 데이터 로딩 ===")
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            lookback=args.lookback,
            pred_horizon=args.pred_horizon
        )
        
        print(f"테스트 배치 수: {len(test_loader)}")
        
        # 모델 평가
        results, performance = evaluator.evaluate_on_test_set(test_loader)
        
        # 종합 보고서 생성
        summary_stats = evaluator.generate_comprehensive_report(
            results, performance, args.save_path
        )
        
        print("\n✅ 테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main() 