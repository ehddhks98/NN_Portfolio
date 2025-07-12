import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from model.w_mlp_model import AdaptivePortfolioOptimizer

class PortfolioTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader,                
                learning_rate=1e-3, device='cuda'):
        """
        TCN 기반 적응형 포트폴리오 최적화 트레이너
        
        Args:
            model: AdaptivePortfolioOptimizer 모델
            train_loader, val_loader, test_loader: 데이터 로더들
            learning_rate: 학습률
            device: 계산 디바이스
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # 옵티마이저 및 스케줄러
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=15, verbose=True, min_lr=1e-6
        )
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        self.train_sharpe = []
        self.val_sharpe = []
        self.train_returns = []
        self.val_returns = []
        
    def train_epoch(self):
        """한 에포크 훈련"""
        self.model.train()
        epoch_loss = 0
        epoch_sharpe = 0
        epoch_returns = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="훈련 중")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 데이터를 디바이스로 이동
            asset_data = batch['asset_data'].to(self.device)
            common_data = batch['common_data'].to(self.device)
            future_returns = batch['future_returns'].to(self.device)
            
            # 순전파: 모델이 모든 계산을 내부적으로 수행
            self.optimizer.zero_grad()
            
            # 모델 순전파 - 딕셔너리 형태로 결과 반환
            results = self.model(asset_data, common_data, future_returns)
            
            # 결과 추출
            loss = results['loss']
            theoretical_sharpe = results['sharpe_ratio']
            realized_sharpe = results['realized_sharpe_ratio']
            weights = results['weights']
            betas = results['betas']
            realized_returns = results['realized_return']
            
            # 역전파 및 최적화
            loss.backward()
            
            # 강화된 그래디언트 클리핑 (폭발 방지)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            # 그래디언트 값 클리핑
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
            
            self.optimizer.step()
            
            # 통계 업데이트
            epoch_loss += loss.item()
            epoch_sharpe += realized_sharpe.mean().item()  # 실현 샤프 비율 사용
            epoch_returns += realized_returns.mean().item()
            num_batches += 1
            
            # 진행률 표시 업데이트
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'TheorSharpe': f'{theoretical_sharpe.mean().item():.4f}',
                'RealSharpe': f'{realized_sharpe.mean().item():.4f}',
                'Return': f'{realized_returns.mean().item():.4f}'
            })
            
            # 첫 번째 배치에서 상세 정보 출력 (디버깅용)
            if batch_idx == 0 and len(self.train_losses) == 0:
                print(f"\n=== 첫 번째 훈련 배치 상세 정보 ===")
                print(f"배치 크기: {asset_data.shape[0]}")
                print(f"자산 수: {asset_data.shape[1] if len(asset_data.shape) > 2 else '확인 불가'}")
                print(f"시계열 길이: {asset_data.shape[2] if len(asset_data.shape) > 2 else '확인 불가'}")
                print(f"손실: {loss.item():.6f}")
                print(f"샤프 비율: {realized_sharpe.mean().item():.6f}")
                print(f"실현 수익률: {realized_returns.mean().item():.6f}")
                
                # 베타 값 상세 분석
                print(f"\n--- 베타 값 분석 ---")
                print(f"베타 평균: {betas.mean().item():.6f}")
                print(f"베타 표준편차: {betas.std().item():.6f}")
                print(f"베타 범위: [{betas.min().item():.4f}, {betas.max().item():.4f}]")
                print(f"베타 중앙값: {betas.median().item():.6f}")
                
                # 각 자산별 베타 (첫 번째 배치 샘플)
                if betas.shape[0] > 0:
                    first_sample_betas = betas[0]  # 첫 번째 샘플의 베타들
                    print(f"첫 번째 샘플 베타들: {[f'{b.item():.4f}' for b in first_sample_betas]}")
                
                # 기대수익률 분석
                expected_rets = results['expected_returns']
                print(f"\n--- 기대수익률 분석 ---")
                print(f"기대수익률 평균: {expected_rets.mean().item():.6f}")
                print(f"기대수익률 범위: [{expected_rets.min().item():.6f}, {expected_rets.max().item():.6f}]")
                if expected_rets.shape[0] > 0:
                    first_sample_rets = expected_rets[0]
                    print(f"첫 번째 샘플 기대수익률: {[f'{r.item():.6f}' for r in first_sample_rets]}")
                
                print(f"\n--- 포트폴리오 가중치 분석 ---")
                print(f"가중치 합: {weights.sum(dim=1).mean().item():.6f}")
                print(f"가중치 범위: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
                if weights.shape[0] > 0:
                    first_sample_weights = weights[0]
                    print(f"첫 번째 샘플 가중치: {[f'{w.item():.4f}' for w in first_sample_weights]}")
                
                # 시장 데이터 확인
                print(f"\n--- 시장 데이터 분석 ---")
                mkt_rf = common_data[:, :, 0]  # 시장 초과 수익률
                rf = common_data[:, :, 1]      # 무위험 수익률
                print(f"시장 초과수익률 평균: {mkt_rf.mean().item():.6f}")
                print(f"무위험 이자율 평균: {rf.mean().item():.6f}")
                print(f"시장 초과수익률 범위: [{mkt_rf.min().item():.6f}, {mkt_rf.max().item():.6f}]")
        
        return epoch_loss / num_batches, epoch_sharpe / num_batches, epoch_returns / num_batches
    
    def validate(self):
        """검증"""
        self.model.eval()
        epoch_loss = 0
        epoch_sharpe = 0
        epoch_returns = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="검증 중")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 데이터를 디바이스로 이동
                asset_data = batch['asset_data'].to(self.device)
                common_data = batch['common_data'].to(self.device)
                future_returns = batch['future_returns'].to(self.device)
                
                # 순전파: 모델이 모든 계산을 내부적으로 수행
                results = self.model(asset_data, common_data, future_returns)
                
                # 결과 추출
                loss = results['loss']
                theoretical_sharpe = results['sharpe_ratio']
                realized_sharpe = results['realized_sharpe_ratio']
                weights = results['weights']
                realized_returns = results['realized_return']
                
                # 통계 업데이트
                epoch_loss += loss.item()
                epoch_sharpe += realized_sharpe.mean().item()  # 실현 샤프 비율 사용
                epoch_returns += realized_returns.mean().item()
                num_batches += 1
                
                # 진행률 표시 업데이트
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Sharpe': f'{realized_sharpe.mean().item():.4f}',
                    'Return': f'{realized_returns.mean().item():.4f}'
                })
                
                # 첫 번째 배치에서 상세 정보 출력 (디버깅용)
                if batch_idx == 0:
                    betas = results['betas']
                    expected_rets = results['expected_returns']
                    
                    print(f"\n=== 첫 번째 검증 배치 정보 ===")
                    print(f"손실: {loss.item():.6f}")
                    print(f"샤프 비율: {realized_sharpe.mean().item():.6f}")
                    print(f"실현 수익률: {realized_returns.mean().item():.6f}")
                    print(f"가중치 합: {weights.sum(dim=1).mean().item():.6f}")
                    
                    print(f"\n--- 검증 베타 값 분석 ---")
                    print(f"베타 평균: {betas.mean().item():.6f}")
                    print(f"베타 범위: [{betas.min().item():.4f}, {betas.max().item():.4f}]")
                    print(f"기대수익률 평균: {expected_rets.mean().item():.6f}")
                    print(f"기대수익률 범위: [{expected_rets.min().item():.6f}, {expected_rets.max().item():.6f}]")
        
        return epoch_loss / num_batches, epoch_sharpe / num_batches, epoch_returns / num_batches
    
    def test(self):
        """테스트"""
        self.model.eval()
        epoch_loss = 0
        epoch_sharpe = 0
        epoch_returns = 0
        num_batches = 0
        
        # 결과 수집을 위한 리스트
        all_results = {
            'betas': [],
            'weights': [],
            'sharpe_ratios': [],
            'realized_returns': [],
            'expected_returns': [],
            'cov_matrices': []
        }
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="테스트 중")
            
            for batch in progress_bar:
                # 데이터를 디바이스로 이동
                asset_data = batch['asset_data'].to(self.device)
                common_data = batch['common_data'].to(self.device)
                future_returns = batch['future_returns'].to(self.device)
                
                # 순전파: 모델이 모든 계산을 내부적으로 수행
                results = self.model(asset_data, common_data, future_returns)
                
                # 결과 저장
                all_results['betas'].append(results['betas'].cpu())
                all_results['weights'].append(results['weights'].cpu())
                all_results['sharpe_ratios'].append(results['realized_sharpe_ratio'].cpu())
                all_results['realized_returns'].append(results['realized_return'].cpu())
                all_results['expected_returns'].append(results['expected_returns'].cpu())
                all_results['cov_matrices'].append(results['cov_matrix'].cpu())
                
                # 통계 업데이트
                epoch_loss += results['loss'].item()
                epoch_sharpe += results['realized_sharpe_ratio'].mean().item()  # 실현 샤프 비율 사용
                epoch_returns += results['realized_return'].mean().item()
                num_batches += 1
                
                # 진행률 표시 업데이트
                progress_bar.set_postfix({
                    'Loss': f'{results["loss"].item():.4f}',
                    'Sharpe': f'{results["realized_sharpe_ratio"].mean().item():.4f}',
                    'Return': f'{results["realized_return"].mean().item():.4f}'
                })
        
        # 결과 결합
        for key in all_results:
            all_results[key] = torch.cat(all_results[key], dim=0)
        
        return {
            'avg_loss': epoch_loss / num_batches,
            'avg_sharpe': epoch_sharpe / num_batches,
            'avg_returns': epoch_returns / num_batches,
            'results': all_results
        }
    
    def train(self, num_epochs, save_path='./checkpoints'):
        """전체 훈련 과정"""
        # 절대 경로로 변환하여 경로 문제 방지
        save_path = os.path.abspath(save_path)
        os.makedirs(save_path, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"모델 저장 경로: {save_path}")
        
        print(f"포트폴리오 최적화 훈련 시작: {num_epochs} 에포크")
        print(f"디바이스: {self.device}")
        print(f"자산 수: {self.model.num_assets}")
        
        for epoch in range(num_epochs):
            print(f"\n에포크 {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # 훈련
            train_loss, train_sharpe, train_returns = self.train_epoch()
            
            # 검증
            val_loss, val_sharpe, val_returns = self.validate()
            
            # 학습률 조정
            self.scheduler.step(val_loss)
            
            # 기록 저장
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_sharpe.append(train_sharpe)
            self.val_sharpe.append(val_sharpe)
            self.train_returns.append(train_returns)
            self.val_returns.append(val_returns)
            
            # 수익률 차이 분석 (더 정밀하게)
            return_diff = abs(train_returns - val_returns)
            print(f"수익률 차이: {return_diff:.6f} ({'정상' if return_diff < 0.001 else '주의'})")
            
            # 결과 출력
            print(f"훈련 - 손실: {train_loss:.4f}, 샤프: {train_sharpe:.4f}, 수익률: {train_returns:.4f}")
            print(f"검증 - 손실: {val_loss:.4f}, 샤프: {val_sharpe:.4f}, 수익률: {val_returns:.4f}")
            print(f"현재 학습률: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 샤프 비율 차이 분석
            sharpe_diff = abs(train_sharpe - val_sharpe)
            print(f"샤프 비율 차이: {sharpe_diff:.6f} ({'정상' if sharpe_diff < 0.05 else '주의'})")
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = os.path.join(save_path, 'best_model.pth')
                
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_sharpe': train_sharpe,
                        'val_sharpe': val_sharpe,
                        'train_returns': train_returns,
                        'val_returns': val_returns
                    }, model_save_path)
                    print(f"✅ 새로운 최고 모델 저장: {model_save_path}")
                except Exception as e:
                    print(f"❌ 모델 저장 실패: {e}")
                    print(f"저장 경로: {model_save_path}")
        
        # 훈련 완료 후 그래프 저장
        self.plot_training_history(save_path)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_sharpe': self.train_sharpe,
            'val_sharpe': self.val_sharpe,
            'train_returns': self.train_returns,
            'val_returns': self.val_returns
        }
    
    def plot_training_history(self, save_path):
        """훈련 히스토리 플롯 (손실, 샤프 비율, 실현 수익률)"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 손실 그래프
        axes[0].plot(self.train_losses, label='훈련 손실', color='blue', linewidth=2)
        axes[0].plot(self.val_losses, label='검증 손실', color='red', linewidth=2)
        axes[0].set_title('훈련 및 검증 손실', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('에포크', fontsize=12)
        axes[0].set_ylabel('손실', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 샤프 비율 그래프
        axes[1].plot(self.train_sharpe, label='훈련 샤프 비율', color='blue', linewidth=2)
        axes[1].plot(self.val_sharpe, label='검증 샤프 비율', color='red', linewidth=2)
        axes[1].set_title('훈련 및 검증 샤프 비율', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('에포크', fontsize=12)
        axes[1].set_ylabel('샤프 비율', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 실현 수익률 그래프
        axes[2].plot(self.train_returns, label='훈련 수익률', color='blue', linewidth=2)
        if hasattr(self, 'val_returns') and len(self.val_returns) > 0:
            axes[2].plot(self.val_returns, label='검증 수익률', color='red', linewidth=2)
        axes[2].set_title('훈련 및 검증 실현 수익률', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('에포크', fontsize=12)
        axes[2].set_ylabel('실현 수익률', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 추가적인 성능 분석 그래프
        self.plot_performance_analysis(save_path)
    
    def plot_performance_analysis(self, save_path):
        """성능 분석 그래프"""
        if len(self.train_losses) < 2:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 손실 대 샤프 비율 상관관계
        ax1.scatter(self.train_losses, self.train_sharpe, alpha=0.7, color='blue', label='훈련')
        ax1.scatter(self.val_losses, self.val_sharpe, alpha=0.7, color='red', label='검증')
        ax1.set_xlabel('손실')
        ax1.set_ylabel('샤프 비율')
        ax1.set_title('손실 vs 샤프 비율')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 수렴성 분석 (손실의 변화율)
        if len(self.train_losses) > 1:
            train_loss_diff = np.diff(self.train_losses)
            val_loss_diff = np.diff(self.val_losses)
            ax2.plot(train_loss_diff, label='훈련 손실 변화율', color='blue')
            ax2.plot(val_loss_diff, label='검증 손실 변화율', color='red')
            ax2.set_xlabel('에포크')
            ax2.set_ylabel('손실 변화율')
            ax2.set_title('손실 수렴성 분석')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 3. 샤프 비율 분포
        all_sharpe = self.train_sharpe + self.val_sharpe
        ax3.hist(all_sharpe, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=np.mean(all_sharpe), color='red', linestyle='--', label=f'평균: {np.mean(all_sharpe):.3f}')
        ax3.set_xlabel('샤프 비율')
        ax3.set_ylabel('빈도')
        ax3.set_title('샤프 비율 분포')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 훈련-검증 성능 격차
        if len(self.train_sharpe) > 0 and len(self.val_sharpe) > 0:
            sharpe_gap = [t - v for t, v in zip(self.train_sharpe, self.val_sharpe)]
            ax4.plot(sharpe_gap, color='purple', linewidth=2)
            ax4.set_xlabel('에포크')
            ax4.set_ylabel('샤프 비율 격차 (훈련 - 검증)')
            ax4.set_title('과적합 분석')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # 과적합 경고
            if len(sharpe_gap) > 10:
                recent_gap = np.mean(sharpe_gap[-10:])
                if recent_gap > 0.1:
                    ax4.text(0.5, 0.9, '⚠️ 과적합 가능성', transform=ax4.transAxes, 
                            fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()



