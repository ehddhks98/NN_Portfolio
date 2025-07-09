import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from model import AdaptivePortfolioOptimizer
from dataloader import create_dataloaders
from trainer import PortfolioTrainer
import os

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='TCN 기반 적응형 포트폴리오 최적화 훈련')
    
    # 스크립트 위치 기준으로 기본 데이터 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, 'data')
    
    # 데이터 관련 인수
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='데이터 디렉토리 경로')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lookback', type=int, default=252, help='과거 데이터 길이 (일)')
    parser.add_argument('--pred_horizon', type=int, default=21, help='예측 기간 (일)')
    parser.add_argument('--normalize', action='store_true', default=False, help='데이터 정규화 활성화')
    
    # 모델 관련 인수
    parser.add_argument('--num_assets', type=int, default=5, help='자산 수')
    parser.add_argument('--hidden_size', type=int, default=64, help='은닉층 크기')
    parser.add_argument('--num_channels', nargs='+', type=int, default=[32, 64, 128], help='TCN 채널 수 리스트')
    parser.add_argument('--kernel_size', type=int, default=3, help='컨볼루션 커널 크기')
    parser.add_argument('--dropout', type=float, default=0.2, help='드롭아웃 비율')
    
    # 훈련 관련 인수
    parser.add_argument('--num_epochs', type=int, default=100, help='훈련 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    parser.add_argument('--device', type=str, default='auto', help='계산 디바이스 (auto, cuda, cpu)')
    
    # 저장 관련 인수
    parser.add_argument('--save_path', type=str, default='checkpoints', help='모델 저장 경로')
    parser.add_argument('--save_interval', type=int, default=10, help='모델 저장 간격 (에포크)')
    
    return parser.parse_args()

def setup_device(device_arg):
    """디바이스 설정"""
    if device_arg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"사용 디바이스: {device}")
    if device.type == 'cuda':
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def create_model(args):
    """모델 생성"""
    model = AdaptivePortfolioOptimizer(
        num_assets=args.num_assets,
        hidden_size=args.hidden_size,
        num_channels=args.num_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout
    )
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== 모델 정보 ===")
    print(f"총 파라미터 수: {total_params:,}")
    print(f"훈련 가능한 파라미터 수: {trainable_params:,}")
    print(f"자산 수: {args.num_assets}")
    print(f"은닉층 크기: {args.hidden_size}")
    print(f"TCN 채널: {args.num_channels}")
    print(f"드롭아웃: {args.dropout}")
    
    return model

def create_data_loaders(args):
    """데이터 로더 생성"""
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
    
    print(f"\n=== 데이터 로딩 ===")
    print(f"데이터 디렉토리: {data_dir}")
    print(f"배치 크기: {args.batch_size}")
    print(f"과거 데이터 길이: {args.lookback}일")
    print(f"예측 기간: {args.pred_horizon}일")
    print(f"데이터 정규화: {'활성화' if args.normalize else '비활성화'}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, 
        batch_size=args.batch_size, 
        lookback=args.lookback, 
        pred_horizon=args.pred_horizon,
        normalize=args.normalize
    )
    
    print(f"훈련 배치 수: {len(train_loader)}")
    print(f"검증 배치 수: {len(val_loader)}")
    print(f"테스트 배치 수: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def train_portfolio_model(args):
    """포트폴리오 모델 훈련"""
    
    # 디바이스 설정
    device = setup_device(args.device)
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_data_loaders(args)
    
    # 모델 생성
    model = create_model(args)
    
    # 트레이너 생성
    trainer = PortfolioTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # 저장 디렉토리 생성 (절대 경로로 변환)
    save_path = Path(args.save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== 훈련 시작 ===")
    print(f"에포크 수: {args.num_epochs}")
    print(f"학습률: {args.learning_rate}")
    print(f"저장 경로: {save_path}")
    
    # 디렉토리 쓰기 권한 확인
    if not os.access(save_path, os.W_OK):
        raise PermissionError(f"저장 경로에 쓰기 권한이 없습니다: {save_path}")
    
    # 훈련 실행
    history = trainer.train(num_epochs=args.num_epochs, save_path=str(save_path))
    
    print(f"\n=== 훈련 완료 ===")
    print(f"최고 검증 손실: {min(history['val_losses']):.4f}")
    print(f"최고 검증 샤프 비율: {max(history['val_sharpe']):.4f}")
    print(f"최종 훈련 손실: {history['train_losses'][-1]:.4f}")
    print(f"최종 검증 손실: {history['val_losses'][-1]:.4f}")
    
    # 최종 테스트 실행
    print(f"\n=== 최종 테스트 실행 ===")
    test_results = trainer.test()
    
    print(f"테스트 손실: {test_results['avg_loss']:.4f}")
    print(f"테스트 샤프 비율: {test_results['avg_sharpe']:.4f}")
    print(f"테스트 수익률: {test_results['avg_returns']:.4f}")
    
    # 테스트 결과 저장
    try:
        torch.save(test_results, save_path / 'final_test_results.pth')
        print(f"✅ 테스트 결과 저장: {save_path / 'final_test_results.pth'}")
    except Exception as e:
        print(f"❌ 테스트 결과 저장 실패: {e}")
    
    # 훈련 히스토리 저장
    try:
        np.save(save_path / 'training_history.npy', history)
        print(f"✅ 훈련 히스토리 저장: {save_path / 'training_history.npy'}")
    except Exception as e:
        print(f"❌ 훈련 히스토리 저장 실패: {e}")
    
    print(f"\n결과가 {save_path}에 저장되었습니다.")
    
    return history, test_results

def main():
    """메인 함수"""
    try:
        # 인수 파싱
        args = parse_arguments()
        
        # 설정 출력
        print("=" * 60)
        print("TCN 기반 적응형 포트폴리오 최적화 훈련")
        print("=" * 60)
        
        # 훈련 실행
        history, test_results = train_portfolio_model(args)
        
        print("\n✅ 훈련이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()

