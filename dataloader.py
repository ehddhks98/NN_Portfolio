import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FinancialDataset(Dataset):
    """
    TCN 기반 포트폴리오 최적화를 위한 금융 데이터셋
    
    데이터 구조:
    - 자산 데이터: 5개 종목(AAPL, IBM, INTC, MSFT, ORCL)의 일일 수익률
    - 시장 데이터: Fama-French 3요인 모델 (Mkt-RF, RF)
    - 시계열 슬라이딩 윈도우 방식으로 과거 데이터와 미래 수익률 생성
    """
    
    def __init__(self, data_dir, lookback=252, pred_horizon=21, split='train', normalize=False, rebalancing_frequency='monthly'):
        """
        Args:
            data_dir (str): 데이터 디렉토리 경로
            lookback (int): 과거 데이터 길이 (일) - 기본값 252일 (1년)
            pred_horizon (int): 예측 기간 (일) - 기본값 21일 (1개월)
            split (str): 데이터 분할 ('train', 'val', 'test')
            normalize (bool): 데이터 정규화 여부
            rebalancing_frequency (str): 리밸런싱 주기 ('daily' 또는 'monthly')
        """
        self.data_dir = Path(data_dir)
        self.lookback = lookback
        self.pred_horizon = pred_horizon
        self.split = split
        self.normalize = normalize
        self.rebalancing_frequency = rebalancing_frequency
        
        # 자산 명칭 정의
        self.asset_names = ['AAPL', 'IBM', 'INTC', 'MSFT', 'ORCL']
        self.num_assets = len(self.asset_names)
        
        # 정규화를 위한 스케일러
        self.asset_scaler = StandardScaler() if normalize else None
        self.market_scaler = StandardScaler() if normalize else None
        
        # 데이터 로드 및 전처리
        self._load_and_preprocess_data()
        
        # 데이터 분할 (시계열 특성을 고려한 순차적 분할)
        self._create_data_split()
        
        if split == 'train':  # 훈련 데이터에서만 상세 정보 출력
            self._print_dataset_info()
            self._print_rebalancing_info()
    
    def _load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print(f"\n=== {self.split.upper()} 데이터 로딩 시작 ===")
        
        # 데이터 디렉토리 존재 확인
        if not self.data_dir.exists():
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {self.data_dir}")
        
        # 자산 데이터 로드
        asset_dfs = self._load_asset_data()
        
        # Fama-French 데이터 로드
        ff_data = self._load_fama_french_data()
        
        # 데이터 병합 및 정리
        self._merge_and_clean_data(asset_dfs, ff_data)
        
        # 데이터 품질 검증
        self._validate_data_quality()
        
        print(f"✅ {self.split.upper()} 데이터 로딩 완료")
    
    def _load_asset_data(self):
        """자산 데이터 로드"""
        asset_files = [f'{name}_daily_technical_data.csv' for name in self.asset_names]
        asset_dfs = []
        
        for i, (name, file) in enumerate(zip(self.asset_names, asset_files)):
            file_path = self.data_dir / file
            
            # 파일 존재 확인
            if not file_path.exists():
                raise FileNotFoundError(f"자산 데이터 파일을 찾을 수 없습니다: {file_path}")
            
            try:
                # CSV 파일 로드
                df = pd.read_csv(file_path)
                
                # 필수 컬럼 존재 확인
                required_cols = ['Date', 'Close']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"{name} 데이터에 필수 컬럼이 없습니다: {missing_cols}")
                
                # 날짜 형식 변환
                df['Date'] = pd.to_datetime(df['Date'])
                
                # 수익률 계산
                df['Returns'] = df['Close'].pct_change()
                
                # 데이터 정렬
                df = df.sort_values('Date').reset_index(drop=True)
                
                asset_dfs.append(df)
                
                if self.split == 'train':
                    print(f"  {name}: {len(df)} 행, 기간: {df['Date'].min()} ~ {df['Date'].max()}")
                    
            except Exception as e:
                raise RuntimeError(f"{name} 데이터 로딩 중 오류 발생: {e}")
        
        return asset_dfs
    
    def _load_fama_french_data(self):
        """Fama-French 데이터 로드"""
        ff_path = self.data_dir / 'F-F_Research_Data_Factors_daily.CSV'
        
        if not ff_path.exists():
            raise FileNotFoundError(f"Fama-French 데이터 파일을 찾을 수 없습니다: {ff_path}")
        
        try:
            # CSV 파일 로드
            ff_data = pd.read_csv(ff_path)
            
            # 필수 컬럼 존재 확인
            required_cols = ['Date', 'Mkt-RF', 'RF']
            missing_cols = [col for col in required_cols if col not in ff_data.columns]
            if missing_cols:
                raise ValueError(f"Fama-French 데이터에 필수 컬럼이 없습니다: {missing_cols}")
            
            # 날짜 형식 변환 (YYYYMMDD -> datetime)
            ff_data['Date'] = pd.to_datetime(ff_data['Date'].astype(str), format='%Y%m%d')
            
            # 퍼센트를 소수로 변환 (이미 일일 수익률)
            ff_data['Mkt-RF'] = ff_data['Mkt-RF'] / 100
            ff_data['RF'] = ff_data['RF'] / 100
            
            # 데이터 정렬
            ff_data = ff_data.sort_values('Date').reset_index(drop=True)
            
            if self.split == 'train':
                print(f"  Fama-French: {len(ff_data)} 행, 기간: {ff_data['Date'].min()} ~ {ff_data['Date'].max()}")
            
            return ff_data
            
        except Exception as e:
            raise RuntimeError(f"Fama-French 데이터 로딩 중 오류 발생: {e}")
    
    def _merge_and_clean_data(self, asset_dfs, ff_data):
        """데이터 병합 및 정리"""
        # 공통 날짜 찾기
        date_sets = [set(df['Date']) for df in asset_dfs]
        date_sets.append(set(ff_data['Date']))
        common_dates = sorted(list(set.intersection(*date_sets)))
        
        if len(common_dates) == 0:
            raise ValueError("자산 데이터와 Fama-French 데이터에 공통 날짜가 없습니다.")
        
        if self.split == 'train':
            print(f"  공통 날짜: {len(common_dates)}개 ({common_dates[0]} ~ {common_dates[-1]})")
        
        # 자산 수익률 데이터 정리
        asset_returns = []
        for df in asset_dfs:
            # 공통 날짜로 필터링 및 정렬
            filtered_df = df[df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
            
            # 수익률 추출 (NaN 제거)
            returns = filtered_df['Returns'].dropna().values
            asset_returns.append(returns)
        
        # 시장 데이터 정리
        ff_filtered = ff_data[ff_data['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
        market_data = ff_filtered[['Mkt-RF', 'RF']].values
        
        # 데이터 길이 맞추기 (가장 짧은 시계열에 맞춤)
        min_len = min([len(returns) for returns in asset_returns] + [len(market_data)])
        
        if min_len < self.lookback + self.pred_horizon + 50:  # 최소 데이터 길이 확인
            raise ValueError(f"데이터가 부족합니다. 최소 {self.lookback + self.pred_horizon + 50}일 필요, 현재: {min_len}일")
        
        # 최종 데이터 배열 생성
        self.asset_data = np.array([returns[:min_len] for returns in asset_returns]).T  # (time, num_assets)
        self.market_data = market_data[:min_len]  # (time, 2)
        
        # 날짜 배열
        common_dates_sorted = sorted(common_dates)[:min_len]
        self.dates = common_dates_sorted
        
        if self.split == 'train':
            print(f"  최종 데이터 형태: 자산 {self.asset_data.shape}, 시장 {self.market_data.shape}")
    
    def _validate_data_quality(self):
        """데이터 품질 검증"""
        # NaN 값 확인
        asset_nan_count = np.isnan(self.asset_data).sum()
        market_nan_count = np.isnan(self.market_data).sum()
        
        if asset_nan_count > 0:
            print(f"⚠️ 자산 데이터에 NaN 값 {asset_nan_count}개 발견")
        
        if market_nan_count > 0:
            print(f"⚠️ 시장 데이터에 NaN 값 {market_nan_count}개 발견")
        
        # 이상치 확인 (절댓값이 0.5보다 큰 일일 수익률)
        extreme_returns = np.abs(self.asset_data) > 0.5
        if extreme_returns.any():
            extreme_count = extreme_returns.sum()
            print(f"⚠️ 극한 수익률 (|return| > 50%) {extreme_count}개 발견")
        
        # 데이터 통계 출력 (훈련 세트에서만)
        if self.split == 'train':
            print(f"  자산 수익률 통계:")
            for i, name in enumerate(self.asset_names):
                returns = self.asset_data[:, i]
                print(f"    {name}: 평균={returns.mean():.4f}, 표준편차={returns.std():.4f}, "
                f"최소={returns.min():.4f}, 최대={returns.max():.4f}")
    
    def _create_data_split(self):
        """데이터 분할 생성"""
        total_days = len(self.dates)
        
        # 시계열 특성을 고려한 순차적 분할 (70% / 15% / 15%)
        train_end = int(total_days * 0.70)
        val_end = int(total_days * 0.85)
        
        # 각 분할에서 유효한 인덱스 범위 계산
        if self.split == 'train':
            start_idx = self.lookback
            end_idx = train_end - self.pred_horizon
        elif self.split == 'val':
            start_idx = train_end
            end_idx = val_end - self.pred_horizon
        else:  # test
            start_idx = val_end
            end_idx = total_days - self.pred_horizon
        
        # 유효한 인덱스 생성
        if end_idx <= start_idx:
            self.valid_indices = []
            print(f"⚠️ {self.split.upper()} 데이터셋에 유효한 샘플이 없습니다.")
        else:
            self.valid_indices = list(range(start_idx, end_idx))
        
        # 정규화 (훈련 데이터로만 학습)
        if self.normalize and self.split == 'train':
            self._fit_scalers()
        elif self.normalize and self.split in ['val', 'test']:
            # 검증/테스트 데이터는 변환만 수행 (훈련에서 학습된 스케일러 사용)
            pass
    
    def _fit_scalers(self):
        """정규화 스케일러 학습 (훈련 데이터만)"""
        if self.asset_scaler is not None:
            train_asset_data = self.asset_data[:len(self.valid_indices) + self.lookback]
            self.asset_scaler.fit(train_asset_data)
        
        if self.market_scaler is not None:
            train_market_data = self.market_data[:len(self.valid_indices) + self.lookback]
            self.market_scaler.fit(train_market_data)
    
    def _print_dataset_info(self):
        """데이터셋 정보 출력"""
        print(f"\n=== 데이터셋 정보 ===")
        print(f"전체 기간: {self.dates[0]} ~ {self.dates[-1]} ({len(self.dates)}일)")
        print(f"자산 수: {self.num_assets} ({', '.join(self.asset_names)})")
        print(f"시계열 길이: {self.lookback}일")
        print(f"예측 기간: {self.pred_horizon}일")
        print(f"정규화: {'Yes' if self.normalize else 'No'}")
        
        # 분할별 샘플 수
        total_days = len(self.dates)
        train_end = int(total_days * 0.70)
        val_end = int(total_days * 0.85)
        
        train_samples = max(0, train_end - self.pred_horizon - self.lookback)
        val_samples = max(0, val_end - self.pred_horizon - train_end)
        test_samples = max(0, total_days - self.pred_horizon - val_end)
        
        print(f"데이터 분할:")
        print(f"  훈련: {train_samples:,}개 샘플")
        print(f"  검증: {val_samples:,}개 샘플") 
        print(f"  테스트: {test_samples:,}개 샘플")
        print(f"  현재 분할({self.split}): {len(self.valid_indices):,}개 샘플")
    
    def _print_rebalancing_info(self):
        """리밸런싱 정보 출력"""
        print(f"\n=== 리밸런싱 설정 정보 ===")
        if self.rebalancing_frequency == 'monthly':
            print(f"월별 리밸런싱 설정:")
            print(f"  - 예측 기간: {self.pred_horizon}일 (약 {self.pred_horizon/21:.1f}개월)")
            print(f"  - 한 달 거래일: 약 21일")
            print(f"  - 연간화 기준: 12개월")
            if self.pred_horizon < 21:
                print(f"  ⚠️ 예측 기간이 한 달(21일)보다 짧습니다. 스케일링을 통해 월간 수익률로 변환됩니다.")
            elif self.pred_horizon >= 21:
                n_months = self.pred_horizon // 21
                print(f"  ✓ 예측 기간에서 {n_months}개의 완전한 월간 데이터를 계산할 수 있습니다.")
        else:
            print(f"일별 리밸런싱 설정:")
            print(f"  - 예측 기간: {self.pred_horizon}일")
            print(f"  - 연간화 기준: 252거래일")
    
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        데이터 샘플 반환
        
        Args:
            idx (int): 샘플 인덱스
            
        Returns:
            dict: {
                'asset_data': 자산 수익률 (num_assets, lookback, 1),
                'common_data': 시장 데이터 (lookback, 2),
                'future_returns': 미래 수익률 (pred_horizon, num_assets)
            }
        """
        if idx >= len(self.valid_indices):
            raise IndexError(f"인덱스 {idx}가 데이터셋 크기 {len(self.valid_indices)}를 초과합니다.")
        
        # 실제 시점 계산
        t = self.valid_indices[idx]
        
        # 과거 데이터 추출
        asset_hist = self.asset_data[t-self.lookback:t]  # (lookback, num_assets)
        market_hist = self.market_data[t-self.lookback:t]  # (lookback, 2)
        
        # 미래 수익률 추출
        future_rets = self.asset_data[t:t+self.pred_horizon]  # (pred_horizon, num_assets)
        
        # 정규화 적용 (if enabled)
        if self.normalize:
            if self.asset_scaler is not None:
                asset_hist = self.asset_scaler.transform(asset_hist)
            if self.market_scaler is not None:
                market_hist = self.market_scaler.transform(market_hist)
        
        # 텐서 변환
        asset_data = torch.FloatTensor(asset_hist).transpose(0, 1).unsqueeze(-1)  # (num_assets, lookback, 1)
        common_data = torch.FloatTensor(market_hist)  # (lookback, 2)
        future_returns = torch.FloatTensor(future_rets)  # (pred_horizon, num_assets)
        
        # 데이터 품질 검증
        if torch.isnan(asset_data).any() or torch.isnan(common_data).any() or torch.isnan(future_returns).any():
            print(f"⚠️ 샘플 {idx}에서 NaN 값 발견")
        
        return {
            'asset_data': asset_data,
            'common_data': common_data, 
            'future_returns': future_returns
        }


def create_dataloaders(data_dir, batch_size=32, lookback=252, pred_horizon=21, 
                      normalize=False, num_workers=0, pin_memory=True, rebalancing_frequency='monthly'):
    """
    훈련/검증/테스트 데이터 로더 생성
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        batch_size (int): 배치 크기
        lookback (int): 과거 데이터 길이 (일)
        pred_horizon (int): 예측 기간 (일)
        normalize (bool): 데이터 정규화 여부
        num_workers (int): 데이터 로딩 워커 수
        pin_memory (bool): GPU 메모리 고정 여부
        rebalancing_frequency (str): 리밸런싱 주기 ('daily' 또는 'monthly')
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print(f"\n{'='*60}")
    print(f"TCN 포트폴리오 데이터 로더 생성")
    print(f"{'='*60}")
    print(f"설정:")
    print(f"  데이터 디렉토리: {data_dir}")
    print(f"  배치 크기: {batch_size}")
    print(f"  시계열 길이: {lookback}일")
    print(f"  예측 기간: {pred_horizon}일")
    print(f"  정규화: {'Yes' if normalize else 'No'}")
    print(f"  리밸런싱 주기: {rebalancing_frequency}")
    print(f"  워커 수: {num_workers}")
    print(f"  메모리 고정: {'Yes' if pin_memory else 'No'}")
    
    try:
        # 데이터셋 생성
        train_dataset = FinancialDataset(
            data_dir=data_dir,
            lookback=lookback,
            pred_horizon=pred_horizon,
            split='train',
            normalize=normalize,
            rebalancing_frequency=rebalancing_frequency
        )
        
        val_dataset = FinancialDataset(
            data_dir=data_dir,
            lookback=lookback,
            pred_horizon=pred_horizon,
            split='val',
            normalize=normalize,
            rebalancing_frequency=rebalancing_frequency
        )
        
        test_dataset = FinancialDataset(
            data_dir=data_dir,
            lookback=lookback,
            pred_horizon=pred_horizon,
            split='test',
            normalize=normalize,
            rebalancing_frequency=rebalancing_frequency
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # 훈련은 셔플
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False  # 마지막 배치도 사용
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 검증/테스트는 순서 유지
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        # 첫 번째 배치로 데이터 형태 검증
        print(f"\n=== 데이터 형태 검증 ===")
        try:
            sample_batch = next(iter(train_loader))
            print(f"자산 데이터 형태: {sample_batch['asset_data'].shape}")
            print(f"시장 데이터 형태: {sample_batch['common_data'].shape}")
            print(f"미래 수익률 형태: {sample_batch['future_returns'].shape}")
            print(f"✅ 데이터 형태 검증 통과")
        except Exception as e:
            print(f"❌ 데이터 형태 검증 실패: {e}")
            raise
        
        # 로더 통계
        print(f"\n=== 데이터 로더 통계 ===")
        print(f"훈련 로더: {len(train_loader)} 배치 ({len(train_dataset)} 샘플)")
        print(f"검증 로더: {len(val_loader)} 배치 ({len(val_dataset)} 샘플)")
        print(f"테스트 로더: {len(test_loader)} 배치 ({len(test_dataset)} 샘플)")
        print(f"총 샘플 수: {len(train_dataset) + len(val_dataset) + len(test_dataset):,}개")
        
        # 메모리 사용량 추정
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        sample_size = lookback * (train_dataset.num_assets + 2) + pred_horizon * train_dataset.num_assets
        estimated_memory_mb = total_samples * sample_size * 4 / (1024 * 1024)  # float32 = 4 bytes
        print(f"예상 메모리 사용량: {estimated_memory_mb:.1f} MB")
        
        print(f"✅ 데이터 로더 생성 완료")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"❌ 데이터 로더 생성 실패: {e}")
        raise


def test_dataloaders(data_dir='data'):
    """데이터 로더 테스트 함수"""
    print("데이터 로더 테스트 시작...")
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            lookback=20,
            pred_horizon=5
        )
        
        # 각 로더에서 샘플 추출 테스트
        for name, loader in [('훈련', train_loader), ('검증', val_loader), ('테스트', test_loader)]:
            print(f"\n{name} 로더 테스트:")
            batch = next(iter(loader))
            
            print(f"  자산 데이터: {batch['asset_data'].shape}")
            print(f"  시장 데이터: {batch['common_data'].shape}")
            print(f"  미래 수익률: {batch['future_returns'].shape}")
            print(f"  NaN 확인: 자산={torch.isnan(batch['asset_data']).sum()}, "
                  f"시장={torch.isnan(batch['common_data']).sum()}, "
                  f"미래={torch.isnan(batch['future_returns']).sum()}")
        
        print("\n✅ 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        raise


if __name__ == "__main__":
    # 테스트 실행
    test_dataloaders()
