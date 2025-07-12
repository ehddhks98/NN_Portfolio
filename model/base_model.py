import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.leaky_relu1 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.leaky_relu2 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.leaky_relu1, self.drop1,
            self.conv2, self.chomp2, self.bn2, self.leaky_relu2, self.drop2
        )

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None
        
        self.final_activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_activation(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            pad = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, ch, kernel_size, stride=1, 
                dilation=dilation, padding=pad, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(num_channels[-1], hidden_size),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        # Conv1d를 위한 전치: (batch, seq, features) → (batch, features, seq)
        x = x.transpose(1, 2)
        
        # TCN 레이어들을 통과
        out = self.network(x)
        
        # 마지막 타임스텝 선택: (batch, channels, seq) → (batch, channels)
        out = out[:, :, -1]
        
        # 최종 은닉 크기로 투영
        out = self.output_projection(out)
        return out


class BetaEstimator(nn.Module):
    def __init__(self, num_assets=5, hidden_size=64, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2):
        super().__init__()
        self.num_assets = num_assets
        self.returns_encoder = TCNEncoder(
            input_size=1,
            hidden_size=hidden_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.asset_embeddings = nn.Embedding(num_assets, hidden_size)
        
        # 베타 예측기 - 더 깊은 네트워크와 잔차 연결
        self.beta_hidden1 = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        
        self.beta_hidden2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5)
        )
        
        self.beta_output = nn.Linear(hidden_size // 2, 1)
        
        # 베타 초기화 - 평균 1.0 근처에서 시작
        nn.init.normal_(self.beta_output.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.beta_output.bias, 1.0)  # 베타 1.0에서 시작

    def forward(self, asset_data, common_data):
        # 데이터로더에서 오는 형태에 맞게 차원 조정
        if asset_data.dim() == 4:
            asset_data = asset_data.squeeze(-1)
        elif asset_data.dim() == 3 and asset_data.shape[2] == 1:
            asset_data = asset_data.squeeze(-1).unsqueeze(0)
        
        if common_data.dim() == 2:
            common_data = common_data.unsqueeze(0)
            
        batch_size, num_assets, seq_len = asset_data.shape
        device = asset_data.device
        
        # 시장 수익률 추출 및 인코딩
        market_returns = common_data[:, :, 0].unsqueeze(-1)  # (batch, seq_len, 1)
        market_context = self.returns_encoder(market_returns)  # (batch, hidden_size)
        
        # 배치 처리를 위한 자산 데이터 재구성
        asset_returns_flat = asset_data.transpose(1, 2).reshape(-1, seq_len).unsqueeze(-1)
        
        # 단일 TCN 호출로 모든 자산 처리
        asset_contexts_flat = self.returns_encoder(asset_returns_flat)  # (batch*assets, hidden_size)
        
        # 개별 자산으로 다시 재구성
        asset_contexts = asset_contexts_flat.view(batch_size, num_assets, -1)  # (batch, assets, hidden_size)
        
        # 모든 자산에 대한 자산 임베딩 한 번에 생성
        asset_ids = torch.arange(num_assets, device=device).unsqueeze(0).expand(batch_size, -1)
        asset_embeddings = self.asset_embeddings(asset_ids)  # (batch, assets, hidden_size)
        
        # 시장 컨텍스트를 자산 차원에 맞게 확장
        market_context_expanded = market_context.unsqueeze(1).expand(-1, num_assets, -1)
        
        # 모든 정보 소스 결합
        combined = torch.cat([
            asset_contexts,           # (batch, assets, hidden_size)
            market_context_expanded,  # (batch, assets, hidden_size)
            asset_embeddings         # (batch, assets, hidden_size)
        ], dim=2)  # (batch, assets, hidden_size * 3)
        
        # MLP 처리를 위한 평탄화 및 모든 베타 한 번에 예측
        combined_flat = combined.view(-1, combined.size(-1))  # (batch*assets, hidden_size*3)
        
        # 베타 예측
        hidden1 = self.beta_hidden1(combined_flat)  # (batch*assets, hidden_size)
        hidden2 = self.beta_hidden2(hidden1)        # (batch*assets, hidden_size//2)
        betas_flat = self.beta_output(hidden2)      # (batch*assets, 1)
        
        # 원래 배치 형태로 재구성
        betas = betas_flat.view(batch_size, num_assets)  # (batch, assets)
        
        return betas


class AdaptivePortfolioOptimizer(nn.Module):
    def __init__(self, num_assets=5, hidden_size=64, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2, risk_free_rate=0.02):
        super().__init__()
        self.num_assets = num_assets
        self.hidden_size = hidden_size
        self.risk_free_rate = risk_free_rate
        
        # 베타 추정기
        self.beta_estimator = BetaEstimator(
            num_assets=num_assets,
            hidden_size=hidden_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 공분산 정규화 파라미터
        self.cov_regularization = 1e-6

    def capm_expected_returns(self, betas, market_return, risk_free_rate=None, common_data=None):
        """
        CAPM을 사용한 기대 수익률 계산: E(R_i) = R_f + β_i * (E(R_m) - R_f)
        
        Args:
            betas: 자산 베타 (batch_size, num_assets)
            market_return: 시장 수익률 (사용하지 않음 - common_data의 Mkt-RF 직접 사용)
            risk_free_rate: 무위험 이자율 (스칼라), None이면 common_data에서 추출
            common_data: 공통 시장 데이터 (batch_size, seq_len, num_features)
                        - 인덱스 0: Mkt-RF (시장 위험 프리미엄) ← 직접 사용
                        - 인덱스 1: RF (무위험 이자율)
            
        Returns:
            expected_returns: 기대 수익률 (batch_size, num_assets)
        """
        if common_data is None or common_data.shape[-1] < 2:
            raise ValueError("common_data가 제공되지 않았거나 Mkt-RF, RF 데이터가 없습니다. Fama-French 데이터가 필요합니다.")
        
        # Fama-French 데이터에서 직접 추출
        # 전체 lookback 기간의 평균값 사용 
        market_premium = common_data[:, :, 0].mean(dim=1)  # Mkt-RF (batch_size,)
        risk_free_rate = common_data[:, :, 1].mean(dim=1)  # RF (batch_size,)
        
        # 시장 프리미엄 검증 (극단적으로 작은 경우만 보정)
        market_premium_abs = torch.abs(market_premium)
        
        # 매우 작은 경우만 최소값 설정 (나머지는 원래 값 사용)
        market_premium_scaled = torch.where(
            market_premium_abs < 1e-6,  # 매우 작은 경우만
            torch.sign(market_premium) * 1e-5,  # 최소값 설정
            market_premium  # 원래 일간 데이터 그대로 사용
        )
        
        # 차원 맞추기
        if market_premium_scaled.dim() == 1:
            market_premium_scaled = market_premium_scaled.unsqueeze(-1)  # (batch_size, 1)
        if risk_free_rate.dim() == 1:
            risk_free_rate = risk_free_rate.unsqueeze(-1)  # (batch_size, 1)
            
        # CAPM: E(R_i) = R_f + β_i * (E(R_m) - R_f)
        # 여기서 market_premium_scaled = E(R_m) - R_f 이므로
        expected_returns = risk_free_rate + betas * market_premium_scaled
        return expected_returns
    
    def compute_covariance_matrix(self, returns_data):
        """
        수익률 데이터에서 표본 공분산 행렬 계산
        
        Args:
            returns_data: 과거 수익률 (batch_size, time_steps, num_assets)
            
        Returns:
            cov_matrix: 공분산 행렬 (batch_size, num_assets, num_assets)
        """
        batch_size, time_steps, num_assets = returns_data.shape
        
        # 데이터 중심화 (평균 차감)
        mean_returns = returns_data.mean(dim=1, keepdim=True)  # (batch, 1, assets)
        centered_returns = returns_data - mean_returns  # (batch, time, assets)
        
        # 공분산 계산: (1/(T-1)) * X^T * X
        cov_matrix = torch.bmm(
            centered_returns.transpose(1, 2),  # (batch, assets, time)
            centered_returns  # (batch, time, assets)
        ) / (time_steps - 1)  # (batch, assets, assets)
        
        # 수치적 안정성을 위한 정규화 추가
        eye = torch.eye(num_assets, device=cov_matrix.device)
        cov_matrix = cov_matrix + self.cov_regularization * eye
        
        return cov_matrix

    def mean_variance_optimization(self, expected_returns, cov_matrix):
        """
        제약 조건이 있는 최소분산 포트폴리오 최적화
        해석적 해: w = (Σ^(-1) * 1) / (1^T * Σ^(-1) * 1)
        
        Args:
            expected_returns: 기대 수익률 (batch_size, num_assets)
            cov_matrix: 공분산 행렬 (batch_size, num_assets, num_assets)
            
        Returns:
            weights: 포트폴리오 가중치 (batch_size, num_assets), 합계 = 정확히 1
        """
        batch_size = expected_returns.shape[0]
        device = expected_returns.device
        
        try:
            # 단위 벡터 (모든 원소가 1인 벡터)
            ones = torch.ones(batch_size, self.num_assets, 1, device=device)
            
            # 수치적 안정성을 위한 정규화
            eye = torch.eye(self.num_assets, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            cov_regularized = cov_matrix + self.cov_regularization * eye
            
            # 최소분산 포트폴리오의 해석적 해
            # w = (Σ^(-1) * 1) / (1^T * Σ^(-1) * 1)
            
            # Step 1: Σ^(-1) * 1 계산
            try:
                # Cholesky 분해 사용 (가장 안정한 방법)
                L = torch.linalg.cholesky(cov_regularized)
                cov_inv_ones = torch.cholesky_solve(ones, L)  # (batch, assets, 1)
            except:
                # Cholesky 실패 시 일반 solve 사용
                try:
                    cov_inv_ones = torch.linalg.solve(cov_regularized, ones)
                except:
                    # 모든 직접 방법 실패 시 pseudo-inverse
                    cov_pinv = torch.linalg.pinv(cov_regularized)
                    cov_inv_ones = torch.bmm(cov_pinv, ones)
            
            # Step 2: 1^T * Σ^(-1) * 1 계산 (스칼라)
            denominator = torch.bmm(ones.transpose(1, 2), cov_inv_ones).squeeze()  # (batch,)
            
            # Step 3: 최종 가중치 계산
            denominator_safe = torch.clamp(denominator, min=1e-8)
            weights = cov_inv_ones.squeeze(-1) / denominator_safe.unsqueeze(-1)  # (batch, assets)            
            
            if not torch.isfinite(weights).all():
                raise RuntimeError("가중치에 NaN 또는 Inf가 발생했습니다.")
            
        except Exception as e:
            print(f"최소분산 포트폴리오 계산 실패: {e}. 동일 가중치를 사용합니다.")
            weights = torch.ones(batch_size, self.num_assets, device=device) / self.num_assets
        
        # 롱온리 제약 (음의 가중치 제거)
        weights = torch.clamp(weights, min=0.0)
        
        # 재정규화 (롱온리 제약 때문에 필요)
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights_sum_safe = torch.clamp(weights_sum, min=1e-8)
        weights = weights / weights_sum_safe
        
        return weights
    
    def validate_portfolio_weights(self, weights, tolerance=1e-6):
        """
        포트폴리오 가중치 제약 조건 검증
        
        Args:
            weights: 포트폴리오 가중치 (batch_size, num_assets)
            tolerance: 허용 오차
            
        Returns:
            bool: 모든 제약 조건이 만족되면 True
        """
        batch_size = weights.shape[0]
        
        # 1. 가중치 합이 1인지 확인
        weights_sum = weights.sum(dim=1)  # (batch_size,)
        sum_constraint = torch.abs(weights_sum - 1.0) <= tolerance
        
        # 2. 가중치가 유한한 값인지 확인 (NaN, Inf 체크)
        finite_constraint = torch.isfinite(weights).all(dim=1)
        
        # 모든 배치 샘플이 제약 조건을 만족하는지 확인
        all_valid = (sum_constraint & finite_constraint).all()
        
        if not all_valid:
            print(f"포트폴리오 가중치 제약 조건 위반:")
            print(f"  - 가중치 합: {weights_sum}")
            print(f"  - 합 제약 만족: {sum_constraint.sum()}/{batch_size}")
            print(f"  - 유한성 제약 만족: {finite_constraint.sum()}/{batch_size}")
            
        return all_valid.item()
    
    def calculate_sharpe_ratio(self, weights, expected_returns, cov_matrix, risk_free_rate=None):
        """
        샤프 비율 계산: (E(R_p) - R_f) / σ_p (연간화)
        
        Args:
            weights: 포트폴리오 가중치 (batch_size, num_assets)
            expected_returns: 기대 수익률 (batch_size, num_assets) - 일간
            cov_matrix: 공분산 행렬 (batch_size, num_assets, num_assets) - 일간
            risk_free_rate: 무위험 이자율 (batch_size,) - 일간
            
        Returns:
            sharpe_ratio: 샤프 비율 (batch_size,) - 연간화
        """
        if risk_free_rate is None:
            raise ValueError("샤프 비율 계산을 위해서는 무위험 이자율이 필요합니다.")
            
        # 포트폴리오 기대 수익률 (일간): w^T * μ
        portfolio_return_daily = torch.sum(weights * expected_returns, dim=1)  # (batch,)
        
        # 포트폴리오 분산 (일간): w^T * Σ * w
        portfolio_variance_daily = torch.bmm(
            torch.bmm(weights.unsqueeze(1), cov_matrix),  # (batch, 1, assets)
            weights.unsqueeze(-1)  # (batch, assets, 1)
        ).squeeze()  # (batch,)
        
        # 포트폴리오 표준편차 (일간)
        portfolio_std_daily = torch.sqrt(torch.clamp(portfolio_variance_daily, min=1e-8))
        
        # 연간화 (252 거래일 기준)
        trading_days_per_year = 252
        
        # 연간 수익률 = 일평균 수익률 × 252 (복리 효과 무시한 단순 근사)
        portfolio_return_annual = portfolio_return_daily * trading_days_per_year
        
        # 연간 변동성 = 일간 표준편차 × √252 (변동성의 제곱근 법칙)
        portfolio_std_annual = portfolio_std_daily * torch.sqrt(torch.tensor(trading_days_per_year, device=portfolio_std_daily.device))
        
        # 연간 무위험 이자율 = 일간 무위험 이자율 × 252
        risk_free_annual = risk_free_rate * trading_days_per_year
        
        # 연간화된 샤프 비율 = (연간수익률 - 연간무위험이자율) / 연간변동성
        sharpe_ratio = (portfolio_return_annual - risk_free_annual) / portfolio_std_annual
        
        return sharpe_ratio
    
    def calculate_realized_sharpe_ratio(self, weights, future_returns, common_data):
        """
        실현 샤프 비율 계산: 실제 미래 수익률 데이터 사용 (연간화)
        
        Args:
            weights: 포트폴리오 가중치 (batch_size, num_assets)
            future_returns: 미래 수익률 (batch_size, pred_horizon, num_assets)
            common_data: 공통 시장 데이터 (batch_size, seq_len, 2)
            
        Returns:
            realized_sharpe_ratio: 실현 샤프 비율 (batch_size,) - 연간화
        """
        # 각 날짜별 포트폴리오 수익률 계산
        # future_returns: (batch_size, pred_horizon, num_assets)
        # weights: (batch_size, num_assets)
        daily_portfolio_returns = torch.sum(
            weights.unsqueeze(1) * future_returns, dim=2
        )  # (batch_size, pred_horizon)
        
        # 포트폴리오 수익률 통계 (일간)
        portfolio_mean_daily = daily_portfolio_returns.mean(dim=1)  # (batch_size,) 일평균
        portfolio_std_daily = daily_portfolio_returns.std(dim=1)    # (batch_size,) 일표준편차
        
        # 무위험 이자율 (일간)
        risk_free_daily = common_data[:, :, 1].mean(dim=1)  # (batch_size,)
        
        # 연간화 (252 거래일 기준)
        trading_days_per_year = 252
        
        # 연간 수익률 = 일평균 수익률 × 252 (복리 효과 무시한 단순 근사)
        portfolio_return_annual = portfolio_mean_daily * trading_days_per_year
        
        # 연간 변동성 = 일표준편차 × √252 (변동성의 제곱근 법칙)
        portfolio_std_annual = portfolio_std_daily * torch.sqrt(torch.tensor(trading_days_per_year, device=portfolio_std_daily.device))
        
        # 연간 무위험 이자율 = 일무위험이자율 × 252
        risk_free_annual = risk_free_daily * trading_days_per_year
        
        # 연간화된 샤프 비율 = (연간수익률 - 연간무위험이자율) / 연간변동성
        realized_sharpe = (portfolio_return_annual - risk_free_annual) / torch.clamp(portfolio_std_annual, min=1e-8)
        
        return realized_sharpe

    def forward(self, asset_data, common_data, future_returns):
        """
        적응형 포트폴리오 최적화기의 전체 순전파 함수
        
        Args:
            asset_data: 자산 수익률 텐서
                - 배치 처리 시: (batch_size, num_assets, seq_len, 1)
                - 단일 샘플: (num_assets, seq_len, 1)
            common_data: 공통 시장 데이터
                - 배치 처리 시: (batch_size, seq_len, num_features)
                - 단일 샘플: (seq_len, num_features)
                - 인덱스 0: Mkt-RF, 인덱스 1: RF
            future_returns: 샤프 계산을 위한 미래 수익률
                - 배치 처리 시: (batch_size, pred_horizon, num_assets)
                - 단일 샘플: (pred_horizon, num_assets)
            
        Returns:
            dict: 모든 중간 결과와 최종 손실을 포함
        """
        # 데이터로더에서 오는 형태에 맞게 차원 조정
        if asset_data.dim() == 4:
            # 배치 처리: (batch_size, num_assets, seq_len, 1) -> (batch_size, num_assets, seq_len)
            asset_data = asset_data.squeeze(-1)
        elif asset_data.dim() == 3 and asset_data.shape[2] == 1:
            # 단일 샘플: (num_assets, seq_len, 1) -> (1, num_assets, seq_len)
            asset_data = asset_data.squeeze(-1).unsqueeze(0)
        
        if common_data.dim() == 2:
            # 단일 샘플: (seq_len, num_features) -> (1, seq_len, num_features)
            common_data = common_data.unsqueeze(0)
        # 배치 처리인 경우 (batch_size, seq_len, num_features)는 그대로 사용
            
        # future_returns 차원 처리 (실현 샤프 비율을 위해 원본 보존)
        original_future_returns = future_returns.clone()
        
        if future_returns.dim() == 2:
            # 단일 샘플: (pred_horizon, num_assets) -> (1, pred_horizon, num_assets)
            original_future_returns = future_returns.unsqueeze(0)
            # 평균 계산: -> (1, num_assets)
            future_returns = future_returns.mean(dim=0).unsqueeze(0)
        elif future_returns.dim() == 3:
            # 배치 처리: (batch_size, pred_horizon, num_assets) -> (batch_size, num_assets) (평균 사용)
            future_returns = future_returns.mean(dim=1)
            
        batch_size = asset_data.shape[0]
        
        # 단계 1: TCN을 사용한 베타 추정
        betas = self.beta_estimator(asset_data, common_data)
        
        # 단계 2: CAPM을 사용한 기대 수익률 계산 (common_data에서 직접 추출)
        expected_returns = self.capm_expected_returns(betas, None, common_data=common_data)
        
        # 단계 3: 과거 데이터에서 공분산 행렬 계산
        # 공분산 계산을 위한 asset_data 재구성: (batch, assets, seq) -> (batch, seq, assets)
        returns_for_cov = asset_data.transpose(1, 2)
        cov_matrix = self.compute_covariance_matrix(returns_for_cov)
        
        # 단계 4: 포트폴리오 가중치에 대한 평균-분산 최적화
        weights = self.mean_variance_optimization(expected_returns, cov_matrix)
        
        # 포트폴리오 가중치 제약 조건 검증 (디버깅용)
        if not self.validate_portfolio_weights(weights):
            print("경고: 포트폴리오 가중치 제약 조건이 위반되었습니다.")
        
        # 단계 5: 미래 수익률을 사용한 샤프 비율 계산
        # 훈련을 위해 future_returns를 사용하여 실현된 포트폴리오 수익률 계산
        realized_portfolio_return = torch.sum(weights * future_returns, dim=1)
        
        # 리스크 추정을 위해 과거 공분산 사용 (더 안정적)
        # 이론적 샤프 비율 계산에서도 실제 RF 값 사용
        current_rf = common_data[:, :, 1].mean(dim=1)  # 전체 기간 평균 RF
        theoretical_sharpe_ratio = self.calculate_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=current_rf)
        
        # 실현 샤프 비율 계산 (실제 미래 수익률 사용)
        realized_sharpe_ratio = self.calculate_realized_sharpe_ratio(weights, original_future_returns, common_data)
        
        # 단계 6: 손실 = -실현 샤프 비율 (실제 투자 성과 기반 학습)
        # 배치 내 샘플들의 실현 샤프 비율 평균을 최대화
        loss = -realized_sharpe_ratio.mean()
        
        return {
            'betas': betas,
            'expected_returns': expected_returns,
            'weights': weights,
            'cov_matrix': cov_matrix,
            'sharpe_ratio': theoretical_sharpe_ratio,  # 이론적 샤프 비율
            'realized_sharpe_ratio': realized_sharpe_ratio,  # 실현 샤프 비율
            'realized_return': realized_portfolio_return,
            'loss': loss
        }

