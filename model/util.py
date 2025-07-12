import torch
import torch.nn as nn
import numpy as np


def validate_portfolio_weights(weights, tolerance=1e-6):
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


def calculate_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=None):
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


def calculate_realized_sharpe_ratio(weights, future_returns, common_data):
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


def compute_covariance_matrix(returns_data, regularization=1e-6):
    """
    공분산 행렬 계산
    
    Args:
        returns_data: 수익률 데이터 (batch_size, seq_len, num_assets)
        regularization: 정규화 파라미터
        
    Returns:
        cov_matrix: 공분산 행렬 (batch_size, num_assets, num_assets)
    """
    batch_size, seq_len, num_assets = returns_data.shape
    
    # 평균 계산
    mean_returns = returns_data.mean(dim=1, keepdim=True)  # (batch, 1, assets)
    
    # 편차 계산
    deviations = returns_data - mean_returns  # (batch, seq, assets)
    
    # 공분산 계산: (batch, seq, assets) -> (batch, assets, assets)
    # 각 배치 샘플에 대해 개별적으로 계산
    cov_matrix = torch.zeros(batch_size, num_assets, num_assets, device=returns_data.device)
    
    for i in range(batch_size):
        # (seq, assets) -> (assets, seq) @ (seq, assets) -> (assets, assets)
        cov_matrix[i] = torch.mm(deviations[i].T, deviations[i]) / (seq_len - 1)
    
    # 수치적 안정성을 위한 정규화 추가
    eye = torch.eye(num_assets, device=cov_matrix.device)
    cov_matrix = cov_matrix + regularization * eye
    
    return cov_matrix


def normalize_weights(logits, method='softmax'):
    """
    포트폴리오 가중치 정규화
    
    Args:
        logits: 원시 가중치 값 (batch_size, num_assets)
        method: 정규화 방법 ('softmax' 또는 'sum')
        
    Returns:
        weights: 정규화된 가중치 (batch_size, num_assets)
    """
    if method == 'softmax':
        # 수치적으로 안정적인 softmax 사용
        weights = torch.softmax(logits, dim=1)
    elif method == 'sum':
        # 기존 방식 (수치적 불안정성 가능성)
        weights_sum = logits.sum(dim=1, keepdim=True)
        weights = logits / torch.clamp(weights_sum, min=1e-8)
    else:
        raise ValueError(f"알 수 없는 정규화 방법: {method}")
    
    return weights


def capm_expected_returns(betas, market_return=None, risk_free_rate=None, common_data=None):
    """
    CAPM을 사용한 기대 수익률 계산
    
    Args:
        betas: 베타 계수 (batch_size, num_assets)
        market_return: 시장 수익률 (batch_size,) 또는 None
        risk_free_rate: 무위험 이자율 (batch_size,) 또는 None
        common_data: 공통 시장 데이터 (batch_size, seq_len, 2)
        
    Returns:
        expected_returns: 기대 수익률 (batch_size, num_assets)
    """
    if common_data is not None:
        # common_data에서 시장 수익률과 무위험 이자율 추출
        market_excess_return = common_data[:, :, 0]  # Mkt-RF
        risk_free_rate = common_data[:, :, 1]        # RF
        
        # 시계열 평균 계산
        market_return = market_excess_return.mean(dim=1)  # (batch_size,)
        risk_free_rate = risk_free_rate.mean(dim=1)       # (batch_size,)
    
    if market_return is None or risk_free_rate is None:
        raise ValueError("시장 수익률과 무위험 이자율이 필요합니다.")
    
    # CAPM 공식: E(R_i) = R_f + β_i * (E(R_m) - R_f)
    # 여기서 market_return은 이미 초과수익률이므로: E(R_i) = R_f + β_i * market_return
    expected_returns = risk_free_rate.unsqueeze(1) + betas * market_return.unsqueeze(1)
    
    return expected_returns


def adjust_data_dimensions(asset_data, common_data, future_returns):
    """
    데이터 차원을 모델에 맞게 조정
    
    Args:
        asset_data: 자산 데이터
        common_data: 공통 데이터
        future_returns: 미래 수익률
        
    Returns:
        tuple: 조정된 데이터들
    """
    # asset_data 차원 조정
    if asset_data.dim() == 4:
        asset_data = asset_data.squeeze(-1)
    elif asset_data.dim() == 3 and asset_data.shape[2] == 1:
        asset_data = asset_data.squeeze(-1).unsqueeze(0)
    
    # common_data 차원 조정
    if common_data.dim() == 2:
        common_data = common_data.unsqueeze(0)
    
    # future_returns 차원 처리
    original_future_returns = future_returns.clone()
    
    if future_returns.dim() == 2:
        original_future_returns = future_returns.unsqueeze(0)
        future_returns = future_returns.mean(dim=0).unsqueeze(0)
    elif future_returns.dim() == 3:
        future_returns = future_returns.mean(dim=1)
    
    return asset_data, common_data, future_returns, original_future_returns 