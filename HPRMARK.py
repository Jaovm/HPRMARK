import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import linkage, fcluster
from pypfopt import EfficientFrontier, risk_models, expected_returns

def calculate_cagr(data):
    years = (data.index[-1] - data.index[0]).days / 365.25
    cagr = (data.iloc[-1] / data.iloc[0]) ** (1 / years) - 1
    return cagr

def calculate_sharpe(cagr, returns, risk_free_rate=0.03):
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = (cagr - risk_free_rate) / volatility
    return sharpe_ratio

def hierarchical_risk_parity(returns):
    cov_matrix = returns.cov()
    linkage_matrix = linkage(cov_matrix, method='ward')
    clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
    cluster_risk = np.sum(cov_matrix, axis=1)
    total_risk = np.sum(cluster_risk)
    return cluster_risk / total_risk

def markowitz_optimization(returns):
    mu = expected_returns.mean_historical_return(returns)
    sigma = risk_models.sample_cov(returns)
    ef = EfficientFrontier(mu, sigma)
    return ef.max_sharpe()

tickers = ['AGRO3.SA', 'BBAS3.SA', 'BBSE3.SA', 'BPAC11.SA', 'EGIE3.SA', 'ITUB3.SA', 'PRIO3.SA', 'PSSA3.SA', 'SAPR3.SA', 'SBSP3.SA', 'VIVT3.SA', 'WEGE3.SA', 'TOTS3.SA', 'B3SA3.SA', 'TAEE3.SA']
data = yf.download(tickers, start='2018-01-01', end='2025-04-01')['Adj Close']
returns = data.pct_change().dropna()

cagr = calculate_cagr(data)
hrp_weights = hierarchical_risk_parity(returns)
markowitz_weights = markowitz_optimization(returns)

sharpe_hrp = calculate_sharpe(np.dot(hrp_weights, cagr), returns)
sharpe_markowitz = calculate_sharpe(np.dot(markowitz_weights, cagr), returns)

st.title("Comparação HRP vs Markowitz com CAGR")
st.subheader("Índice de Sharpe com CAGR")
st.write(f"Sharpe HRP: {sharpe_hrp:.2f}")
st.write(f"Sharpe Markowitz: {sharpe_markowitz:.2f}")
