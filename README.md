# Equity Signal Lab

Language: [English](#english) | [中文](#中文)

<a id="english"></a>
<details open>
<summary><strong>English (default)</strong></summary>

Dataset-agnostic pipeline for predicting overnight and intraday equity returns using time-series cross-validation. Ships with a sample dataset of 5 tickers.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Reproducible training (full pipeline):**
```bash
PYTHONHASHSEED=42 PYTHONPATH=. python scripts/train.py \
  --models ridge,elasticnet --features v1,v2 --target overnight_return
```

**Quick single-model run:**
```bash
PYTHONHASHSEED=42 PYTHONPATH=. python scripts/train.py --models ridge --features v1
```

**Custom dataset:**
```bash
PYTHONPATH=. python scripts/train.py --data data/raw/my_data.xlsx --models ridge --features v1
```

**Prediction:**
```bash
PYTHONPATH=. python scripts/predict.py --date 2026-02-13 --model ridge --features v2
```

## Results

Out-of-sample CV results (sample dataset, expanding window 21-756 days, corporate-action days filtered, hyperparameters learned per fold):

**Overnight return:**

| Model | R² | Dir Acc | Baseline Acc | IC (x-sectional) | Sharpe (portfolio) |
|-------|------|---------|--------------|-------------------|--------------------|
| Ridge v1 | -0.011 | 0.596 | 0.602 | 0.028 | 2.72 |
| ElasticNet v1 | -0.008 | 0.598 | 0.602 | 0.020 | 2.81 |
| Ridge v2 | -0.018 | 0.588 | 0.603 | 0.038 | 2.67 |
| ElasticNet v2 | -0.006 | 0.605 | 0.603 | 0.041 | 2.99 |

**Intraday return:**

| Model | R² | Dir Acc | Baseline Acc | IC (x-sectional) | Sharpe (portfolio) |
|-------|------|---------|--------------|-------------------|--------------------|
| Ridge v1 | -0.007 | 0.491 | 0.494 | -0.006 | 0.19 |
| ElasticNet v1 | -0.007 | 0.489 | 0.494 | -0.013 | 0.12 |
| Ridge v2 | -0.006 | 0.490 | 0.493 | -0.003 | 0.43 |
| ElasticNet v2 | -0.009 | 0.492 | 0.493 | 0.002 | 0.32 |

**Key findings:**
- Overnight: ElasticNet v2 marginally beats baseline (60.5% vs 60.3%) with learned hyperparameters
- Intraday: no model beats the baseline (~49% direction accuracy, near coin-flip)
- High overnight Sharpe (2.7-3.0) reflects the sample period's bull market, not model alpha
- Intraday Sharpe is low (0.1-0.4), consistent with weak signal
- Cross-sectional IC is near zero for both targets
- R² is negative across all models (expected for daily equity returns)

## Methodology

**Loss function:** All models minimize MSE (mean squared error). MSE is the natural choice for continuous return prediction - large errors are penalized quadratically, matching their outsized portfolio impact.

**Evaluation metrics:**
- **R²** - standard regression quality; expected to be low (0.01-0.05) for daily equity returns
- **Direction Accuracy** - % correct sign predictions; directly maps to trading profitability (>50% = edge)
- **IC (Spearman)** - rank correlation between predicted and actual; industry-standard alpha quality metric
- **Sharpe Ratio** - risk-adjusted return of a simplified long/short strategy (no transaction costs)

**Hyperparameters:** Learned across the basket via built-in CV. Ridge uses leave-one-out CV over alphas {0.01, 0.1, 1.0, 10.0, 100.0}. ElasticNet uses 3-fold CV over 20 alpha values and l1_ratio {0.1, 0.5, 0.9}. Hyperparameters are re-selected in each outer CV fold to avoid data leakage.

## Testing

All 23 tests pass:
```bash
pytest tests/ -v

# Test against a different dataset:
EQUITY_DATA=data/raw/other_data.xlsx pytest tests/ -v
```

## Project Structure

- `data/`: Raw Excel data
- `src/`: Core modules (data, features, models, cv, evaluation, predict)
- `scripts/`: CLI training and prediction
- `tests/`: Unit tests + smoke test (23 passing)
- `results/`: Saved models, figures, comparison CSV

</details>

<a id="中文"></a>
<details>
<summary><strong>中文</strong></summary>

通用股票收益预测流水线，支持隔夜和日内收益的时间序列交叉验证。附带 5 只股票的样本数据集。

## 环境安装

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 使用方式

**可复现实验（完整流程）：**
```bash
PYTHONHASHSEED=42 PYTHONPATH=. python scripts/train.py \
  --models ridge,elasticnet --features v1,v2 --target overnight_return
```

**快速单模型训练：**
```bash
PYTHONHASHSEED=42 PYTHONPATH=. python scripts/train.py --models ridge --features v1
```

**自定义数据集：**
```bash
PYTHONPATH=. python scripts/train.py --data data/raw/my_data.xlsx --models ridge --features v1
```

**预测：**
```bash
PYTHONPATH=. python scripts/predict.py --date 2026-02-13 --model ridge --features v2
```

## 实验结果

样本外 CV 结果（样本数据集，窗口从 21 到 756 天扩展，公司行为日期已过滤，每折内自动学习超参数）：

**隔夜收益：**

| 模型 | R² | 方向准确率 | 基准准确率 | IC（横截面） | Sharpe（组合） |
|------|------|------------|------------|---------------|----------------|
| Ridge v1 | -0.011 | 0.596 | 0.602 | 0.028 | 2.72 |
| ElasticNet v1 | -0.008 | 0.598 | 0.602 | 0.020 | 2.81 |
| Ridge v2 | -0.018 | 0.588 | 0.603 | 0.038 | 2.67 |
| ElasticNet v2 | -0.006 | 0.605 | 0.603 | 0.041 | 2.99 |

**日内收益：**

| 模型 | R² | 方向准确率 | 基准准确率 | IC（横截面） | Sharpe（组合） |
|------|------|------------|------------|---------------|----------------|
| Ridge v1 | -0.007 | 0.491 | 0.494 | -0.006 | 0.19 |
| ElasticNet v1 | -0.007 | 0.489 | 0.494 | -0.013 | 0.12 |
| Ridge v2 | -0.006 | 0.490 | 0.493 | -0.003 | 0.43 |
| ElasticNet v2 | -0.009 | 0.492 | 0.493 | 0.002 | 0.32 |

**主要结论：**
- 隔夜：ElasticNet v2 略微超过基准（60.5% vs 60.3%），使用自动学习的超参数
- 日内：没有模型超过基准（约 49% 方向准确率，接近随机）
- 较高的隔夜 Sharpe（2.7-3.0）主要来自样本期间的牛市环境，不代表模型 alpha
- 日内 Sharpe 较低（0.1-0.4），信号较弱
- 横截面 IC 在两个目标上均接近零
- 所有模型 R² 为负（在日频收益预测中较常见）

## 方法说明

**损失函数：** 所有模型都最小化 MSE。对于连续收益预测，MSE 能更严惩大误差，和实盘风险影响一致。

**评估指标：**
- **R²**：回归拟合优度；日频股票收益通常很低
- **方向准确率**：预测涨跌方向正确的比例；与交易可用性直接相关
- **IC（Spearman）**：预测值与真实值的秩相关；衡量 alpha 质量的常用指标
- **Sharpe Ratio**：简化多空策略的风险调整后收益（未计交易成本）

**超参数：** 通过内置 CV 在整个股票篮子上自动学习。Ridge 使用留一法 CV，搜索 alpha {0.01, 0.1, 1.0, 10.0, 100.0}。ElasticNet 使用 3 折 CV，搜索 20 个 alpha 值和 l1_ratio {0.1, 0.5, 0.9}。每个外层 CV 折内重新选择超参数，避免数据泄漏。

## 测试

共 23 个测试通过：
```bash
pytest tests/ -v

# 使用其他数据集测试：
EQUITY_DATA=data/raw/other_data.xlsx pytest tests/ -v
```

## 项目结构

- `data/`：原始 Excel 数据
- `src/`：核心模块（data、features、models、cv、evaluation、predict）
- `scripts/`：训练与预测命令行入口
- `tests/`：单元测试和冒烟测试（23 通过）
- `results/`：模型文件、图表、对比 CSV

</details>
