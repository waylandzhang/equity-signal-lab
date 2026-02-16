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

Out-of-sample CV results (overnight_return, sample dataset, expanding window 21-756 days, corporate-action days filtered):

| Model | R² | Dir Acc | Baseline Acc | IC (x-sectional) | Sharpe (portfolio) |
|-------|------|---------|--------------|-------------------|--------------------|
| Ridge v1 | -0.013 | 0.592 | 0.602 | 0.025 | 2.61 |
| ElasticNet v1 | -0.012 | 0.592 | 0.602 | 0.027 | 2.63 |
| Ridge v2 | -0.012 | 0.593 | 0.603 | 0.042 | 2.82 |
| ElasticNet v2 | -0.013 | 0.590 | 0.603 | 0.028 | 2.45 |

**Key findings:**
- No model beats the always-long baseline (~60% direction accuracy)
- All models underperform baseline by ~1%, meaning the features add noise, not signal
- High portfolio Sharpe (2.5-2.8) reflects the sample period's bull market, not model alpha
- Cross-sectional IC is positive but tiny (0.02-0.04) - negligible stock-picking power
- R² is negative across all models (expected for daily equity returns)

## Methodology

**Loss function:** All models minimize MSE (mean squared error). MSE is the natural choice for continuous return prediction - large errors are penalized quadratically, matching their outsized portfolio impact.

**Evaluation metrics:**
- **R²** - standard regression quality; expected to be low (0.01-0.05) for daily equity returns
- **Direction Accuracy** - % correct sign predictions; directly maps to trading profitability (>50% = edge)
- **IC (Spearman)** - rank correlation between predicted and actual; industry-standard alpha quality metric
- **Sharpe Ratio** - risk-adjusted return of a simplified long/short strategy (no transaction costs)

**Hyperparameters:** Fixed conservative defaults (Ridge alpha=1.0, ElasticNet alpha=0.0001/l1=0.1). No grid search - with many CV folds per evaluation and a small feature set, the computational cost outweighs expected gains.

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

样本外 CV 结果（目标为 `overnight_return`，样本数据集，窗口从 21 到 756 天扩展，公司行为日期已过滤）：

| 模型 | R² | 方向准确率 | 基准准确率 | IC（横截面） | Sharpe（组合） |
|------|------|------------|------------|---------------|----------------|
| Ridge v1 | -0.013 | 0.592 | 0.602 | 0.025 | 2.61 |
| ElasticNet v1 | -0.012 | 0.592 | 0.602 | 0.027 | 2.63 |
| Ridge v2 | -0.012 | 0.593 | 0.603 | 0.042 | 2.82 |
| ElasticNet v2 | -0.013 | 0.590 | 0.603 | 0.028 | 2.45 |

**主要结论：**
- 没有模型超过始终做多基准（约 60% 方向准确率）
- 所有模型较基准低约 1%，说明当前特征更多带来噪声而非有效信号
- 较高的组合 Sharpe（2.5-2.8）主要来自样本期间的牛市环境，不代表模型 alpha
- 横截面 IC 为正但很小（0.02-0.04），选股能力有限
- 所有模型 R² 为负（在日频收益预测中较常见）

## 方法说明

**损失函数：** 所有模型都最小化 MSE。对于连续收益预测，MSE 能更严惩大误差，和实盘风险影响一致。

**评估指标：**
- **R²**：回归拟合优度；日频股票收益通常很低
- **方向准确率**：预测涨跌方向正确的比例；与交易可用性直接相关
- **IC（Spearman）**：预测值与真实值的秩相关；衡量 alpha 质量的常用指标
- **Sharpe Ratio**：简化多空策略的风险调整后收益（未计交易成本）

**超参数：** 使用保守固定值（Ridge alpha=1.0，ElasticNet alpha=0.0001/l1=0.1）。没有做网格搜索：在 CV 折数较多且特征规模较小的前提下，额外计算开销通常大于收益。

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
