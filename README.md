# suffering

`suffering` 是一个面向后续持续迭代的 Python 量化研究项目骨架。前几轮都刻意控制范围：先把项目结构、依赖管理、配置读取、命令行入口和测试底座整理好，再补上“最小可用的数据层”和“最小可用的特征工程层”。第四轮在现有 raw data cache + feature cache 之上补上了“最小可用的 label 生成 + panel dataset 组装层”；第五轮继续沿着同样思路推进，在现有 `panel_5d` dataset 之上接入“单次时间顺序切分 + baseline 训练 + 基础评估”的最小闭环；第六轮则在这个 baseline 闭环之上补上了“最小可用的 walk-forward / rolling 时间验证”。

当前仓库已经支持从 `yfinance` 获取最基础的美股日线数据，并以 CSV 形式缓存到本地；也支持在这些已缓存日线数据之上生成最小日频特征表、单 symbol 标签表、一个按 `date + symbol` 对齐的 panel dataset、一个基于 `HistGradientBoostingRegressor` 的 baseline 训练闭环，以及一个更严格的 walk-forward 时间验证闭环。但仍然不包含 LTR/XGBoost/LightGBM、正式回测和远端训练实现。原因很简单：在量化研究项目里，如果数据层、特征层和标签层都还不稳定，就过早引入更复杂的训练与回测，后续很容易在目录组织、配置管理和测试体验上持续返工。

## 当前目录说明

项目采用 `src/` 布局，并预留了后续量化研究常见模块：

- `src/suffering/config/`：配置读取与运行时设置
- `src/suffering/data/`：最小数据层，包括 universe、provider、storage 和 service
- `src/suffering/features/`：最小日频特征工程
- `src/suffering/ranking/`：最小 label 生成与 panel dataset 组装，后续继续扩展训练逻辑
- `src/suffering/backtest/`：后续回测与评估
- `src/suffering/infra/`：后续本地/远端基础设施相关能力
- `tests/`：最小测试与后续测试扩展

## 使用 `uv` 初始化环境

建议在项目根目录执行：

```bash
uv sync --python 3.12
```

这会根据 `pyproject.toml` 创建项目内的 `.venv`，并安装运行依赖与开发依赖。

如果你的本机还没有 Python 3.12，`uv` 会协助解析和使用对应版本。

## 激活 `.venv`

macOS / Linux:

```bash
source .venv/bin/activate
```

激活后可以直接使用 `python`、`pytest`、`ruff` 等命令。

## 配置环境变量

先复制示例配置：

```bash
cp .env.example .env
```

当前只保留少量、确定会用到的变量：

- `APP_ENV`
- `LOG_LEVEL`
- `DATA_DIR`
- `ARTIFACTS_DIR`
- `DEFAULT_DATA_PROVIDER`
- `DEFAULT_START_DATE`
- `DEFAULT_SYMBOLS`

## 运行 CLI

查看欢迎信息：

```bash
uv run suffering
```

检查当前项目环境：

```bash
uv run suffering doctor
```

`doctor` 会输出当前 Python 版本、项目名、默认数据 provider、默认起始日期、默认 symbols、`.env` 是否存在，以及当前状态说明。

## 前四轮已支持的数据层、特征层与 dataset 层

当前已经具备一个最小可用的数据闭环、特征闭环，以及训练前 dataset 闭环：

- 通过 `yfinance` 抓取股票日线 OHLCV 数据
- 统一字段名为 `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `symbol`
- 把原始数据缓存到 `data/raw/daily/<SYMBOL>.csv`
- 通过 service 层优先读缓存，不存在时再下载
- 通过 CLI 命令抓取和查看缓存数据
- 在单个 symbol 的日线数据上构建最基础的日频特征表
- 当前已支持的特征类别包括收益率、波动率、均线位置、日内结构、成交量与成交额
- 把特征表缓存到 `data/features/daily/<SYMBOL>.csv`
- 特征构造严格只使用当日及历史数据，不做未来值填充
- 在单个 symbol 的标准化 raw daily 数据上生成最小标签表
- 把标签缓存到 `data/labels/daily/<SYMBOL>.csv`
- 从多个 symbol 的 feature cache + label cache 合并出 panel dataset
- 把 dataset 缓存到 `data/datasets/daily/panel_5d.csv`
- 为每个交易日的横截面样本生成朴素的 `relevance_5d_5q`

## 当前标签定义

第四轮只支持一个默认 horizon：`5d`，标签定义固定如下：

- 信号日期：`t`
- 特征：只能使用 `t` 当日及以前的数据
- 假设入场价格：`open[t+1]`
- 标签价格：`close[t+5]`
- 连续标签：
  `future_return_5d = close[t+5] / open[t+1] - 1`

实现中刻意保持这个定义不变：

- 不是 close-to-close
- 不是 open-to-open
- 不加入交易成本
- 不加入止损、止盈、过滤器等策略逻辑

末尾没有足够未来数据的行会保留为 `NaN`，不会做 `fillna`；在组装 panel dataset 时，这些 `future_return_5d` 为空的行会被丢弃，因为它们不能作为监督样本。

## 抓取示例数据

抓取默认 universe：

```bash
uv run suffering data-fetch
```

抓取指定 symbol：

```bash
uv run suffering data-fetch AAPL MSFT --start-date 2024-01-01 --end-date 2024-12-31
```

查看本地缓存：

```bash
uv run suffering data-show AAPL
```

原始日线数据默认会缓存到：

```text
data/raw/daily/
```

## 构建和查看特征

先确认原始日线缓存已经存在：

```bash
uv run suffering data-fetch AAPL MSFT --start-date 2024-01-01 --end-date 2024-12-31
```

在本地原始缓存基础上构建特征：

```bash
uv run suffering feature-build AAPL MSFT
```

如果不传 symbol，就会使用默认 universe：

```bash
uv run suffering feature-build
```

查看某个 symbol 的已缓存特征：

```bash
uv run suffering feature-show AAPL
```

特征默认会缓存到：

```text
data/features/daily/
```

当前特征层只负责把原始 OHLCV 数据加工成稳定、可复用的日频特征表。后续轮次会继续在这些特征之上补 label 生成、训练数据组装、排序训练和回测流程。

## 构建和查看 label

先确认 raw cache 已经存在：

```bash
uv run suffering data-fetch AAPL MSFT --start-date 2024-01-01 --end-date 2024-12-31
```

在本地 raw cache 基础上构建 label：

```bash
uv run suffering label-build AAPL MSFT
```

如果不传 symbol，就会使用默认 universe：

```bash
uv run suffering label-build
```

label 默认会缓存到：

```text
data/labels/daily/
```

## 组装和查看 dataset

先确认 feature cache 和 label cache 都已经存在：

```bash
uv run suffering feature-build AAPL MSFT
uv run suffering label-build AAPL MSFT
```

组装默认 panel dataset：

```bash
uv run suffering dataset-build AAPL MSFT
```

如果不传 symbol，就会使用默认 universe：

```bash
uv run suffering dataset-build
```

查看已缓存 dataset：

```bash
uv run suffering dataset-show
```

dataset 默认会缓存到：

```text
data/datasets/daily/panel_5d.csv
```

当前 dataset 会包含：

- `date`
- `symbol`
- 所有已缓存特征列
- `future_return_5d`
- `relevance_5d_5q`

其中 `relevance_5d_5q` 的含义是：对每个交易日，基于当日所有 symbol 的 `future_return_5d` 做横截面排序，再划成最多 5 桶，最差桶记为 `0`。如果某日可用 symbol 数太少，桶数会自然退化，但流程仍然保持稳定可复现。

## 第五轮与第六轮已支持的训练闭环

当前仓库已经可以直接在已缓存的 `panel_5d` dataset 上完成两层最小训练闭环：

- 第五轮：单次时间顺序切分的 baseline 训练
- 第六轮：更严格的 walk-forward / rolling 时间验证

这两层闭环都继续使用同一个 baseline 模型：

- `scikit-learn` 的 `HistGradientBoostingRegressor`

当前训练阶段会：

- 只使用数值特征
- 明确排除 `date`、`symbol`、`future_return_5d`、`relevance_5d_5q`
- 固定预测目标为 `future_return_5d`
- 不做 shuffle，不做随机切分
- 复用同一套最小评估指标和 artifacts 风格

当前会输出的评估指标包括：

- 回归指标：`mae`、`rmse`
- 排序指标：`overall_spearman_corr`
- 按日 rank IC：`daily_rank_ic_mean`、`daily_rank_ic_std`
- 朴素 top-k 指标：`top_5_mean_future_return`、`top_10_mean_future_return`

### 单次时间切分

单次 baseline 训练仍然保留第五轮的实现：

- 按唯一交易日做一次 `60% train / 20% validation / 20% test` 时间切分
- 训练一次 baseline
- 生成 validation / test 预测
- 输出单次切分对应的 metrics report

训练产物默认会写到：

```text
artifacts/models/baseline_hist_gbr.pkl
artifacts/reports/baseline_hist_gbr_metrics.json
artifacts/predictions/baseline_hist_gbr_validation.csv
artifacts/predictions/baseline_hist_gbr_test.csv
```

### 第六轮 walk-forward 如何工作

第六轮新增的 walk-forward 验证继续基于唯一交易日顺序工作，但不再只做一次切分。当前实现保持朴素：

- 先按唯一交易日排序
- 默认用 `20% validation / 20% test`
- train 使用当前 fold 之前全部可用历史日期
- 按 test window 的长度向前滚动
- 每个 fold 都满足 `train < validation < test`
- 日期不重叠，不 shuffle，不随机抽样

它和单次切分训练的区别很直接：

- 单次切分只会得到一组 validation / test 结果
- walk-forward 会得到多组时间上连续前滚的 test 结果
- 汇总结果会按 fold 统计 mean / std / min / max
- test 预测会按 `fold_id + date + symbol` 拼接成一张总表，便于后续继续接更正式的组合评估

walk-forward 产物默认会写到：

```text
artifacts/reports/baseline_hist_gbr_walkforward_summary.json
artifacts/reports/baseline_hist_gbr_walkforward_folds.csv
artifacts/predictions/baseline_hist_gbr_walkforward_test_predictions.csv
```

## 从 raw data 到 walk-forward 验证

下面是一条当前可用的最小闭环命令链：

```bash
uv run suffering data-fetch
uv run suffering feature-build
uv run suffering label-build
uv run suffering dataset-build
uv run suffering train-baseline
uv run suffering train-show
uv run suffering train-walkforward
uv run suffering train-walkforward-show
```

如果你想看某一步缓存内容，也可以分别执行：

```bash
uv run suffering data-show AAPL
uv run suffering feature-show AAPL
uv run suffering dataset-show
```

`train-baseline` 会打印：

- dataset 名称
- 总样本数
- 特征列数与特征列名
- train / validation / test 的样本数和日期范围
- validation / test 的主要指标
- 模型、预测和报告保存路径

`train-walkforward` 会打印：

- dataset 名称
- fold 数
- 每个 fold 的 train / validation / test 日期范围
- 汇总后的主要 test 指标均值
- summary / folds / predictions 的保存路径

`train-show` 会读取已保存的单次切分 metrics report，并检查模型文件、预测文件是否存在。

`train-walkforward-show` 会读取已保存的 walk-forward summary report，并检查 folds / predictions 产物是否存在。

## 当前仍然不支持

- LTR / XGBoost / LightGBM / CatBoost
- 超参数搜索
- 正式回测、累计收益曲线、交易成本、仓位管理
- benchmark 对比
- 远程训练 / GPU / Docker / 数据库 / CI / 调度系统
- 复杂特征预处理 pipeline
- 实验管理平台

## 运行测试

```bash
uv run pytest
```

## 后续规划

当前仓库已经完成“项目骨架 + 最小数据层 + 最小特征层 + 最小 label / dataset 层 + baseline 单次切分训练闭环 + 最小 walk-forward 验证闭环”。下一轮可以按下面的方向逐步扩展：

- `training`：在这套更严格时间验证之上接入更强模型，例如 `XGBoost`
- `ranking`：接 ranking 模型与更贴近排序目标的训练方式
- `backtest`：实现更正式的组合评估、持仓模拟与指标汇总

在这些能力真正落地之前，当前仓库仍然保持小而清晰，避免过早引入复杂抽象。
