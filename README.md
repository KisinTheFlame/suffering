# suffering

`suffering` 是一个面向后续持续迭代的 Python 量化研究项目骨架。前几轮都刻意控制范围：先把项目结构、依赖管理、配置读取、命令行入口和测试底座整理好，再补上“最小可用的数据层”和“最小可用的特征工程层”。第四轮在现有 raw data cache + feature cache 之上补上了“最小可用的 label 生成 + panel dataset 组装层”；第五轮继续沿着同样思路推进，在现有 `panel_5d` dataset 之上接入“单次时间顺序切分 + baseline 训练 + 基础评估”的最小闭环；第六轮在这个 baseline 闭环之上补上了“最小可用的 walk-forward / rolling 时间验证”；第七轮引入了第二个回归模型 `xgb_regressor`；第八轮则在同一套 dataset / split / walk-forward / artifact / CLI 框架之上，最小增量接入了 ranking 模型 `xgb_ranker`；第九轮开始在已有 walk-forward test predictions 之上补上“最小可用的组合评估 / 轻量回测层”。

当前仓库已经支持从 `yfinance` 获取最基础的美股日线数据，并以 CSV 形式缓存到本地；也支持在这些已缓存日线数据之上生成最小日频特征表、单 symbol 标签表、一个按 `date + symbol` 对齐的 panel dataset、两个回归模型训练闭环、一个 ranking 模型训练闭环、更严格的 walk-forward 时间验证闭环，以及一个只基于样本外 walk-forward test predictions 的最小组合评估层。但仍然不包含正式生产级回测和远端训练实现。原因很简单：在量化研究项目里，如果数据层、特征层、标签层和训练验证层都还不稳定，就过早引入更复杂的回测与部署，后续很容易在目录组织、配置管理和测试体验上持续返工。

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

## 第八轮已支持的训练闭环

当前仓库已经可以直接在已缓存的 `panel_5d` dataset 上完成四层连续演进后的最小训练闭环：

- 第五轮：单次时间顺序切分的 baseline 训练
- 第六轮：更严格的 walk-forward / rolling 时间验证
- 第七轮：在同一套训练与验证框架内并存支持第二个回归模型 `xgb_regressor`
- 第八轮：在同一套训练与验证框架内最小增量接入 ranking 模型 `xgb_ranker`

当前支持的模型有三个：

- `hist_gbr`：`scikit-learn` 的 `HistGradientBoostingRegressor`
- `xgb_regressor`：`xgboost.XGBRegressor`
- `xgb_ranker`：`xgboost.XGBRanker`

三种模型都会复用同一套训练约束：

- 只使用数值特征
- 明确排除 `date`、`symbol`、`future_return_5d`、`relevance_5d_5q`
- 不做 shuffle，不做随机切分
- 复用同一套 dataset cache、split、walk-forward、artifact 和 CLI 风格

其中回归与 ranking 的训练标签定义分别是：

- `hist_gbr` / `xgb_regressor`：训练标签固定为 `future_return_5d`
- `xgb_ranker`：训练标签固定为 `relevance_5d_5q`

`xgb_ranker` 的 query group 定义也保持非常克制：

- 每个交易日就是一个 query group
- 同一天内的所有 symbol 属于同一个 group
- train / validation / test / walk-forward 每个 split 都基于自己的数据独立重建 groups
- group 顺序严格按 `date` 稳定排序

当前会输出的评估指标包括：

- 回归误差指标：`mae`、`rmse`
- 排序相关指标：`overall_spearman_corr`
- 按日 rank IC：`daily_rank_ic_mean`、`daily_rank_ic_std`
- 朴素 top-k 指标：`top_5_mean_future_return`、`top_10_mean_future_return`
- 轻量 ranking-native 指标：`ndcg_at_5_mean`

对 `xgb_ranker` 而言，评估时仍然会继续结合：

- `future_return_5d`：作为真实未来收益列，用于相关性与 top-k 收益统计
- `relevance_5d_5q`：作为真实相关性，用于计算 `ndcg_at_5_mean`

### 单次时间切分

单次训练继续使用按唯一交易日的 `60% train / 20% validation / 20% test` 时间切分，并会输出 validation / test 预测与 metrics report。

不同模型的单次训练产物会按模型名分别写到：

```text
artifacts/models/hist_gbr.pkl
artifacts/reports/hist_gbr_metrics.json
artifacts/predictions/hist_gbr_validation.csv
artifacts/predictions/hist_gbr_test.csv

artifacts/models/xgb_regressor.pkl
artifacts/reports/xgb_regressor_metrics.json
artifacts/predictions/xgb_regressor_validation.csv
artifacts/predictions/xgb_regressor_test.csv

artifacts/models/xgb_ranker.pkl
artifacts/reports/xgb_ranker_metrics.json
artifacts/predictions/xgb_ranker_validation.csv
artifacts/predictions/xgb_ranker_test.csv
```

### walk-forward 如何工作

walk-forward 验证继续基于唯一交易日顺序工作，保持现有朴素实现：

- 先按唯一交易日排序
- 默认用 `20% validation / 20% test`
- train 使用当前 fold 之前全部可用历史日期
- 按 test window 的长度向前滚动
- 每个 fold 都满足 `train < validation < test`
- 日期不重叠，不 shuffle，不随机抽样

walk-forward 会得到多组时间上连续前滚的 test 结果，汇总结果会按 fold 统计 `mean / std / min / max`，同时把 test 预测按 `fold_id + date + symbol` 拼接成一张总表，便于后续继续接更正式的组合评估。

不同模型的 walk-forward 产物会按模型名分别写到：

```text
artifacts/reports/hist_gbr_walkforward_summary.json
artifacts/reports/hist_gbr_walkforward_folds.csv
artifacts/predictions/hist_gbr_walkforward_test_predictions.csv

artifacts/reports/xgb_regressor_walkforward_summary.json
artifacts/reports/xgb_regressor_walkforward_folds.csv
artifacts/predictions/xgb_regressor_walkforward_test_predictions.csv

artifacts/reports/xgb_ranker_walkforward_summary.json
artifacts/reports/xgb_ranker_walkforward_folds.csv
artifacts/predictions/xgb_ranker_walkforward_test_predictions.csv
```

## 从 raw data 到 walk-forward 验证

下面是一条当前可用的最小闭环命令链：

```bash
uv sync
uv run suffering data-fetch
uv run suffering feature-build
uv run suffering label-build
uv run suffering dataset-build
```

如果你想看某一步缓存内容，也可以分别执行：

```bash
uv run suffering data-show AAPL
uv run suffering feature-show AAPL
uv run suffering dataset-show
```

分别运行两种模型：

```bash
uv run suffering train-baseline --model hist_gbr
uv run suffering train-baseline --model xgb_regressor
uv run suffering train-baseline --model xgb_ranker
uv run suffering train-show --model hist_gbr
uv run suffering train-show --model xgb_regressor
uv run suffering train-show --model xgb_ranker
uv run suffering train-walkforward --model hist_gbr
uv run suffering train-walkforward --model xgb_regressor
uv run suffering train-walkforward --model xgb_ranker
uv run suffering train-walkforward-show --model hist_gbr
uv run suffering train-walkforward-show --model xgb_regressor
uv run suffering train-walkforward-show --model xgb_ranker
```

`train-baseline` 会打印 dataset 名称、当前模型名、任务类型、样本数、特征列、各 split 日期范围、validation / test 指标，以及模型 / 预测 / 报告路径。

`train-walkforward` 会打印 dataset 名称、当前模型名、任务类型、fold 数、每个 fold 的 train / validation / test 日期范围、汇总后的主要 test 指标均值，以及 summary / folds / predictions 的保存路径。

`train-show` 会读取指定模型的单次切分 metrics report，并检查模型文件、预测文件是否存在。

`train-walkforward-show` 会读取指定模型的 walk-forward summary report，并检查 folds / predictions 产物是否存在。

## 第九轮已支持的最小组合评估

第九轮在现有 `walk-forward test predictions` 之上，补上了一个刻意克制的组合评估闭环：

- 只使用 walk-forward 的 `test predictions`
- 对 ranking 模型统一读取 `score_pred`
- 对回归模型统一读取 `y_pred`
- 每个信号日按预测分数降序选择 top-k
- 信号日 `t` 发信号，在 `t+1` 开盘买入，在 `t+5` 收盘卖出
- 每个信号日形成一个 cohort，cohort 内等权
- 组合采用重叠持仓，每个 cohort 固定占总资金的 `1 / holding_days`
- 使用 raw daily price cache 还原日度 gross / net 收益，而不是直接把 `future_return_5d` 当成整条净值曲线

之所以这轮明确只使用 walk-forward 的 test predictions，是为了尽量保证组合评估建立在样本外信号之上，避免把 train / validation 结果混入后高估策略质量。

### 当前组合规则

当前最小策略定义固定如下：

- 信号来源：`artifacts/predictions/<model>_walkforward_test_predictions.csv`
- 默认支持：`hist_gbr`、`xgb_regressor`、`xgb_ranker`
- 选股规则：每个信号日按 signal score 取 top-k
- 持仓规则：`open[t+1]` 买入，`close[t+5]` 卖出
- 权重规则：cohort 内等权，cohort 间固定按 `1 / holding_days` 分配
- 成本规则：默认 `cost_bps_per_side = 5`，即单边 5 bps、往返 10 bps

如果某只股票在持有期内缺少必要 raw price 数据，当前实现会直接跳过这笔 trade，并在 backtest summary 里统计 `skipped_trade_count`。cohort 内剩余可执行股票会重新等权。

### 当前输出内容

最小组合评估现在会输出：

- gross / net daily returns
- gross / net equity curve
- trade 明细
- 组合摘要 summary
- 一组最小组合级指标

当前 summary 至少包括：

- `total_return_gross`
- `total_return_net`
- `annualized_return_gross`
- `annualized_return_net`
- `annualized_volatility`
- `sharpe_ratio_gross`
- `sharpe_ratio_net`
- `max_drawdown_gross`
- `max_drawdown_net`
- `daily_hit_rate_gross`
- `daily_hit_rate_net`
- `average_daily_turnover`
- `average_active_positions`

artifact 会按参数写到：

```text
artifacts/backtests/<model>_top5_h5_cost10_summary.json
artifacts/backtests/<model>_top5_h5_cost10_daily_returns.csv
artifacts/backtests/<model>_top5_h5_cost10_equity_curve.csv
artifacts/backtests/<model>_top5_h5_cost10_trades.csv
```

### 运行最小组合评估

先准备数据、特征、label、dataset 和 walk-forward 预测：

```bash
uv run suffering data-fetch
uv run suffering feature-build
uv run suffering label-build
uv run suffering dataset-build
uv run suffering train-walkforward --model xgb_ranker
```

然后运行最小组合评估：

```bash
uv run suffering backtest-walkforward --model xgb_ranker --top-k 5 --holding-days 5 --cost-bps-per-side 5
uv run suffering backtest-show --model xgb_ranker --top-k 5 --holding-days 5 --cost-bps-per-side 5
```

`backtest-walkforward` 会打印模型名、top-k、holding days、成本假设、信号日期范围、组合日期范围、交易笔数、gross / net 总收益、gross / net Sharpe、gross / net max drawdown，以及 summary / daily returns / equity curve / trades 的保存路径。

`backtest-show` 会读取已保存的 summary，并检查 `daily_returns`、`equity_curve`、`trades` 产物是否存在。

### 为什么这还不是正式回测

这一层仍然只是“最小但可信”的组合评估，不是完整生产级回测，原因包括：

- 还没有 benchmark 对比
- 还没有更细的滑点 / 冲击 / 容量成本模型
- 还没有行业中性、组合约束、风险模型和风控规则
- 还没有更完整的现金管理、停牌处理和执行层假设

后续轮次会继续在这套轻量组合评估之上增量补：

- benchmark 对比
- 更细的成本模型
- 更丰富的组合约束与风控

## 当前仍然不支持

- benchmark 对比
- 复杂正式回测框架
- 行业中性、风险模型、仓位优化
- 更细的滑点 / 冲击成本 / 容量模型
- LightGBM / CatBoost / 多模型 ranking 对比框架
- 超参数搜索
- 远程训练 / GPU / Docker / 数据库 / CI / 调度系统
- 复杂特征预处理 pipeline
- 实验管理平台

## 运行测试

```bash
uv run pytest
```

## 后续规划

当前仓库已经完成“项目骨架 + 最小数据层 + 最小特征层 + 最小 label / dataset 层 + 双回归模型与单个 ranking 模型训练闭环 + 最小 walk-forward 验证闭环 + 最小组合评估层”。后续可以按下面的方向逐步扩展：

- `backtest`：在现有 walk-forward 组合评估之上接 benchmark 对比
- `backtest`：补上更细的成本模型、组合约束与风控

在这些能力真正落地之前，当前仓库仍然保持小而清晰，避免过早引入复杂抽象。
