# suffering

`suffering` 是一个面向后续持续迭代的 Python 量化研究项目骨架。前几轮都刻意控制范围：先把项目结构、依赖管理、配置读取、命令行入口和测试底座整理好，再补上“最小可用的数据层”和“最小可用的特征工程层”。第四轮继续沿着同样思路推进，在现有 raw data cache + feature cache 之上，补上“最小可用的 label 生成 + panel dataset 组装层”，为后续训练切分与 baseline 模型训练准备输入数据。

当前仓库已经支持从 `yfinance` 获取最基础的美股日线数据，并以 CSV 形式缓存到本地；也支持在这些已缓存日线数据之上生成最小日频特征表、单 symbol 标签表，以及一个按 `date + symbol` 对齐的 panel dataset。但仍然不包含训练集/验证集切分、LTR/XGBoost/LightGBM、回测和远端训练实现。原因很简单：在量化研究项目里，如果数据层、特征层和标签层都还不稳定，就过早引入训练与回测，后续很容易在目录组织、配置管理和测试体验上持续返工。

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

## 第四轮已支持的数据层、特征层与 dataset 层

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

当前仍然明确不支持：

- 训练集/验证集/测试集切分
- LTR / XGBoost / LightGBM
- 回测
- 远程训练 / GPU / Docker / 数据库 / 调度系统
- benchmark 下载与相对强弱特征
- 复杂标准化、去极值、中性化流程
- 多 horizon 标签体系

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

## 运行测试

```bash
uv run pytest
```

## 后续规划

当前仓库已经完成“项目骨架 + 最小数据层 + 最小特征层 + 最小 label / dataset 层”。下一轮可以按下面的方向逐步扩展：

- `ranking`：补训练集/验证集切分、baseline 排序模型与训练评估
- `backtest`：实现策略评估、持仓模拟与指标汇总
- `remote training`：补充远端训练、产物同步与任务编排

在这些能力真正落地之前，当前仓库仍然保持小而清晰，避免过早引入复杂抽象。
