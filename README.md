# suffering

`suffering` 是一个面向后续持续迭代的 Python 量化研究项目骨架。前两轮都刻意控制范围：先把项目结构、依赖管理、配置读取、命令行入口和测试底座整理好，再补上“最小可用的数据层”，让后续的数据接入、特征工程、LTR 排序训练、回测验证以及远端训练都能在一个清晰的目录中演进。

当前仓库已经支持从 `yfinance` 获取最基础的美股日线数据，并以 CSV 形式缓存到本地；但仍然不包含特征工程、标签生成、模型训练和回测实现。原因很简单：在量化研究项目里，如果数据层都还不稳定，就过早引入训练与回测，后续很容易在目录组织、配置管理和测试体验上持续返工。

## 当前目录说明

项目采用 `src/` 布局，并预留了后续量化研究常见模块：

- `src/suffering/config/`：配置读取与运行时设置
- `src/suffering/data/`：最小数据层，包括 universe、provider、storage 和 service
- `src/suffering/features/`：后续特征工程
- `src/suffering/ranking/`：后续 LTR 或其他排序训练逻辑
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

## 第二轮已支持的数据层

当前已经具备一个最小可用的数据闭环：

- 通过 `yfinance` 抓取股票日线 OHLCV 数据
- 统一字段名为 `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `symbol`
- 把原始数据缓存到 `data/raw/daily/<SYMBOL>.csv`
- 通过 service 层优先读缓存，不存在时再下载
- 通过 CLI 命令抓取和查看缓存数据

当前仍然明确不支持：

- 特征工程
- label 生成
- 训练集/验证集切分
- LTR / XGBoost / LightGBM
- 回测
- 远程训练 / GPU / Docker / 数据库 / 调度系统

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

后续轮次会在这些已缓存的日线数据之上，继续补特征工程、样本构造和排序训练流程。

## 运行测试

```bash
uv run pytest
```

## 后续规划

当前仓库已经完成“项目骨架 + 最小数据层”。下一轮可以按下面的方向逐步扩展：

- `features`：构建因子、标签、样本生成与特征版本管理
- `ranking`：接入 LTR 或其他排序/打分训练流程
- `backtest`：实现策略评估、持仓模拟与指标汇总
- `remote training`：补充远端训练、产物同步与任务编排

在这些能力真正落地之前，当前仓库仍然保持小而清晰，避免过早引入复杂抽象。
