# suffering

`suffering` 是一个面向后续持续迭代的 Python 量化研究项目骨架。第一轮目标不是直接堆出“可跑的策略”，而是先把项目结构、依赖管理、配置读取、命令行入口和测试底座整理好，让后续的数据接入、特征工程、LTR 排序训练、回测验证以及远端训练都能在一个清晰的目录中演进。

这次初始化刻意不包含真实数据下载、特征工程、模型训练和回测实现。原因很简单：在量化研究项目里，前期如果没有稳定的工程骨架，后续很容易在目录组织、环境切换、配置管理和测试体验上持续返工。先把“研究工作台”搭好，后面的每一轮迭代才会更稳。

## 当前目录说明

项目采用 `src/` 布局，并预留了后续量化研究常见模块：

- `src/suffering/config/`：配置读取与运行时设置
- `src/suffering/data/`：后续数据接入与数据集组织
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

当前第一轮只保留少量、确定会用到的变量：

- `APP_ENV`
- `LOG_LEVEL`
- `DATA_DIR`
- `ARTIFACTS_DIR`

## 运行 CLI

查看欢迎信息：

```bash
uv run suffering
```

检查当前项目环境：

```bash
uv run suffering doctor
```

`doctor` 会输出当前 Python 版本、项目名、`.env` 是否存在，以及当前骨架状态说明。

## 运行测试

```bash
uv run pytest
```

## 后续规划

当前仓库只完成了“第一轮项目骨架初始化”。下一轮可以按下面的方向逐步扩展：

- `data`：接入市场数据、基础清洗、数据切片与缓存
- `features`：构建因子、标签、样本生成与特征版本管理
- `ranking`：接入 LTR 或其他排序/打分训练流程
- `backtest`：实现策略评估、持仓模拟与指标汇总
- `remote training`：补充远端训练、产物同步与任务编排

在这些能力真正落地之前，当前仓库先保持小而清晰，避免过早引入复杂抽象。

