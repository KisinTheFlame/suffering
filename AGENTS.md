# AGENTS.md

本文件用于给进入本仓库工作的代理提供最小但明确的协作约定。除非用户另有说明，默认先遵守这里的规则，再结合任务上下文执行。

## 项目概览

- 项目名称：`suffering`
- 类型：Python 量化研究项目骨架
- Python 版本：`3.12`
- 依赖管理与运行入口：`uv`
- 包结构：`src/` 布局
- CLI 入口：`uv run suffering`

当前仓库已经具备以下最小闭环：

- `data`：抓取并缓存日线 OHLCV 数据
- `features`：构建单标的日频特征
- `ranking`：生成 label 并组装 panel dataset
- `training`：单次时间切分训练与 walk-forward 验证
- `backtest`：基于样本外预测做最小组合评估

## 目录约定

- `src/suffering/config/`：配置读取与运行时设置
- `src/suffering/data/`：universe、provider、storage、service
- `src/suffering/features/`：特征定义、转换、缓存
- `src/suffering/ranking/`：label、panel dataset、缓存与服务
- `src/suffering/training/`：模型、切分、评估、walk-forward、artifact 持久化
- `src/suffering/backtest/`：信号、组合、指标、回测结果持久化
- `tests/`：pytest 测试
- `data/`：本地数据缓存，默认不提交
- `artifacts/`：训练/验证/回测产物，默认不提交

## 工作方式

1. 先读相关模块，再改代码；不要在没有确认现有约定的情况下直接重构。
2. 优先做最小增量修改，保持当前“逐轮扩展、避免过度抽象”的项目风格。
3. 尽量沿用已有命名、文件组织和 CLI 风格，不随意发明新层级。
4. 若任务涉及数据、特征、标签、训练、回测中的任一环节，优先检查是否已经存在对应的 `service`、`storage` 或 CLI 子命令。
5. 不要把生成数据、训练产物、缓存文件纳入版本控制，除非用户明确要求。

## 环境与常用命令

首次进入仓库时优先使用：

```bash
uv sync --python 3.12
```

如需加载本地环境变量：

```bash
cp .env.example .env
```

常用命令：

```bash
uv run suffering
uv run suffering doctor
uv run pytest
uv run ruff check .
```

常见流水线命令：

```bash
uv run suffering data-fetch
uv run suffering feature-build
uv run suffering label-build
uv run suffering dataset-build
uv run suffering train-baseline --model hist_gbr
uv run suffering train-walkforward --model xgb_ranker
uv run suffering backtest-walkforward --model xgb_ranker --top-k 5 --holding-days 5 --cost-bps-per-side 5
```

## 修改约定

- 新增 Python 代码时，优先放在 `src/suffering/` 下对应领域模块，不要把业务逻辑堆到 `cli.py`。
- `cli.py` 主要负责参数解析、调用 service、打印结果；复杂逻辑应下沉到模块内部。
- 涉及缓存读写时，优先沿用现有 `storage.py` 模式。
- 涉及业务编排时，优先沿用现有 `service.py` 模式。
- 涉及训练或评估时，保持时间顺序，不要引入随机切分或 shuffle，除非用户明确要求改变研究设定。
- 除非任务明确要求，否则不要顺手修改标签定义、时间切分规则、交易假设、成本假设等研究口径。

## 测试与验证

- 代码改动后，至少运行与改动范围直接相关的测试。
- 如果修改影响 CLI、训练流程、缓存格式或回测产物，优先补或跑对应的 `tests/test_cli_*`、`tests/test_training_*`、`tests/test_backtest_*`。
- 若改动范围较大，建议执行：

```bash
uv run pytest
uv run ruff check .
```

- 如果因为环境、耗时或外部依赖无法完成验证，需要在汇报中明确说明未验证项。

## 提交前检查

- 确认没有误改 `data/`、`artifacts/`、`.venv/`、`__pycache__/` 等生成内容。
- 确认新增依赖已更新 `pyproject.toml`，并说明引入原因。
- 确认 README 中受影响的命令或行为仍然准确；如果不准确，一并更新。

## 沟通偏好

- 默认使用简体中文交流。
- 汇报时优先说明：做了什么、验证了什么、还有什么风险或未完成项。
- 如果需求不清但可以安全假设，先执行并在结果中说明假设；如果会影响研究口径或目录结构，再暂停确认。
