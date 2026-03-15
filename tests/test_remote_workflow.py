from suffering.infra import (
    RemoteWorkflowConfig,
    build_full_remote_command,
    build_remote_cli_command,
    build_remote_prepare_command,
    build_ssh_command,
    build_uv_sync_command,
)


def test_build_uv_sync_command_includes_requested_groups() -> None:
    command = build_uv_sync_command("3.12", ("gpu", "dev"))

    assert command == "uv sync --python 3.12 --group gpu --group dev"


def test_build_remote_prepare_command_includes_pull_and_sync_steps() -> None:
    config = RemoteWorkflowConfig(
        ssh_target="gpu-wsl",
        ssh_port=34000,
        remote_dir="/home/kisin/workspace/suffering",
        branch="master",
    )

    command = build_remote_prepare_command(config)

    assert command == (
        "cd /home/kisin/workspace/suffering && "
        "git pull --rebase --autostash origin master && "
        "uv sync --python 3.12 --group gpu"
    )


def test_build_remote_cli_command_escapes_arguments() -> None:
    command = build_remote_cli_command(
        ["train-walkforward", "--model", "xgb_ranker", "--dataset-name", "panel 5d"]
    )

    assert (
        command
        == "uv run suffering train-walkforward --model xgb_ranker --dataset-name 'panel 5d'"
    )


def test_build_full_remote_command_supports_raw_commands_without_pull_or_sync() -> None:
    config = RemoteWorkflowConfig(
        ssh_target="gpu-wsl",
        remote_dir="/home/kisin/workspace/suffering",
    )

    command = build_full_remote_command(
        config,
        raw_command="uv run python scripts/run_nasdaq100_current_static_pipeline.py",
        include_pull=False,
        include_sync=False,
    )

    assert command == (
        "cd /home/kisin/workspace/suffering && "
        "uv run python scripts/run_nasdaq100_current_static_pipeline.py"
    )


def test_build_ssh_command_uses_optional_port() -> None:
    config = RemoteWorkflowConfig(
        ssh_target="gpu-wsl",
        ssh_port=34000,
        remote_dir="/home/kisin/workspace/suffering",
    )

    command = build_ssh_command(
        config,
        "cd /home/kisin/workspace/suffering && uv run suffering doctor",
    )

    assert command == [
        "ssh",
        "-p",
        "34000",
        "gpu-wsl",
        "cd /home/kisin/workspace/suffering && uv run suffering doctor",
    ]
