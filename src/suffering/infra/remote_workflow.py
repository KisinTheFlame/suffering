"""Helpers for the local-dev plus remote-WSL execution workflow."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteWorkflowConfig:
    ssh_target: str
    remote_dir: str
    branch: str = "master"
    git_remote: str = "origin"
    python_version: str = "3.12"
    uv_groups: tuple[str, ...] = ("gpu",)
    ssh_port: int | None = None


def build_uv_sync_command(
    python_version: str,
    uv_groups: Sequence[str] = ("gpu",),
) -> str:
    command: list[str] = ["uv", "sync", "--python", python_version]
    for group in uv_groups:
        command.extend(["--group", group])
    return shlex.join(command)


def build_remote_prepare_command(
    config: RemoteWorkflowConfig,
    *,
    include_pull: bool = True,
    include_sync: bool = True,
) -> str:
    commands = [f"cd {shlex.quote(config.remote_dir)}"]
    if include_pull:
        commands.append(
            "git pull --rebase --autostash "
            f"{shlex.quote(config.git_remote)} {shlex.quote(config.branch)}"
        )
    if include_sync:
        commands.append(build_uv_sync_command(config.python_version, config.uv_groups))
    return " && ".join(commands)


def build_remote_cli_command(cli_args: Sequence[str]) -> str:
    if not cli_args:
        raise ValueError("Remote CLI command requires at least one suffering subcommand")
    return "uv run suffering " + shlex.join(list(cli_args))


def build_full_remote_command(
    config: RemoteWorkflowConfig,
    *,
    cli_args: Sequence[str] | None = None,
    raw_command: str | None = None,
    include_pull: bool = True,
    include_sync: bool = True,
) -> str:
    if bool(cli_args) == bool(raw_command):
        raise ValueError("Provide exactly one of cli_args or raw_command")

    commands = [
        build_remote_prepare_command(
            config,
            include_pull=include_pull,
            include_sync=include_sync,
        )
    ]
    if raw_command is not None:
        commands.append(raw_command)
    else:
        commands.append(build_remote_cli_command(cli_args or ()))
    return " && ".join(commands)


def build_ssh_command(config: RemoteWorkflowConfig, remote_command: str) -> list[str]:
    command = ["ssh"]
    if config.ssh_port is not None:
        command.extend(["-p", str(config.ssh_port)])
    command.append(config.ssh_target)
    command.append(remote_command)
    return command
