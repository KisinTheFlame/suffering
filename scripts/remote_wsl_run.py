"""Push local commits, update the remote WSL checkout, and run a command there."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence

from suffering.infra import RemoteWorkflowConfig, build_full_remote_command, build_ssh_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the suffering workflow on a remote WSL GPU machine.",
    )
    parser.add_argument("--ssh-target", required=True, help="SSH target such as gpu-wsl")
    parser.add_argument("--ssh-port", type=int, help="Optional SSH port")
    parser.add_argument(
        "--remote-dir",
        required=True,
        help="Remote repository path such as /home/kisin/workspace/suffering",
    )
    parser.add_argument("--branch", help="Git branch to push and pull, defaults to current branch")
    parser.add_argument("--git-remote", default="origin", help="Git remote name")
    parser.add_argument("--python-version", default="3.12", help="Python version for uv sync")
    parser.add_argument(
        "--uv-group",
        action="append",
        default=["gpu"],
        help="uv dependency group to install on remote, can be passed multiple times",
    )
    parser.add_argument("--no-push", action="store_true", help="Skip local git push")
    parser.add_argument("--no-pull", action="store_true", help="Skip remote git pull")
    parser.add_argument("--no-sync", action="store_true", help="Skip remote uv sync")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow local uncommitted changes when --no-push is used",
    )
    parser.add_argument(
        "--raw-command",
        help="Run a raw remote shell command instead of `uv run suffering ...`",
    )
    parser.add_argument(
        "cli_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to `uv run suffering ...` after `--`",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    branch = args.branch or current_branch()
    cli_args = list(args.cli_args)
    if cli_args and cli_args[0] == "--":
        cli_args = cli_args[1:]

    if not args.raw_command and not cli_args:
        parser.error("pass a suffering command after `--`, or use --raw-command")
    if args.allow_dirty and not args.no_push:
        parser.error("--allow-dirty can only be used together with --no-push")

    if not args.no_push:
        ensure_clean_worktree()
        run_command(["git", "push", args.git_remote, branch])
    elif not args.allow_dirty:
        ensure_clean_worktree()

    config = RemoteWorkflowConfig(
        ssh_target=args.ssh_target,
        ssh_port=args.ssh_port,
        remote_dir=args.remote_dir,
        branch=branch,
        git_remote=args.git_remote,
        python_version=args.python_version,
        uv_groups=tuple(args.uv_group),
    )
    remote_command = build_full_remote_command(
        config,
        cli_args=cli_args or None,
        raw_command=args.raw_command,
        include_pull=not args.no_pull,
        include_sync=not args.no_sync,
    )
    run_command(build_ssh_command(config, remote_command))
    return 0


def current_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip()
    if not branch:
        raise SystemExit("Unable to resolve current git branch")
    return branch


def ensure_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise SystemExit(
            "Local worktree is dirty. Commit or stash changes before remote deployment."
        )


def run_command(command: Sequence[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(list(command), check=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
