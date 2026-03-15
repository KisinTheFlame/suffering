"""Run the static current Nasdaq-100 main experiment remotely and fetch artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from suffering.infra import (
    MODEL_NAME,
    RUN_NAME,
    local_artifacts_output_dir,
    remote_artifacts_relative_root,
    report_relative_path,
)

REMOTE_SCRIPT_PATH = "scripts/run_nasdaq100_current_static_pipeline.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the static current Nasdaq-100 experiment on the remote WSL GPU machine.",
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
        "--local-output-dir",
        help="Local directory used to store fetched experiment artifacts",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Run the remote experiment but do not fetch artifacts back locally",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_remote_experiment(args)
    if args.skip_fetch:
        return 0

    local_output_dir = (
        Path(args.local_output_dir) if args.local_output_dir else local_artifacts_output_dir()
    )
    if local_output_dir.exists():
        shutil.rmtree(local_output_dir)
    local_output_dir.parent.mkdir(parents=True, exist_ok=True)

    fetch_artifacts(
        ssh_target=args.ssh_target,
        ssh_port=args.ssh_port,
        remote_dir=args.remote_dir,
        local_output_dir=local_output_dir,
    )
    remote_commit = read_remote_commit(
        ssh_target=args.ssh_target,
        ssh_port=args.ssh_port,
        remote_dir=args.remote_dir,
    )
    manifest_path = write_manifest(
        local_output_dir=local_output_dir,
        ssh_target=args.ssh_target,
        ssh_port=args.ssh_port,
        remote_dir=args.remote_dir,
        remote_commit=remote_commit,
    )

    report_path = local_output_dir / Path(report_relative_path())
    print(f"local_output_dir: {local_output_dir}")
    print(f"local_report_path: {report_path}")
    print(f"manifest_path: {manifest_path}")
    return 0


def run_remote_experiment(args: argparse.Namespace) -> None:
    command = [
        "uv",
        "run",
        "python",
        "scripts/remote_wsl_run.py",
        "--ssh-target",
        args.ssh_target,
        "--remote-dir",
        args.remote_dir,
        "--git-remote",
        args.git_remote,
        "--python-version",
        args.python_version,
        "--raw-command",
        f"uv run python {REMOTE_SCRIPT_PATH}",
    ]
    if args.ssh_port is not None:
        command.extend(["--ssh-port", str(args.ssh_port)])
    if args.branch:
        command.extend(["--branch", args.branch])
    for group in args.uv_group:
        command.extend(["--uv-group", group])
    if args.no_push:
        command.append("--no-push")
    if args.no_pull:
        command.append("--no-pull")
    if args.no_sync:
        command.append("--no-sync")
    if args.allow_dirty:
        command.append("--allow-dirty")
    run_command(command)


def fetch_artifacts(
    *,
    ssh_target: str,
    ssh_port: int | None,
    remote_dir: str,
    local_output_dir: Path,
) -> None:
    command = ["scp"]
    if ssh_port is not None:
        command.extend(["-P", str(ssh_port)])
    remote_source_path = Path(remote_dir) / remote_artifacts_relative_root()
    remote_source = f"{ssh_target}:{remote_source_path.as_posix()}"
    command.extend(["-r", remote_source, str(local_output_dir.parent)])
    run_command(command)


def read_remote_commit(*, ssh_target: str, ssh_port: int | None, remote_dir: str) -> str:
    command = ["ssh"]
    if ssh_port is not None:
        command.extend(["-p", str(ssh_port)])
    command.extend([ssh_target, f"cd {remote_dir} && git rev-parse HEAD"])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def write_manifest(
    *,
    local_output_dir: Path,
    ssh_target: str,
    ssh_port: int | None,
    remote_dir: str,
    remote_commit: str,
) -> Path:
    manifest = {
        "run_name": RUN_NAME,
        "model_name": MODEL_NAME,
        "remote_script_path": REMOTE_SCRIPT_PATH,
        "ssh_target": ssh_target,
        "ssh_port": ssh_port,
        "remote_dir": remote_dir,
        "remote_commit": remote_commit,
    }
    manifest_path = local_output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def run_command(command: Sequence[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(list(command), check=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
