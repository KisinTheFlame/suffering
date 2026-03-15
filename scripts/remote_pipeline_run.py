"""Run the full remote research pipeline and fetch key artifacts back locally."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from suffering.infra import (
    RemotePipelineSpec,
    build_local_pipeline_output_dir,
    build_pipeline_artifact_relative_paths,
    build_remote_pipeline_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full remote research pipeline and fetch report artifacts back.",
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
    parser.add_argument("--model", default="xgb_ranker", help="Model name")
    parser.add_argument("--top-k", type=int, default=5, help="Backtest top-k selection size")
    parser.add_argument(
        "--holding-days",
        type=int,
        default=5,
        help="Backtest holding window in trading days",
    )
    parser.add_argument(
        "--cost-bps-per-side",
        type=float,
        default=5.0,
        help="Single-side transaction cost in basis points",
    )
    parser.add_argument(
        "--with-robustness",
        action="store_true",
        help="Also run backtest-robustness and fetch its outputs",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Artifacts directory relative to the remote repository root",
    )
    parser.add_argument(
        "--local-output-dir",
        help="Local directory used to store fetched pipeline artifacts",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Run the remote pipeline but do not fetch artifacts back locally",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    spec = RemotePipelineSpec(
        model_name=args.model,
        top_k=args.top_k,
        holding_days=args.holding_days,
        cost_bps_per_side=args.cost_bps_per_side,
        include_robustness=args.with_robustness,
    )

    run_remote_pipeline(args, spec)
    if args.skip_fetch:
        return 0

    local_output_dir = (
        Path(args.local_output_dir)
        if args.local_output_dir
        else build_local_pipeline_output_dir(spec)
    )
    local_output_dir.mkdir(parents=True, exist_ok=True)
    fetched_files = fetch_pipeline_artifacts(
        ssh_target=args.ssh_target,
        ssh_port=args.ssh_port,
        remote_dir=args.remote_dir,
        artifacts_dir=args.artifacts_dir,
        local_output_dir=local_output_dir,
        relative_paths=build_pipeline_artifact_relative_paths(spec),
    )
    remote_commit = read_remote_commit(
        ssh_target=args.ssh_target,
        ssh_port=args.ssh_port,
        remote_dir=args.remote_dir,
    )
    manifest_path = write_pipeline_manifest(
        local_output_dir=local_output_dir,
        spec=spec,
        ssh_target=args.ssh_target,
        ssh_port=args.ssh_port,
        remote_dir=args.remote_dir,
        remote_commit=remote_commit,
        fetched_files=fetched_files,
    )

    report_path = (
        local_output_dir / "reports" / "research" / f"{spec.model_name}_research_report.md"
    )
    print(f"local_output_dir: {local_output_dir}")
    print(f"local_report_path: {report_path}")
    print(f"manifest_path: {manifest_path}")
    return 0


def run_remote_pipeline(args: argparse.Namespace, spec: RemotePipelineSpec) -> None:
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
        build_remote_pipeline_command(spec),
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


def fetch_pipeline_artifacts(
    *,
    ssh_target: str,
    ssh_port: int | None,
    remote_dir: str,
    artifacts_dir: str,
    local_output_dir: Path,
    relative_paths: Sequence[PurePosixPath],
) -> list[str]:
    fetched_files: list[str] = []
    remote_artifacts_root = PurePosixPath(remote_dir) / artifacts_dir
    for relative_path in relative_paths:
        remote_path = remote_artifacts_root / relative_path
        local_path = local_output_dir / Path(relative_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        command = ["scp"]
        if ssh_port is not None:
            command.extend(["-P", str(ssh_port)])
        command.extend([f"{ssh_target}:{remote_path.as_posix()}", str(local_path)])
        run_command(command)
        fetched_files.append(str(local_path))
    return fetched_files


def read_remote_commit(*, ssh_target: str, ssh_port: int | None, remote_dir: str) -> str:
    command = ["ssh"]
    if ssh_port is not None:
        command.extend(["-p", str(ssh_port)])
    command.extend([ssh_target, f"cd {remote_dir} && git rev-parse HEAD"])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def write_pipeline_manifest(
    *,
    local_output_dir: Path,
    spec: RemotePipelineSpec,
    ssh_target: str,
    ssh_port: int | None,
    remote_dir: str,
    remote_commit: str,
    fetched_files: Sequence[str],
) -> Path:
    manifest = {
        "model_name": spec.model_name,
        "top_k": spec.top_k,
        "holding_days": spec.holding_days,
        "cost_bps_per_side": spec.cost_bps_per_side,
        "include_robustness": spec.include_robustness,
        "ssh_target": ssh_target,
        "ssh_port": ssh_port,
        "remote_dir": remote_dir,
        "remote_commit": remote_commit,
        "fetched_files": list(fetched_files),
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
