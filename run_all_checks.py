from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    title: str
    script_path: Path


def run_task(task: Task, python_executable: str, env: dict[str, str]) -> tuple[bool, float]:
    start = time.time()
    print(f"\n开始：{task.title}")
    completed = subprocess.run(
        [python_executable, task.script_path.name],
        cwd=task.script_path.parent,
        check=False,
        env=env,
    )
    elapsed = time.time() - start
    if completed.returncode == 0:
        print(f"完成：{task.title}（{elapsed:.1f}s）")
        return True, elapsed
    print(f"失败：{task.title}（退出码 {completed.returncode}）")
    return False, elapsed


def build_tasks(workspace_dir: Path) -> list[Task]:
    return [
        Task("亮度检测-生成明细", workspace_dir / "wafer_brightnessCheck" / "brightness_analyzer.py"),
        Task("亮度检测-生成评分", workspace_dir / "wafer_brightnessCheck" / "generate_report.py"),
        Task("亮度检测-生成图表", workspace_dir / "wafer_brightnessCheck" / "visualize_brightness.py"),
        Task("清晰度检测", workspace_dir / "wafer_sharpnessCheck" / "main.py"),
        Task("位置偏移检测-批量结果", workspace_dir / "wafer_shiftCheck" / "run_batch.py"),
        Task("位置偏移检测-随机抽样", workspace_dir / "wafer_shiftCheck" / "main.py"),
        Task("形变翘曲检测-批量结果", workspace_dir / "wafer_warpCheck" / "run_batch.py"),
        Task("形变翘曲检测-随机抽样", workspace_dir / "wafer_warpCheck" / "main.py"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parent
    input_root = args.input_path.resolve()
    output_root = args.output_path.resolve()
    if not input_root.exists():
        print(f"失败：输入目录不存在 {input_root}")
        return 1
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(workspace_dir)
    total_start = time.time()
    success_count = 0
    child_env = os.environ.copy()
    child_env["WAFER_DATA_ROOT"] = str(input_root)
    child_env["WAFER_OUTPUT_ROOT"] = str(output_root)

    print(f"工作目录：{workspace_dir}")
    print(f"输入目录：{input_root}")
    print(f"输出目录：{output_root}")
    print("开始执行四个子程序的一键流程")

    for task in tasks:
        if not task.script_path.exists():
            print(f"失败：未找到脚本 {task.script_path}")
            if not args.continue_on_error:
                return 1
            continue
        ok, _ = run_task(task, sys.executable, child_env)
        if ok:
            success_count += 1
        elif not args.continue_on_error:
            print("已停止执行（遇到失败任务）")
            return 1

    total_elapsed = time.time() - total_start
    print(f"\n流程结束：成功 {success_count}/{len(tasks)}，总耗时 {total_elapsed:.1f}s")
    return 0 if success_count == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
