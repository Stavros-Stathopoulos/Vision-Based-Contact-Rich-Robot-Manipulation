"""Orchestrator that runs the full training and evaluation pipeline in order.

Pipeline stages:
    1. Collect expert demonstrations  → expert_demos.npz
    2. Train Behavior Cloning policy  → bc_model.pth
    3. Evaluate Behavior Cloning      → logs success rate
    4. Train RL (SAC) agent           → SAC_200k.zip
    5. Test RL agent (live render)    → evaluation plot

Each stage runs as a separate subprocess so MuJoCo memory is fully
reclaimed between stages. A failing stage aborts the pipeline.

Usage:
    python run_pipeline.py              # Run all stages (skips live render)
    python run_pipeline.py --all        # Run all stages including live render
    python run_pipeline.py --from rl    # Resume from RL training onward
    python run_pipeline.py --only bc    # Run only BC training
"""

import argparse
import logging
import subprocess
import sys
import time
from typing import Optional

logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(_handler)

STAGES: dict[str, dict[str, str]] = {
    "collect": {
        "script": "src/demos/collect_demos.py",
        "description": "Collecting expert demonstrations",
    },
    "bc": {
        "script": "src/models/train_bc.py",
        "description": "Training Behavior Cloning policy",
    },
    "eval_bc": {
        "script": "src/models/evaluate_bc.py",
        "description": "Evaluating Behavior Cloning policy",
    },
    "rl": {
        "script": "src/models/train_rl.py",
        "description": "Training RL (SAC) agent",
    },
    "test_rl": {
        "script": "src/models/test_rl.py",
        "description": "Testing RL agent (live render)",
    },
}

STAGE_ORDER: list[str] = ["collect", "bc", "eval_bc", "rl", "test_rl"]


def run_stage(name: str, python: str) -> bool:
    """Runs a single pipeline stage as a subprocess.

    Args:
        name: Stage key from STAGES dict.
        python: Path to the Python interpreter.

    Returns:
        True if the stage completed successfully, False otherwise.
    """
    stage = STAGES[name]
    logger.info("[%s] %s", name, stage["description"])

    start = time.time()
    result = subprocess.run(
        [python, "-u", stage["script"]],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(
            "[%s] Failed with exit code %d after %.1fs",
            name, result.returncode, elapsed,
        )
        return False

    logger.info("[%s] Completed in %.1fs", name, elapsed)
    return True


def resolve_stages(
    from_stage: Optional[str],
    only_stage: Optional[str],
    include_render: bool,
) -> list[str]:
    """Determines which stages to run based on CLI arguments.

    Args:
        from_stage: If set, start from this stage onward.
        only_stage: If set, run only this single stage.
        include_render: Whether to include the live-render test_rl stage.

    Returns:
        Ordered list of stage keys to execute.

    Raises:
        SystemExit: If an invalid stage name is provided.
    """
    if only_stage:
        if only_stage not in STAGES:
            logger.error(
                "Unknown stage '%s'. Valid stages: %s",
                only_stage, ", ".join(STAGE_ORDER),
            )
            sys.exit(1)
        return [only_stage]

    stages = list(STAGE_ORDER)

    if from_stage:
        if from_stage not in STAGES:
            logger.error(
                "Unknown stage '%s'. Valid stages: %s",
                from_stage, ", ".join(STAGE_ORDER),
            )
            sys.exit(1)
        idx = stages.index(from_stage)
        stages = stages[idx:]

    if not include_render and "test_rl" in stages:
        stages.remove("test_rl")

    return stages


def main() -> None:
    """Parses arguments and runs the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full training and evaluation pipeline.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include the live-render test_rl stage (requires display).",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        choices=STAGE_ORDER,
        help="Resume the pipeline from this stage onward.",
    )
    parser.add_argument(
        "--only",
        dest="only_stage",
        choices=STAGE_ORDER,
        help="Run only a single stage.",
    )
    args = parser.parse_args()

    stages = resolve_stages(args.from_stage, args.only_stage, args.all)
    python = sys.executable

    logger.info("Pipeline stages: %s", " → ".join(stages))

    for name in stages:
        if not run_stage(name, python):
            logger.error("Pipeline aborted at stage '%s'", name)
            sys.exit(1)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
