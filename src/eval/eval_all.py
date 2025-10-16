import os
import sys
from src import PROJECT_ROOT, get_logger
import subprocess
import itertools

logger = get_logger()


def main():
    os.chdir(PROJECT_ROOT)
    eval_scripts = ["src.eval.eval_contactnet", "src.eval.eval_gaitnet"]
    difficulties = [0.0, 0.05, 0.01, 0.15, 0.2]
    speeds = [0.05, 0.1, 0.15, 0.2]

    for difficulty, speed, script in itertools.product(
        difficulties, speeds, eval_scripts
    ):
        logger.info(f"Running {script} with difficulty {difficulty} and speed {speed}")
        subprocess_args = [
            sys.executable,
            "-m",
            script,
            "--difficulty",
            str(difficulty),
            "--velocity",
            str(speed),
            "--headless",
            "--trials",
            "1",
            "--num_envs",
            "50",
            "--no-log-file",
        ]
        logger.debug(f"Running command: {' '.join(subprocess_args)}")
        subprocess.run(subprocess_args)


if __name__ == "__main__":
    main()
