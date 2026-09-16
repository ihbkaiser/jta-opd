from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


app = modal.App("bellmanopd-jta-tensorsketch-smoke")
repo = Path("/workspace/BellmanOPD")
image = modal.Image.from_registry("ihbkaiserdev/sparse-vllm:qr-b200")
image = image.run_commands("python -m pip uninstall -y flash-attn")
image = image.pip_install("vllm>=0.17.1,<0.18", "flashinfer-cubin==0.6.4")
image = image.add_local_dir(str(repo), remote_path="/root/BellmanOPD")


@app.function(
    image=image,
    gpu="B200",
    timeout=1800,
)
def run() -> None:
    os.environ["PYTHONPATH"] = "/root/BellmanOPD"
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    from huggingface_hub import snapshot_download

    student = snapshot_download("Qwen/Qwen3-1.7B", local_dir="/tmp/qwen3-1.7b")
    teacher = snapshot_download("Qwen/Qwen3-8B", local_dir="/tmp/qwen3-8b")
    dataset = Path("/tmp/jta-smoke.jsonl")
    dataset.write_text(
        """{"prompt": "Solve 1 + 1. Give only the answer."}
{"prompt": "Solve 2 + 2. Give only the answer."}
{"prompt": "Solve 3 + 3. Give only the answer."}
{"prompt": "Solve 4 + 4. Give only the answer."}
""",
        encoding="utf-8",
    )
    eval_dataset = Path("/tmp/jta-smoke-eval.jsonl")
    eval_dataset.write_text(
        '{"problem": "What is 1 + 1?", "answer": "2", "unique_id": "smoke"}\n',
        encoding="utf-8",
    )
    output = Path("/tmp/jta-tensorsketch-output")
    command = [
        "python",
        "-m",
        "b200_experiment.cli",
        "train",
        "--config",
        "/root/BellmanOPD/configs/qwen3_b200_jta.yaml",
        "--set",
        f"models.student_path={student}",
        "--set",
        f"models.teacher_path={teacher}",
        "--set",
        f"data.path={dataset}",
        "--set",
        "data.split=null",
        "--set",
        "data.prompt_key=prompt",
        "--set",
        "data.max_prompt_tokens=128",
        "--set",
        "data.overlong_prompt_policy=filter",
        "--set",
        "evaluation.benchmark_names=[Competition-MATH]",
        "--set",
        f"evaluation.benchmarks.Competition-MATH.path={eval_dataset}",
        "--set",
        "evaluation.benchmarks.Competition-MATH.question_key=problem",
        "--set",
        "evaluation.benchmarks.Competition-MATH.answer_key=answer",
        "--set",
        "evaluation.benchmarks.Competition-MATH.id_key=unique_id",
        "--set",
        "rollout.batch_size=4",
        "--set",
        "rollout.num_responses=1",
        "--set",
        "rollout.max_new_tokens=32",
        "--set",
        "rollout.vllm.attention_backend=TRITON_ATTN",
        "--set",
        "training.max_steps=1",
        "--set",
        "training.epochs=1",
        "--set",
        "training.micro_batch_size_per_gpu=1",
        "--set",
        "training.ppo_mini_batch_size=4",
        "--set",
        "training.save_checkpoints=false",
        "--set",
        "training.save_optimizer=false",
        "--set",
        "training_evaluation.enabled=false",
        "--set",
        "logging.tensorboard.enabled=false",
        "--set",
        "selector.score_micro_batch_size=1",
        "--set",
        "jta.epsilon=0.1",
        "--set",
        "jta.sketch_dim=512",
        "--set",
        "jta.sketch_seed=42",
        "--set",
        "jta.hidden_sketch_seed=1729",
        "--set",
        "jta.fw_max_iterations=20",
        "--set",
        "jta.kl_bisection_iterations=50",
        "--set",
        "jta.token_chunk_size=4096",
        "--set",
        f"experiment.output_dir={output}",
        "--set",
        "experiment.require_b200=true",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    print(completed.stdout[-12000:], flush=True)
    print(completed.stderr[-12000:], flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"B200 TensorSketch smoke failed: {completed.returncode}")
    metrics_path = output / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    row = rows[-1]
    print(
        "BELLMANOPD_JTA_TENSORSKETCH_SMOKE_JSON "
        + json.dumps(
            {
                "returncode": completed.returncode,
                "method": row.get("method"),
                "loss": row.get("loss"),
                "jta_backend": row.get("jta_embedding_backend"),
                "jta_sketch_dim": row.get("jta_sketch_dim"),
                "jta_hidden_hash_seed": row.get("jta_hidden_hash_seed"),
                "selector_time": row.get("jta_allocation_time"),
                "achieved_kl_max": row.get("selector", {}).get("achieved_kl_max"),
                "coefficient_norm_mean": row.get("selector", {}).get("coefficient_norm", {}).get("mean"),
                "hidden_state_norm_mean": row.get("selector", {}).get("hidden_state_norm", {}).get("mean"),
                "tensorsketch_embedding_norm_mean": row.get("selector", {}).get("tensorsketch_embedding_norm", {}).get("mean"),
                "peak_gpu_allocated_gb": row.get("peak_gpu_allocated_gb"),
                "wall_clock_step_time": row.get("wall_clock_step_time"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with app.run():
        run.remote()
