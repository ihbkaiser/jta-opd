# Hướng dẫn chạy BellmanOPD trên B200

File này là runbook thực hành cho các method trong repository hiện tại: OPD thuần,
TA-OPD, Bellman-RAC, PGT và CMT-OPD. Mọi lệnh đều chạy từ thư mục:

```bash
cd /mnt/hdd/nhatminh/OPD/BellmanOPD
```

`RUN_B200.md` chứa phần giải thích hạ tầng và tuning chi tiết hơn; file này tập trung vào các
lệnh thường dùng có thể copy-paste.

## 1. Chuẩn bị môi trường

Nếu cluster đã có PyTorch/vLLM environment chuẩn cho B200, dùng environment đó. Nếu chưa:

```bash
cd /mnt/hdd/nhatminh/OPD/BellmanOPD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python scripts/check_b200_env.py
```

Đặt model, dataset và tokenizer protocol. Các path tương đối được resolve dưới `STORAGE_ROOT`.
Mặc định là Qwen3-8B teacher, Qwen3-1.7B-Base student và Competition-MATH:

```bash
export STORAGE_ROOT=/workspace/storage-shared
export TEACHER_MODEL=models/Qwen3-8B
export STUDENT_MODEL=nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base
export TRAIN_DATA=nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet
export PROMPT_KEY=problem
```

Preset DAPO-Math:

```bash
unset TRAIN_DATA TRAIN_DATA_PATH PROMPT_KEY TRAIN_PROMPT_KEY TRAIN_DATA_SPLIT
export TRAIN_DATASET=dapo_math
```

Dataset custom:

```bash
unset TRAIN_DATA TRAIN_DATA_PATH PROMPT_KEY TRAIN_PROMPT_KEY TRAIN_DATA_SPLIT
export TRAIN_DATASET=custom
export TRAIN_DATA_PATH=/absolute/path/to/train.parquet
export TRAIN_DATA_SPLIT=null
export TRAIN_PROMPT_KEY=problem
```

Kiểm tra preflight asset/GPU trước khi chạy full:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/smoke_test_b200.sh
```

Smoke FSDP thật (tùy chọn, nhưng nên chạy trước full run):

```bash
CUDA_VISIBLE_DEVICES=0,1 METHOD=opd bash scripts/smoke_test_fsdp_multigpu.sh
CUDA_VISIBLE_DEVICES=0,1 METHOD=cmt bash scripts/smoke_test_fsdp_multigpu.sh
```

Preflight report và autotune được namespace theo model/data fingerprint và số GPU. Không dùng
chung một `OUTPUT_DIR` cho hai experiment khác nhau.

## 2. Shared configuration cho một comparison công bằng

Tạo run name riêng nhưng giữ mọi shared hyperparameter giống nhau:

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export OPD_RUN_NAME="opd_${PAIR}"
export TA_RUN_NAME="ta_${PAIR}"
export RAC_RUN_NAME="rac_${PAIR}"
export PGT_RUN_NAME="pgt_${PAIR}"
export CMT_RUN_NAME="cmt_${PAIR}"

export CUDA_VISIBLE_DEVICES=0,1
export DISTRIBUTED_STRATEGY=fsdp
export BATCH_SIZE=64
export NUM_RESPONSES=1
export MICRO_BATCH_SIZE_PER_GPU=8
export PPO_MINI_BATCH_SIZE=16
export LR=1e-6
export NUM_EPOCHS=1
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=7168
export TOP_K=16
export SAVE_INTERVAL=50
export EVAL_INTERVAL=50
export TRAIN_EVAL_ENABLED=true
export TRAIN_EVAL_NUM_RESPONSES=16
export TRAIN_EVAL_SYNC_TIMEOUT_SEC=86400
export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
```

`MAX_STEPS` là tổng số optimizer steps mục tiêu. Không đặt hoặc đặt `MAX_STEPS=-1` để chạy hết
epoch đã cấu hình; đặt `MAX_STEPS=1` hoặc `2` cho debug.

`BATCH_SIZE` và `PPO_MINI_BATCH_SIZE` đều là global. Với `PPO_MINI_BATCH_SIZE=16`, một global
PPO step dùng 16 trajectories: 16/GPU trên 1 GPU, 8/GPU trên 2 GPU, 4/GPU trên 4 GPU.
`MICRO_BATCH_SIZE_PER_GPU` chỉ điều khiển chunk local; code tự cap nó ở local PPO share khi cần.
Các PPO minibatch hoàn chỉnh được rank-interleave deterministic để mọi rank cùng tham gia một
optimizer step; final partial minibatch được giữ nguyên sample count và sẽ báo lỗi rõ nếu một rank
không có real trajectory để tham gia collective.

## 3. Training

### OPD thuần

```bash
RUN_NAME="$OPD_RUN_NAME" bash scripts/train_opd_b200.sh
```

OPD dùng weight uniform trên mọi response token hợp lệ.

### CMT-OPD

```bash
RUN_NAME="$CMT_RUN_NAME" bash scripts/train_cmt_b200.sh
```

CMT mặc định dùng `CMT_ALLOCATION_KL=0.5`, `CMT_GAMMA=1.0`,
`CMT_SUCCESSOR_LAMBDA=1.0`, Top-K union support và `top_p=1`. Full-vocabulary CMT diagnostics
không bật mặc định:

```bash
CMT_FULL_VOCAB_DIAGNOSTICS=true RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/train_cmt_b200.sh
```

### TA-OPD

```bash
RUN_NAME="$TA_RUN_NAME" bash scripts/train_ta_b200.sh
```

TA hard-select fraction mặc định `TA_RHO=0.10`:

```bash
TA_RHO=0.10 RUN_NAME="$TA_RUN_NAME" bash scripts/train_ta_b200.sh
```

### Bellman-RAC và PGT (nếu cần baseline đầy đủ)

```bash
RUN_NAME="$RAC_RUN_NAME" bash scripts/train_rac_b200.sh
RUN_NAME="$PGT_RUN_NAME" bash scripts/train_pgt_b200.sh
```

Các launcher dùng chung rollout, checkpoint, TensorBoard, teacher scoring và evaluation pipeline.

Training-time evaluation tự dùng toàn bộ GPU training khi `world_size>1`: mỗi rank chạy một
vLLM replica độc lập với `tensor_parallel_size=1`, nhận shard deterministic của benchmark, rồi
rank 0 merge lại thành đúng `summary.json`, prediction files và `model_outputs_detailed.jsonl.gz`
trong `training_eval/step-*`. Trong lúc generation/grade kéo dài, các rank chờ bằng filesystem
sentinel chứ không giữ NCCL barrier; chỉ có collective ngắn sau khi mọi shard đã hoàn tất.
Nếu một rank lỗi, sentinel lỗi được phát hiện và toàn job dừng với thông báo rõ. `sync_timeout_sec`
(mặc định 24 giờ) điều khiển timeout này. Với một GPU, pipeline cũ vẫn được giữ nguyên.
Multi-GPU training-time evaluation yêu cầu `training_evaluation.backend=vllm`; backend `hf` vẫn
dùng được cho single-GPU.
Có thể ghi pass@8 ngay trong periodic evaluation bằng
`TRAIN_EVAL_NUM_RESPONSES=8 TRAIN_EVAL_METRIC=pass@8`.
Các rank phải cùng nhìn thấy `experiment.output_dir` (filesystem dùng chung) để đọc sentinel và
merge shard.
Output mặc định:

```text
outputs/<run-name>/opd/
outputs/<run-name>/ta_opd/
outputs/<run-name>/rac_opd/
outputs/<run-name>/pgt_opd/
outputs/<run-name>/cmt_opd/
```

### Chạy ngắn để kiểm tra launch/config

```bash
MAX_STEPS=1 TRAIN_EVAL_ENABLED=false RUN_NAME="debug_opd" \
  bash scripts/train_opd_b200.sh

MAX_STEPS=1 TRAIN_EVAL_ENABLED=false RUN_NAME="debug_cmt" \
  bash scripts/train_cmt_b200.sh
```

Nếu OOM, giảm cùng một cách cho các method cần so sánh:

```bash
BATCH_SIZE=64 MICRO_BATCH_SIZE_PER_GPU=4 SCORE_MICRO_BATCH_SIZE=4 \
  RUN_NAME="$CMT_RUN_NAME" bash scripts/train_cmt_b200.sh
```

## 4. Resume

### Tự tìm checkpoint mới nhất

Giữ nguyên `RUN_NAME`; `RESUME=auto` sẽ tìm checkpoint hoàn chỉnh mới nhất trong output tương ứng:

```bash
RUN_NAME="$OPD_RUN_NAME" RESUME=auto MAX_STEPS=200 \
  bash scripts/train_opd_b200.sh

RUN_NAME="$TA_RUN_NAME" RESUME=auto MAX_STEPS=200 \
  bash scripts/train_ta_b200.sh

RUN_NAME="$CMT_RUN_NAME" RESUME=auto MAX_STEPS=200 \
  bash scripts/train_cmt_b200.sh
```

`MAX_STEPS=200` ở đây là target cuối cùng, không phải chạy thêm 200 steps.

### Chỉ rõ checkpoint

```bash
RUN_NAME="$CMT_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/${CMT_RUN_NAME}/cmt_opd/checkpoint-000100" \
MAX_STEPS=200 \
  bash scripts/train_cmt_b200.sh
```

Nếu resume với config khác có chủ ý, cần bật rõ:

```bash
RESUME_ALLOW_CONFIG_MISMATCH=true \
  RUN_NAME="$CMT_RUN_NAME" RESUME=auto MAX_STEPS=200 \
  bash scripts/train_cmt_b200.sh
```

Không nên đổi model, tokenizer, dataset, seed, rollout protocol hoặc shared optimizer settings
giữa các lần resume nếu mục tiêu là tiếp tục cùng một experiment.

Số GPU khi resume có thể khác lúc tạo checkpoint. Checkpoint FSDP (`fsdp_full_v1`) được chuyển
về state integer-ID khi chạy single-GPU/non-FSDP; checkpoint standard được chuyển sang full
name-keyed state trước khi FSDP scatter. Mapping được kiểm tra theo tên, thứ tự và số lượng
parameter của optimizer; nếu model/param-group topology khác, chương trình sẽ báo lỗi thay vì
khôi phục nhầm state. Ví dụ chuyển từ 1 GPU sang 2 GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_NAME="$CMT_RUN_NAME" RESUME=auto MAX_STEPS=200 \
  bash scripts/train_cmt_b200.sh
```

## 5. TensorBoard và artifact chính

```bash
tensorboard --logdir_spec \
  "OPD:outputs/${OPD_RUN_NAME}/opd/tensorboard,TA:outputs/${TA_RUN_NAME}/ta_opd/tensorboard,CMT:outputs/${CMT_RUN_NAME}/cmt_opd/tensorboard,RAC:outputs/${RAC_RUN_NAME}/rac_opd/tensorboard" \
  --bind_all --port 6006
```

Các file cần kiểm tra:

```text
outputs/<run>/<method>/resolved_config.yaml
outputs/<run>/<method>/metrics.jsonl
outputs/<run>/<method>/train_metrics.csv
outputs/<run>/<method>/tensorboard/
outputs/<run>/<method>/checkpoint-*/
outputs/<run>/<method>/final/
```

CMT có thêm selector diagnostics như support coverage, raw/conditional common mass, bounded
transition weight, `R`, `M`, `H`, successor excess và sequential gain.

## 6. Evaluation checkpoint cuối

Đánh giá riêng từng method; mặc định dùng vLLM, `temperature=0.7`, `top_p=0.95`, `n=16`:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" bash scripts/eval_opd_b200.sh
TA_RUN_NAME="$TA_RUN_NAME" bash scripts/eval_ta_b200.sh
CMT_RUN_NAME="$CMT_RUN_NAME" bash scripts/eval_cmt_b200.sh
```

Đánh giá accuracy một response:

```bash
EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 \
  CMT_RUN_NAME="$CMT_RUN_NAME" bash scripts/eval_cmt_b200.sh
```

Đổi checkpoint/output trực tiếp:

```bash
CMT_CHECKPOINT="outputs/${CMT_RUN_NAME}/cmt_opd/checkpoint-000100" \
CMT_EVAL_OUTPUT="results/manual_eval/cmt_step100" \
  bash scripts/eval_cmt_b200.sh
```

Kết quả gồm `summary.json`, prediction files và detailed JSONL dưới `results/<comparison>/eval/`
(hoặc thư mục được chỉ định bởi `*_EVAL_OUTPUT`).

### Eval và aggregate tất cả method

`eval_all_b200.sh` luôn chạy Base, OPD, TA và RAC; bật thêm PGT/CMT khi cần:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" \
TA_RUN_NAME="$TA_RUN_NAME" \
RAC_RUN_NAME="$RAC_RUN_NAME" \
CMT_RUN_NAME="$CMT_RUN_NAME" \
RUN_CMT_EVAL=true \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_all_b200.sh
```

Thêm PGT:

```bash
RUN_PGT_EVAL=true RUN_CMT_EVAL=true \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" \
RAC_RUN_NAME="$RAC_RUN_NAME" PGT_RUN_NAME="$PGT_RUN_NAME" \
CMT_RUN_NAME="$CMT_RUN_NAME" bash scripts/eval_all_b200.sh
```

Giới hạn evaluation để debug:

```bash
EVAL_LIMIT=20 EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 \
  CMT_RUN_NAME="$CMT_RUN_NAME" bash scripts/eval_cmt_b200.sh
```

### Eval một checkpoint bất kỳ

```bash
EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 \
  bash scripts/eval_checkpoint_b200.sh cmt \
  "outputs/${CMT_RUN_NAME}/cmt_opd/checkpoint-000100" \
  results/checkpoint_eval/cmt_step100
```

Thay `cmt` bằng `opd`, `ta`, `rac` hoặc `pgt`.

Pass@8 cho một checkpoint bất kỳ:

```bash
EVAL_NUM_RESPONSES=8 EVAL_METRIC=pass@8 EVAL_TEMPERATURE=0.7 \
  CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/eval_checkpoint_b200.sh cmt \
  "outputs/${CMT_RUN_NAME}/cmt_opd/checkpoint-000100" \
  results/checkpoint_eval/cmt_pass8_step100
```

Với `CUDA_VISIBLE_DEVICES` có nhiều GPU, evaluator tự chia benchmark deterministic thành các
shard, chạy một vLLM `TP=1` trên mỗi GPU rồi merge lại; với một GPU behavior không đổi. Có thể
chọn số worker bằng `EVAL_WORLD_SIZE`.

## 7. Re-evaluate toàn bộ checkpoint

Dry-run trước để chỉ kiểm tra danh sách checkpoint, không ghi file:

```bash
REEVAL_METHODS=cmt REEVAL_DRY_RUN=true \
  CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/reeval_method_checkpoints_b200.sh cmt
```

Chạy thật toàn bộ checkpoint của CMT:

```bash
REEVAL_METHODS=cmt \
REEVAL_NUM_RESPONSES=16 REEVAL_TEMPERATURE=0.7 REEVAL_TOP_P=0.95 \
  CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/reeval_method_checkpoints_b200.sh cmt
```

Shortcut re-eval pass@8 (hoạt động cho `opd`, `ta`, `rac`, `pgt`, `cmt`):

```bash
REEVAL_DRY_RUN=true CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/reeval_pass8_b200.sh cmt "$CMT_RUN_NAME"
REEVAL_WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/reeval_pass8_b200.sh cmt "$CMT_RUN_NAME"
```

Pass@8 được lưu riêng trong `eval_history_pass_at_8.jsonl`,
`eval_metrics_pass_at_8.csv`, `checkpoint_reevaluation_manifest_pass_at_8.json` và
`training_eval_pass_at_8/`; history metric khác không bị ghi đè. Chạy lại đúng pass@8 sẽ replace
bộ artifact pass@8 hiện có.

Re-evaluate nhiều method cùng protocol:

```bash
REEVAL_METHODS="opd ta cmt" \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Script này ghi lại `training_eval/step-*`, `eval_history.jsonl` và `eval_metrics.csv` của method
được chọn. Dùng `REEVAL_DRY_RUN=true` trước khi ghi đè.

## 8. Vẽ hình

### Accuracy/loss theo training step

Các run được chọn phải có `eval_history.jsonl`; khi vẽ nhiều method, cần thêm `metrics.jsonl`.

```bash
PLOT_METHODS="opd ta cmt" \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name opd_ta_cmt
```

So sánh đầy đủ:

```bash
PLOT_METHODS="opd ta rac pgt cmt" \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" \
RAC_RUN_NAME="$RAC_RUN_NAME" PGT_RUN_NAME="$PGT_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name all_methods
```

Chỉ vẽ CMT:

```bash
PLOT_METHODS=cmt CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name cmt_progress
```

Ảnh và manifest được ghi dưới:

```text
results/<comparison-name>/plots/<plot-name>/
```

Smoothing mặc định là 10 step; đổi bằng:

```bash
SMOOTHING_WINDOW=1 PLOT_METHODS="opd cmt" \
  OPD_RUN_NAME="$OPD_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name raw_opd_cmt
```

### Vẽ final evaluation bar chart

Sau `eval_all_b200.sh`, dùng:

```bash
RUN_NAME="$OPD_RUN_NAME" \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" \
RAC_RUN_NAME="$RAC_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
RESULTS_DIR="results/${OPD_RUN_NAME}_vs_${TA_RUN_NAME}_vs_${RAC_RUN_NAME}" \
  bash scripts/plot_results.sh --plot-name final_comparison
```

`plot_results.sh` là plot final aggregate; `plot_training_progress.sh` là plot diễn biến theo
checkpoint. Nếu có CMT/PGT trong aggregate, cần truyền đúng output directory và đã chạy eval cho
method đó.

## 9. Một số override thường dùng

```bash
# Chạy trên một GPU để debug DDP/FSDP protocol
CUDA_VISIBLE_DEVICES=0 DISTRIBUTED_STRATEGY=ddp MAX_STEPS=1 \
  RUN_NAME=debug_one_gpu bash scripts/train_cmt_b200.sh

# Không chạy periodic evaluation trong lúc debug
TRAIN_EVAL_ENABLED=false MAX_STEPS=2 \
  RUN_NAME=debug_no_eval bash scripts/train_opd_b200.sh

# Chạy eval nhanh trên một subset
EVAL_LIMIT=20 EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 \
  bash scripts/eval_checkpoint_b200.sh opd \
  "outputs/${OPD_RUN_NAME}/opd/checkpoint-000050"

# Bật sanity check log-prob HF/vLLM cho một rollout
VLLM_LOGPROB_SANITY_ENABLED=true VLLM_LOGPROB_SANITY_FAIL=true \
  MAX_STEPS=1 RUN_NAME=debug_logprob bash scripts/train_cmt_b200.sh
```

Trong comparison chính, không thay đổi riêng một method về batch, rollout length, seed,
temperature, dataset, optimizer hoặc evaluation protocol. CMT training nên giữ `ROLLOUT_TOP_P=1`
để bounded raw-kernel estimator đúng theo scored student distribution; evaluation có thể dùng
`top_p=0.95` như protocol benchmark riêng.

## 10. Troubleshooting nhanh

- **Missing preflight**: chạy lại `bash scripts/smoke_test_b200.sh` với đúng model/data và đúng số GPU.
- **OOM**: giảm `MICRO_BATCH_SIZE_PER_GPU`, sau đó `SCORE_MICRO_BATCH_SIZE`; giữ `BATCH_SIZE` và
  các knob này giống nhau giữa các baseline khi so sánh.
- **`Only ... GiB VRAM is free, below ... headroom` khi re-eval**: đây là thiếu VRAM trên
  GPU đang được chọn, không phải thiếu RAM hệ thống và cũng không phải lỗi pass@8/checkpoint.
  `gpu_memory_utilization=auto` cố ý dừng trước khi khởi tạo vLLM nếu không còn tối thiểu
  `EVAL_VLLM_GPU_HEADROOM_GIB` (mặc định 4 GiB). Kiểm tra process đang giữ GPU:

  ```bash
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  ```

  Dừng các process cũ do chính bạn sở hữu hoặc chọn GPU còn trống. Khi chạy nhiều GPU, phải
  khai báo rõ danh sách GPU; script tự tạo một replica vLLM `TP=1` trên mỗi GPU:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 REEVAL_WORLD_SIZE=2 \
    bash scripts/reeval_pass8_b200.sh cmt "$CMT_RUN_NAME"
  ```

  Không bọc script re-eval này trong `torchrun`; nó tự quản lý các worker vLLM. Có thể truyền
  `EVAL_VLLM_GPU_MEMORY_UTILIZATION=0.90` (hoặc `REEVAL_VLLM_GPU_MEMORY_UTILIZATION=0.90`) chỉ
  sau khi đã xác nhận GPU đủ chỗ; tùy chọn số sẽ bỏ qua kiểm tra headroom và không làm mô hình
  vừa vào GPU chỉ còn 0.7 GiB.
- **Resume báo config mismatch**: kiểm tra `resolved_config.yaml`; chỉ dùng
  `RESUME_ALLOW_CONFIG_MISMATCH=true` khi thay đổi là có chủ ý.
- **`FileExistsError: Refusing to overwrite checkpoint path`**: checkpoint hoàn chỉnh là bất
  biến; dùng `RESUME=auto` hoặc `RESUME_FROM_CHECKPOINT` để tiếp tục từ checkpoint gần nhất,
  không chạy lại đúng cùng step. Nếu lần chạy trước bị ngắt khi đang ghi, rank 0 sẽ tự dọn đúng
  thư mục staging `.checkpoint-*.incomplete` rồi ghi lại an toàn. Các rank phụ không còn kiểm tra
  filesystem trước barrier nên không phát sinh race với thư mục staging của rank 0.
- **Plot thiếu method**: kiểm tra `PLOT_METHODS`, `*_RUN_NAME`, `eval_history.jsonl` và
  `metrics.jsonl` trong output tương ứng.
- **CMT cảnh báo top-p**: đây là cảnh báo đúng; `top_p<1` vẫn bounded nhưng estimator không còn
  unbiased cho raw truncated kernel. Không thêm importance correction thủ công.
