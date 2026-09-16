# JTA-OPD Implementation Plan for BellmanOPD

## 1. Phạm vi và quyết định thiết kế

Triển khai **JTA-OPD (Joint Token Allocation for OPD)** thành method mới `jta` trong pipeline B200 hiện tại.

JTA-OPD chỉ thay `position_weights`; toàn bộ OPD lõi được giữ nguyên:

- rollout từ student hiện tại;
- student Top-K support (`only_stu`);
- advantage `(teacher_logp - student_logp) * student_p`;
- PPO clipping, dual clipping và `token-mean` aggregation;
- mọi response token hợp lệ đều được supervise.

Hai phạm vi phải được tách rõ:

1. **Reference scope — active rollout batch:** token của prompt `p` được đánh giá bằng các token thuộc mọi active prompt khác trong cùng rollout batch.
2. **Allocation scope — một prompt:** weight của prompt `p` chỉ được tối ưu trên các token thuộc prompt đó. Prompt không lấy hoặc nhường budget cho prompt khác.

Script chính hiện tại dùng:

```text
rollout.batch_size    = 64 prompts
rollout.num_responses = 1 rollout/prompt
```

Implementation không được hard-code `R=1`. Với `R = num_responses >= 1`, candidate set của một prompt phải tự động gồm token từ toàn bộ `R` rollout của prompt đó. Khi sau này đổi thành `R=2,4,...`, reference, allocator, budget và PPO partition vẫn chạy đúng mà không viết lại method.

V1 không dùng suffix, future return, teachability, Bellman recursion hoặc difficulty score. Allocation được tính từ frozen pre-update OPD reference và phải `detach` trước training.

## 2. Các invariant từ code BellmanOPD hiện tại

Plan dựa trên các behavior đang có trong repo:

- `expand_prompt_batch` dùng `repeat_interleave(num_responses)`, nên local trajectory rows có thứ tự prompt-major:

  ```text
  prompt-0/rollout-0, ..., prompt-0/rollout-(R-1),
  prompt-1/rollout-0, ..., prompt-1/rollout-(R-1), ...
  ```

- `padded_local_indices` phân prompt cho rank trước khi expand, nên toàn bộ `R` rollout của một prompt nằm trên cùng rank.
- `objective_valid = rollout.valid_mask & active_trajectories[:, None]`; inactive tail-padding rows không được tham gia reference, allocation, loss hoặc metrics.
- `_opd_train_step` tối ưu global weighted token mean và detach `position_weights`.
- `topk_candidate_ppo_loss` trả một local loss `[B,T]` sau khi sum candidate dimension `K`.
- Baseline OPD truyền weight bằng 1 cho mọi valid token.

Các invariant này phải được giữ và được test trực tiếp.

## 3. Đặc tả toán học

### 3.1 Prompt group và prior

Với active prompt `p`, rollout `r in {1,...,R}` và response position `t`, ký hiệu token state là:

\[
i=(p,r,t).
\]

Candidate set của prompt `p`:

\[
I_p=\{(p,r,t):\operatorname{valid}_{prt}=1\}.
\]

Đặt:

\[
N_p=|I_p|,\qquad
N=\sum_{p=1}^{P}N_p,
\]

trong đó `P` là số active prompt thật trong current rollout batch.

Prior nội bộ prompt:

\[
u_{pi}=\frac{1}{N_p},\qquad i\in I_p.
\]

Đây là prior bắt buộc của primary method vì BellmanOPD đang dùng global token mean. Nó bảo đảm `epsilon=0` tái tạo chính xác OPD: mọi valid token có final loss multiplier bằng 1.

Khi `R>1`, prior này vẫn đều theo token, không đều theo rollout. Rollout dài hơn có nhiều baseline mass hơn, đúng với vanilla OPD. Rollout-balanced prior `1/(R T_{pr})` chỉ được thêm sau này như một ablation có tên riêng vì nó thay đổi baseline objective.

### 3.2 Local OPD gradient embedding

Không materialize per-token gradient trên toàn bộ model parameters. Với FSDP FULL_SHARD và response dài, cách đó không khả thi.

V1 dùng gradient của local OPD/PPO surrogate theo student candidate log-probabilities tại on-policy point. Với:

\[
\ell_i(\theta)=\sum_{k=1}^{K}\ell^{PPO}_{ik}(\theta),
\]

tại thời điểm allocation, `current_log_prob == old_student_log_prob`, ratio bằng 1 và:

\[
\delta_{ik}
=\left.
\frac{\partial\ell_i}
{\partial\log p_\theta(a_{ik}|s_i)}
\right|_{\theta=\theta_{old}}
=-A_{ik},
\]

với `A = opd_reference.advantages`. PPO clipping và dual clipping không thay đổi đạo hàm này tại ratio 1.

V1 dùng TensorSketch có context. Với mỗi valid state `i`, lấy cùng hidden
state cuối cùng trước LM head đã tạo ra logits dùng để xếp hạng candidate:

\[
c_i[v]=\sum_{k:a_{ik}=v}(-A_{ik}),
\qquad
e_i=\operatorname{TensorSketch}(c_i\otimes h_i).
\]

`c_i` là sparse Top-K vocabulary coefficient vector; duplicate candidate IDs
phải được coalesce bằng `scatter_add_`. Không materialize `V x d_h`.

Triển khai TensorSketch bằng hai CountSketch độc lập:

\[
h_v(v)\in\{0,\ldots,d-1\},
\qquad
\sigma_v(v)\in\{-1,+1\},
\]

\[
CS_v(c_i)[b]
=\sum_v c_i[v]\sigma_v(v)\mathbf 1[h_v(v)=b],
\]

\[
CS_h(h_i)[b]
=\sum_j h_i[j]\sigma_h(j)\mathbf 1[h_h(j)=b],
\]

\[
e_i=\operatorname{IFFT}\left(
\operatorname{FFT}(CS_v(c_i))\odot
\operatorname{FFT}(CS_h(h_i))\right).
\]

Yêu cầu:

- vocabulary và hidden hash dùng hai seed deterministic độc lập;
- hash deterministic giữa rank, batch order, CPU và CUDA;
- candidate duplicate/collision và hidden collision được cộng bằng FP32;
- FFT/IFFT và circular convolution chạy trong FP32;
- kernel xấp xỉ `⟨c_i,c_j⟩⟨h_i,h_j⟩`;
- hidden state phải là detached pre-update final representation causal-aligned
  với đúng response logits đang score candidate;
- không normalize riêng từng `e_i`;
- invalid/padded state có hidden sketch và embedding bằng 0;
- metadata gọi đúng backend là `topk_context_tensorsketch`;
- đây là sparse Top-K output-gradient/context proxy, không phải full-parameter
  gradient hoặc exact LM-head gradient vì bỏ dense softmax-tail term.

Không giữ hidden states full batch. Scoring stream từng micro-batch, chuyển
hidden state thành `CS_h(h_i)` FP32 rồi chỉ giữ hidden sketches/compressed
embeddings và các streamed statistics. `token_chunk_size` điều khiển cả
candidate scatter và hidden sketch accumulation.

### 3.3 Batch reference, leave-one-prompt-out

Tính sketch sum của từng prompt trên tất cả `R` rollout:

\[
S_p=\sum_{i\in I_p}e_i.
\]

Tính batch totals bằng một distributed all-reduce:

\[
S=\sum_{p=1}^{P}S_p,
\qquad
N=\sum_{p=1}^{P}N_p,
\qquad
P=\text{number of active prompts}.
\]

Reference cho prompt `p` là toàn bộ valid token thuộc các active prompt khác:

\[
r_{-p}=\frac{S-S_p}{N-N_p}.
\]

Utility của token `i` trong prompt `p`:

\[
a_{pi}=\langle e_i,r_{-p}\rangle.
\]

Không đưa bất kỳ rollout nào của prompt `p` vào reference của chính nó. Khi `R>1`, toàn bộ `R` rollout của `p` đều bị loại khỏi `r_{-p}`.

Nếu `P<2` hoặc `N-N_p=0`, reference không xác định. Khi đó trả prior `w_p=u_p`, log `reference_fallback=true`, và tiếp tục bằng OPD baseline.

### 3.4 Prompt-local optimization objective

Với mỗi prompt `p`, giải một bài toán độc lập:

\[
\max_{w_p}F_p(w_p)
=\sum_{i\in I_p}w_{pi}a_{pi}
-\frac{1}{2}
\left\|
\sum_{i\in I_p}w_{pi}e_i
\right\|_2^2
\]

với:

\[
w_{pi}\ge0,
\qquad
\sum_{i\in I_p}w_{pi}=1,
\qquad
D_{KL}(w_p\|u_p)\le\epsilon.
\]

Ý nghĩa:

- linear term ưu tiên token có gradient direction phù hợp với OPD signal của các prompt khác trong active batch;
- quadratic term xét correlation, cancellation và redundancy giữa các token trong cùng prompt;
- với `R=1`, correlation nằm giữa các positions của một rollout;
- với `R>1`, correlation nằm giữa mọi token của mọi rollout thuộc cùng prompt;
- objective không coupling decision variables giữa hai prompt;
- budget và KL trust region được áp riêng cho từng prompt;
- correlation coefficient cố định `lambda=1`, không có config cho lambda trong V1.

Ma trận ngầm của prompt `p` là:

\[
Q^{(p)}_{ij}=\langle e_i,e_j\rangle,
\qquad i,j\in I_p.
\]

Không materialize `Q`; chỉ tính weighted prompt sketch.

### 3.5 Weight đưa vào OPD loss

Solver trả probability có tổng 1. Chuyển sang convention của `_opd_train_step`:

\[
\omega_{pi}=N_p w_{pi}.
\]

Do đó:

\[
\sum_{i\in I_p}\omega_{pi}=N_p
\]

cho từng prompt, và:

\[
\sum_{p,i}\omega_{pi}=N.
\]

Final training loss vẫn là:

\[
L_{JTA}
=\frac{
\sum_{p,i}\omega_{pi}\ell_{pi}
}{
\sum_{p,i}\omega_{pi}
}.
\]

Khi `epsilon=0`:

\[
w_{pi}=1/N_p
\quad\Longrightarrow\quad
\omega_{pi}=1,
\]

nên JTA phải trùng OPD baseline cả loss và gradient.

## 4. Prompt-local Frank-Wolfe solver

Mỗi prompt được giải độc lập, hoàn toàn trên rank sở hữu prompt đó. Không có collective trong FW iterations.

### 4.1 Gradient và KL linear oracle

Khởi tạo:

\[
w_p^{(0)}=u_p.
\]

Tại một iterate, đặt:

\[
z_p=\sum_iw_{pi}e_i.
\]

Gradient score:

\[
s_{pi}
=\frac{\partial F_p}{\partial w_{pi}}
=a_{pi}-\langle e_i,z_p\rangle.
\]

Linear oracle:

\[
v_p=\arg\max_v\langle s_p,v\rangle
\quad
\text{s.t.}
\quad
v\in\Delta,
\ D_{KL}(v\|u_p)\le\epsilon.
\]

Nghiệm có dạng:

\[
v_{pi}(\tau)
=\frac{u_{pi}\exp(s_{pi}/\tau)}
{\sum_ju_{pj}\exp(s_{pj}/\tau)}.
\]

Tìm `tau` bằng bisection để KL đạt boundary. Trước `exp`, trừ maximum logit để ổn định số.

Các case bắt buộc:

- `epsilon <= constraint_tolerance`: trả prior;
- prompt có một valid token: trả `[1]`;
- mọi score bằng nhau: trả prior;
- epsilon đủ lớn: mass hội tụ vào argmax, chia theo prior nếu tie;
- output finite, không âm, sum bằng 1.

### 4.2 Exact quadratic line search

Đặt:

\[
d_p=v_p-w_p,
\qquad
\Delta z_p=\sum_id_{pi}e_i.
\]

Frank-Wolfe gap:

\[
G_p=\langle s_p,d_p\rangle\ge0.
\]

Exact step:

\[
\gamma_p=
\begin{cases}
\operatorname{clip}
\left(G_p/\|\Delta z_p\|^2,0,1\right),
&\|\Delta z_p\|^2>0,\\
1,&\|\Delta z_p\|^2=0\text{ và }G_p>0,\\
0,&\text{ngược lại}.
\end{cases}
\]

Cập nhật:

\[
w_p\leftarrow(1-\gamma_p)w_p+\gamma_pv_p.
\]

Dừng riêng từng prompt khi:

```text
gap_p <= fw_gap_tolerance * max(1, abs(objective_p))
```

hoặc đạt `fw_max_iterations`.

Implementation có thể vectorize prompts có cùng padded width, nhưng correctness path trước tiên phải xử lý từng prompt với mask rõ ràng. Prompt đã converged không cập nhật nữa; mọi prompt vẫn hoàn tất mà không cần đồng bộ giữa rank.

## 5. Distributed và PPO minibatch behavior

### 5.1 Collective cần thiết

Allocator chỉ cần một nhóm collective trước khi giải local objectives:

- all-reduce `S [d]`;
- all-reduce global valid-token count `N`;
- all-reduce active-prompt count `P`.

Không all-gather token embeddings, token weights hoặc correlation matrix. Không all-reduce trong từng Frank-Wolfe iteration.

Thêm vào `DistributedContext`:

```python
def sum_tensor(self, value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone().contiguous()
    if self.enabled:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result
```

Vector reduction dùng FP32 trên `distributed.device`.

### 5.2 Giữ nguyên prompt group khi `R>1`

Current `distributed_ppo_minibatch_partition` partition theo trajectory rows. Đối với JTA, khi `R>1`, cách này có thể đưa các rollout của cùng prompt vào optimizer steps khác nhau, làm prompt budget không còn được áp nguyên khối.

Thêm `distributed_prompt_group_minibatch_partition`:

1. input là số active prompt trên từng rank, `num_responses=R`, global trajectory `ppo_mini_batch_size` và minibatch index;
2. yêu cầu `ppo_mini_batch_size % R == 0`;
3. tính `prompts_per_ppo_minibatch = ppo_mini_batch_size // R`;
4. rank-interleave ở cấp prompt, không phải trajectory;
5. prompt được chọn phải expand thành đủ local rows `[prompt*R, ..., prompt*R+R-1]`;
6. không bao giờ chia `R` rollout của một prompt qua hai optimizer steps.

Trong `_opd_train_step`, dùng grouped partition khi method là `jta`; các method cũ giữ nguyên partition hiện tại.

Với primary run `R=1`, grouped partition suy biến thành trajectory partition và giữ behavior hiện tại. Do đó primary config tiếp tục dùng `ppo_mini_batch_size=16`; không bắt buộc gom cả 64 trajectories vào một optimizer step.

Khi `R>1`, startup validation yêu cầu:

```text
ppo_mini_batch_size % num_responses == 0
ppo_mini_batch_size / num_responses >= world_size
```

Một active prompt group phải nằm trọn trong một optimizer step. Allocation/reference vẫn được tính một lần trên full rollout batch, giống frozen OPD reference hiện tại; các later PPO minibatches dùng detached allocation đã tính trước.

## 6. Thay đổi theo file

### 6.1 Thêm `b200_experiment/selectors/jta_selector.py`

API đề xuất:

```python
@dataclass
class JTAOutput:
    weights: torch.Tensor          # omega [B,T], invalid = 0
    probabilities: torch.Tensor    # w [B,T], invalid = 0
    diagnostics: dict[str, Any]

class CountSketchOperator:
    # Explicit ablation: top-k log-probability-only CountSketch.
    def reduce(self, coefficients, candidate_ids, token_weights, mask): ...
    def dot(self, coefficients, candidate_ids, vectors, mask): ...
    def squared_norm(self, coefficients, candidate_ids, token_weights, mask): ...

class TensorSketchOperator:
    vocab_hash_seed: int
    hidden_hash_seed: int
    def reduce_vocab(self, coefficients, candidate_ids, mask): ...
    def reduce_hidden(self, hidden_states): ...
    def convolve(self, vocab_sketch, hidden_sketch): ...
    def coefficient_norm(self, coefficients, candidate_ids, mask): ...

def kl_ball_linear_oracle(scores, prior, epsilon, ...): ...

class JTASelector:
    def allocate(
        self,
        reference: TopKOPDReference,
        valid_mask: torch.Tensor,
        active_prompt_mask: torch.Tensor,
        num_responses: int,
        distributed: DistributedContext,
        hidden_sketches: torch.Tensor | None = None,
        hidden_state_norm: torch.Tensor | None = None,
    ) -> JTAOutput: ...
```

`allocate` thực hiện đúng thứ tự:

1. validate `B == local_prompt_slots * R`;
2. reshape prompt-major thành `[local_prompt_slots,R,T,...]`;
3. loại inactive prompt khỏi mọi count/reduction;
4. lấy `delta = -reference.advantages`, áp `support_mask` nếu có;
5. với backend context, nhận hidden `CS_h(h_i)` đã stream từ cùng forward
   causal với candidate logits; zero mọi invalid/padded state;
6. tạo `CS_v(c_i)`, FFT circular-convolve với `CS_h(h_i)`, chỉ giữ compressed
   embeddings theo prompt chunk;
7. tính `S_p`, `N_p`, rồi distributed `S`, `N`, `P`;
8. tính `r_-p` và utility của từng token;
9. giải một Frank-Wolfe problem cho mỗi active prompt;
10. chuyển probability thành multiplier `omega=N_p*w_p`;
11. scatter về đúng `[B,T]` ban đầu;
12. materialize output thành normal detached tensors, không giữ inference tensor.

Thêm export `JTASelector`, `JTAOutput`, `kl_ball_linear_oracle` trong `selectors/__init__.py`.

Không reuse `cmt_selector.kl_constrained_allocation`: CMT helper giải một global linear objective với uniform prior, còn JTA cần repeated prompt-local KL oracles bên trong quadratic Frank-Wolfe.

### 6.2 Sửa `b200_experiment/distributed.py`

Thêm:

- `DistributedContext.sum_tensor`;
- `distributed_prompt_group_minibatch_partition`;
- validation và docstring chỉ rõ prompt-major row contract.

Không thay behavior của `distributed_ppo_minibatch_partition` cho method cũ.

### 6.3 Sửa `b200_experiment/trainer.py`

Các insertion point:

1. Thêm `jta: JTA-OPD` vào `METHOD_DISPLAY_NAMES` và method whitelist.
2. Khởi tạo `JTASelector` từ `config["jta"]`.
3. Validation:
   - `num_responses >= 1`;
   - rollout batch phải có ít nhất 2 active prompt cho normal path;
   - `epsilon >= 0`, `sketch_dim > 0`, iteration counts dương;
   - PPO group constraints ở mục 5.2;
   - OPD controlled settings vẫn là `only_stu/student_p/token_reward_direct/token-mean`.
4. JTA dùng standard OPD student/teacher scoring; không bật PGT union hoặc TA cross-scoring.
5. Student scoring phải lấy final hidden states trong cùng forward với response
   logits, slice `prompt_width - 1 : prompt_width - 1 + T`, rồi stream thành
   `CS_h(h_i)`; không dùng hidden state lệch một token.
6. Sau `build_topk_opd_reference`, gọi `jta_selector.allocate` với
   `objective_valid`, active prompt mask, runtime `num_responses` và hidden
   sketches/norms.
7. `SelectorOutput.scores = jta.weights`; `selected = objective_valid` vì JTA không hard-drop token.
8. Gather các scalar token diagnostics chỉ cho logging: `utility`, `redundancy`, `marginal_score`, `coefficient_norm`, `hidden_state_norm`, `tensorsketch_embedding_norm`, `w_probability`, `w`.
9. Truyền `jta.weights` vào `_opd_train_step`.
10. Truyền `prompt_group_size=num_responses` để JTA dùng prompt-group PPO partition.
11. Bổ sung metadata và final metrics:
    - reference: `active_batch_leave_one_prompt_out`;
    - allocation: `independent_within_prompt`;
    - candidate scope: `all_rollouts_of_prompt`;
    - prior: `uniform_valid_tokens_within_prompt`;
    - backend `topk_context_tensorsketch`, dimension, vocabulary/hidden hash seeds;
    - explicit note that dense softmax-tail/full-parameter gradient is omitted;
    - fixed lambda 1;
    - `all_response_tokens_supervised=true`.

### 6.4 Diagnostics và logging

Sửa:

- `b200_experiment/diagnostics.py`;
- `b200_experiment/selector_logging.py`;
- `b200_experiment/tensorboard_logging.py`.

Token summaries:

- `utility = a_i`;
- `redundancy = <e_i,z_p>`;
- `marginal_score = utility - redundancy`;
- `embedding_norm`;
- `coefficient_norm = ||c_i||_2` after duplicate coalescing;
- `hidden_state_norm = ||h_i||_2`;
- `tensorsketch_embedding_norm = ||e_i||_2`;
- `w_probability`;
- `w = omega`;
- effective sample size và ESS/valid-token ratio.

Prompt-local objective/solver summaries, aggregate bằng mean/std/min/max trên active prompts:

- initial/final objective và improvement;
- linear term;
- quadratic term;
- FW iterations, final gap, converged fraction;
- last/mean gamma;
- achieved KL, KL boundary fraction;
- ESS ratio, maximum-to-mean weight ratio, matching residual;
- probability sum error và multiplier budget error;
- prompt token count;
- reference fallback count.

Batch-reference summaries:

- active prompt count và valid token count;
- norm của global batch mean sketch;
- norm của `r_-p`, aggregate theo prompt;
- utility positive/negative fraction;
- sketch/reference/solver/gather time;
- sketch dimension, backend và estimated temporary bytes.

Khi `R>1`, thêm per-prompt rollout-mass summaries:

- probability mass của từng rollout;
- min/max/std giữa các rollout;
- số prompt có một rollout chiếm trên 90% mass.

Với `R=1`, rollout mass luôn bằng 1; không ghi histogram vô nghĩa, chỉ ghi `num_responses=1` trong metadata.

TensorBoard tối thiểu:

```text
jta/objective_improvement_mean
jta/linear_term_mean
jta/quadratic_term_mean
jta/fw_gap_mean
jta/fw_iterations_mean
jta/fw_converged_fraction
jta/kl_mean
jta/kl_max
jta/weight_std
jta/weight_max
jta/effective_token_fraction
jta/reference_norm_mean
jta/coefficient_norm_mean
jta/hidden_state_norm_mean
jta/tensorsketch_embedding_norm_mean
jta/ess_ratio_mean
jta/max_to_mean_weight_ratio_mean
jta/matching_residual_mean
jta/reference_fallback_count
system/jta_allocation_time
```

`TokenScoreStatsLogger` thêm histogram cho `utility`, `redundancy`,
`coefficient_norm`, `hidden_state_norm`, `tensorsketch_embedding_norm`,
`embedding_norm`, `w_probability`, `w`. Metadata/manifest ghi backend, sketch
dimension và cả hai hash seeds.

Không dùng `SelectedTokenLogger` để dump mọi JTA token. Thêm bounded debug logger, mặc định tắt:

```yaml
logging:
  jta_debug_enabled: false
  jta_debug_interval: 50
  jta_debug_prompts_per_rank: 2
  jta_debug_tokens_per_prompt: 32
```

Khi bật, ghi prompt summary và top/bottom tokens theo multiplier. Mỗi token sample gồm dataset index, rollout index, response position, token ID/text, utility, redundancy, coefficient norm, hidden-state norm, TensorSketch norm, prior, probability và multiplier. Không ghi raw sketch, hidden states hoặc full logits.

## 7. Config và scripts

### 7.1 Primary config hiện tại: một rollout

Thêm `configs/qwen3_b200_jta.yaml`:

```yaml
_base_: qwen3_b200_base.yaml

experiment:
  method: jta
  output_dir: outputs/jta_opd

rollout:
  batch_size: 64
  num_responses: 1

training:
  ppo_mini_batch_size: 16

jta:
  epsilon: 0.1
  sketch_dim: 512
  sketch_seed: 42
  hidden_sketch_seed: 1729
  embedding_backend: topk_context_tensorsketch
  fw_max_iterations: 20
  fw_gap_tolerance: 1.0e-5
  kl_bisection_iterations: 50
  constraint_tolerance: 1.0e-6
  token_chunk_size: 4096
```

Không thêm `lambda` vào config; V1 cố định bằng 1. `epsilon` là allocation hyperparameter chính. `topk_logprob_countsketch` vẫn tồn tại nhưng chỉ được chọn explicit cho ablation. Sau correctness tests, chạy ablation ít nhất `{0.0, 0.1, 0.5, 1.0}` và backend comparison.

### 7.2 Runtime extension cho nhiều rollout

Không tạo code path khác cho `R>1`. Launcher đọc `NUM_RESPONSES` hoặc config override, còn allocator luôn lấy runtime `rollout.num_responses`.

Nếu tăng `R`, người chạy phải đặt `PPO_MINI_BATCH_SIZE` chia hết cho `R` và đủ ít nhất một prompt/rank. Ví dụ world size 2:

```text
R=2, PPO_MINI_BATCH_SIZE=16 -> 8 prompts/update
R=4, PPO_MINI_BATCH_SIZE=16 -> 4 prompts/update
```

Không cần đổi objective, reference hoặc weight normalization.

### 7.3 Launch/evaluation integration

Thêm `scripts/train_jta_b200.sh` theo launcher pattern hiện tại và cập nhật:

- `train_all_b200.sh`;
- `eval_all_b200.sh` và `eval_jta_b200.sh`;
- `eval_checkpoint_b200.sh`;
- `reeval_method_checkpoints_b200.sh`;
- `reeval_all_checkpoints_b200.sh`;
- `smoke_test_fsdp_2gpu.sh`;
- plotting method registry nếu JTA xuất hiện trong comparison plots.

Hỗ trợ aliases `jta` và `jta-opd`; default output `outputs/jta_opd`. Primary launcher phải explicit `NUM_RESPONSES=1`, nhưng không reject giá trị 2/4 nếu PPO group constraints hợp lệ.

## 8. Runtime checks

Fail fast nếu vi phạm:

1. `B == local_prompt_slots * num_responses`.
2. Toàn bộ rows của một prompt nằm trên cùng rank và liên tiếp.
3. Mỗi active prompt có `N_p>0`; inactive prompt đóng góp đúng 0.
4. Sketch, utility, objective, probabilities và multipliers đều finite.
5. Valid weights không âm; invalid/padded weights đúng 0.
6. `abs(sum(w_p)-1) <= 1e-5` cho từng prompt.
7. `KL(w_p||u_p) <= epsilon + 1e-5`.
8. `abs(sum(omega_p)-N_p) <= 1e-4*max(1,N_p)`.
9. Global multiplier mass bằng global valid-token count.
10. `epsilon=0` cho multiplier bằng 1 trên mọi valid token trong `1e-6`.
11. Frank-Wolfe objective của mỗi prompt không giảm ngoài numerical tolerance.
12. Reference của `p` không chứa bất kỳ token nào thuộc prompt `p`.
13. Khi `R>1`, PPO minibatch không chia prompt group.
14. Rollout tensors không bị selector sửa đổi.
15. Allocation tensor detached trước `_opd_train_step`.

Chỉ `P<2` được fallback về prior. NaN, sai grouping, KL violation hoặc budget violation phải raise.

## 9. Test plan

### 9.1 Unit tests: `tests/test_jta_selector.py`

1. `-advantages` trùng autograd derivative của `topk_candidate_ppo_loss` tại ratio 1.
2. Vocabulary/hidden CountSketch deterministic theo coordinate và independent seed, không phụ thuộc batch order/device.
3. Duplicate candidate IDs được coalesce đúng vào sparse `c_i`; invalid candidate/state mask tạo embedding zero.
4. FFT TensorSketch trùng explicit small `c_i \otimes h_i` outer-product sketch.
5. Chunked TensorSketch và inner products xấp xỉ kernel `⟨c_i,c_j⟩⟨h_i,h_j⟩`.
6. Causal hidden slice của mỗi response state khớp chính xác logits candidate cùng forward, không lệch một token.
7. `r_-p` đúng bằng mean sketch của mọi active prompt khác.
8. Tất cả rollout của prompt hiện tại bị loại khỏi reference khi `R=2/4`.
9. `R=1` grouping và allocation đúng với variable response lengths.
10. `R=2/4` candidate set gồm mọi valid token trong prompt, không lẫn prompt khác.
11. KL oracle thỏa nonnegative/sum/KL; epsilon 0 và constant score trả prior; large epsilon tiến tới argmax.
12. Frank-Wolfe giữ feasibility và objective không giảm.
13. Nghiệm gần exhaustive-grid optimum cho prompt có 2 token.
14. Một active prompt trong batch kích hoạt prior fallback.
15. Tail-padded prompt không ảnh hưởng batch reference.
16. Probability scatter-back khớp row/position gốc.

### 9.2 Distributed and integration tests

- `sum_tensor` trùng single-process reference.
- World size 1 và 2 cho cùng `r_-p`, utility và local weights trên cùng synthetic batch.
- Distributed TensorSketch hashes/embeddings nhất quán khi prompt partition thay đổi.
- Prompt-group partition với `R=1,2,4` không split group và cover mỗi real trajectory đúng một lần.
- Uneven prompt tail giữ active/inactive semantics.
- `epsilon=0` JTA và OPD cho cùng weighted loss và parameter gradient.
- `epsilon=0` với context backend vẫn khôi phục chính xác vanilla OPD loss/gradient.
- Mỗi prompt giữ multiplier mass `N_p` trong từng PPO update chứa nó.
- Primary config `64x1`, PPO minibatch 16 tạo đúng behavior hiện tại.
- Invalid `ppo_mini_batch_size % R != 0` bị reject.
- Resume scheduling vẫn đúng với số PPO minibatches thực tế.
- JSON metrics, score stats và TensorBoard tags không có NaN.
- Launcher nhận `jta`/`jta-opd` và không regression method cũ.

### 9.3 B200 smoke sequence

1. 1 GPU, `4 prompts x 1 rollout`, response ngắn, `epsilon=0`: so loss/grad với OPD.
2. 1 GPU, `4 x 1`, `epsilon=0.1`: kiểm tra local objectives tăng và constraints.
3. 2 GPU FSDP, `8 x 1`: kiểm tra batch reference collective và không deadlock.
4. 2 GPU FSDP, `4 prompts x 2 rollouts`: kiểm tra generic prompt grouping và PPO partition.
5. 2 GPU, primary `64 x 1`: đo allocation time và peak memory.
6. Chỉ sau đó chạy max response length 7168 và full evaluation schedule.

## 10. Thứ tự triển khai

1. Viết independent vocabulary/hidden CountSketch maps, FP32 TensorSketch
   convolution và explicit small outer-product reference implementation.
2. Viết KL oracle và prompt-local Frank-Wolfe cùng unit tests.
3. Thêm batch totals all-reduce và LOPO reference.
4. Thêm generic grouping theo runtime `R`; test `R=1,2,4`.
5. Thêm prompt-group PPO partition và distributed tests.
6. Nối causal hidden sketch vào frozen `TopKOPDReference` và hoàn thành
   `epsilon=0` OPD regression.
7. Nối trainer/config primary `64x1`, giữ ablation backend explicit.
8. Thêm metrics, TensorBoard và bounded debug logger.
9. Thêm launch/evaluation/plot integration.
10. Chạy full existing test suite rồi B200 smoke sequence.

## 11. Definition of done

Implementation sẵn sàng cho experiment khi:

- primary launcher chạy `64 prompts x 1 rollout`;
- cùng implementation chạy đúng `R=2/4` mà không đổi objective hoặc allocator code;
- reference của mỗi prompt dùng toàn active batch trừ chính prompt đó;
- objective, correlation penalty, KL constraint và budget đều prompt-local;
- không có transfer budget giữa prompts;
- `epsilon=0` tái tạo OPD loss và gradient;
- không materialize full per-token parameter gradients, dense `E [N,d]` hoặc `Q [N_p,N_p]`;
- grouped PPO partition không split các rollout của cùng prompt;
- distributed results không phụ thuộc rank partition và không dùng padding làm reference;
- runtime checks và logs đủ để xác định lỗi reference, grouping, utility, redundancy, KL solver và normalization;
- test suite cũ không regression và toàn bộ JTA unit/integration/B200 smoke tests pass;
- metadata mô tả đúng V1 là `topk_context_tensorsketch`: sparse Top-K
  output-gradient/context proxy với dense softmax-tail và full-parameter
  gradient bị loại bỏ;
- old `topk_logprob_countsketch` chỉ chạy khi được chọn explicit cho ablation;
- causal alignment test chứng minh mỗi hidden sketch dùng đúng state feeding
  candidate logits;
- diagnostics ghi backend, sketch dimension, cả hai hash seeds, coefficient /
  hidden / TensorSketch norms, achieved KL, ESS ratio, max-to-mean ratio,
  matching residual và solver convergence.
