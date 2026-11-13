# LLM Load test

Please refer to the [`benchmark_suite.ipynb`](benchmark_suite.ipynb) for a detailed example of how to use the load_test.py script and run different types of benchmark suites.

## Installation

The load test relies on [Locust package](https://locust.io/). Install it from pip.

```bash
pip install -r requirements.txt
```

Then run the commands described below from the enclosing directory. Locust will pick up the settings from `locust.conf` automatically.

## Usage

The load test script exercises LLM generation endpoint under varying load. See below for the common configuration options. Check `--help` for the full list.

### Target

- `-H`: target endpoint URL (preceding `/v1/...`). E.g. `-H http://localhost` or `-H https://api.fireworks.ai/inference`. Defaults to `localhost:80`.
- (optional) `-m`: model to send requests too. Can be omitted for a local test if the server has a single model loaded only.
- (optional) `--provider`: provider name like `fireworks` or `openai`. APIs have slight differences that the script accounts for. If omitted the script tries to guess based on URI and API return information. Must be specified for non-OpenAI-compatible providers like Triton.
- `-k`: API key to be passed as `Authorization: Bearer ...`.

### Rate of requests

There are several primary modes the script can be used:

1. **Fixed concurrency**. N workers are created. Each sends a request, waits for the response and then sends the next request. Thus as concurrency increases, the server will get more loaded and latency will grow. Usually increasing concurrency beyond some point doesn't increase throughput and just leads to growing latency.
   - `-u`: the number of concurrent workers to spawn (standard Locust argument)
   - `-r`: the rate per second of spawning concurrent workers. If processing workload takes a while (more than several seconds), it makes sense to set this value to something lower than `-u` for a gradual ramp-up to avoid request bursts.
   - (optionally) `--burst <period in seconds>`: synchronizes all N workers to issue requests in one go with the specified interval. The maximum latency should be less than the period, otherwise some workers may fall behind.

2. **Fixed QPS**. The script ensures that input requests are issued at specific times to average out at the specified rate per second. Sending is open loop: each request is sent in its own greenlet at its scheduled time, however long earlier responses take. If the target QPS is too high and the server is overloaded it will likely drop additional requests or stall.
   - `--qps`: the desired rate of requests per second. Can be a fractional number, e.g. `0.1`.
   - `-u <number> -r <same number>`: each user prepares its next request before its send time, so `-u` bounds how many requests can be prepared ahead of a burst; something like `-u 100 -r 100` is a good choice. Sending starts once all users have spawned (and, with `--shape-mix`, after the prompt pool is built, which happens before spawning so it never runs inside the measured window). The script warns when sends leave more than 50 ms late.
   - (optional) `--qps-max-lateness S` (default 0.5): a send more than `S` seconds behind schedule means the generator stalled. The schedule then moves later once, so the oldest pending send is due now, and every send claimed before the shift moves with it: what fell due during the stall keeps its spacing, with no burst, no hole and nothing skipped. The summary reports `Schedule Shift S` (and `Late Sends Skipped`, now normally 0).
   - (optional) `--qps-distribution`: specify how to space out requests. Default is `constant` meaning evenly spaced out. `exponential` is an option simulating [Poisson distribution](https://en.wikipedia.org/wiki/Traffic_generation_model#Poisson_traffic_model). `gamma` draws gamma inter-arrival times with mean `1/qps` and squared coefficient of variation `--qps-cv2` (default 1, which is the same as `exponential`). `--qps-cv2` above 1 clumps arrivals into bursts like production traffic; below 1 spaces them more evenly.
   - (optional) `--qps-modulation-window W --qps-dispersion D`: a renewal process (any `--qps-distribution`) can't reproduce production's overdispersed request counts over longer windows. With these flags the rate is redrawn every `W` seconds from a gamma with mean `--qps`, sized so request counts in 10-second windows have index of dispersion (variance / mean) `D`; arrivals keep `--qps-distribution` within each window. Rate variation alone raises the gap CV², so for `exponential` and `gamma` the within-window CV² is lowered to keep the overall gap CV² at `--qps-cv2` (1 for exponential); the summary's `Concurrency` shows the within-window value, and a warning says when the target can't be reached. Rates are drawn in blocks covering the whole `-t` run (30 windows without `-t`) and scaled to average exactly `--qps`, so every load tier sends its nominal rate (for D=42 at 30 QPS, a 120 s tier's rate spread drops from ±3.8 to ±0.44 QPS standard deviation). With `--shape-mix`, `D` defaults to the profile's `arrivals.dispersion_10s` (or `arrivals.router_level.dispersion_10s`).
   - The schedule is seeded from `--request-seed` (or the shape mix's `request_seed`), so runs with the same seed send at the same offsets.
   - Delivered send times are stamped when the request has been written to the socket (not before the POST call), so they include connection setup and request writing.
   - The summary reports the intended and delivered gap CV² and 1 s / 10 s count dispersion (averaged over bin alignments), plus send lateness P50/P99/max, so a generator that can't keep up shows in the results. The start-up is reported apart from the run: `Startup Offset S` is the first send's delay after all users spawned, and `Startup Drain S` / `Startup Drain Peak Ms` are the leading run of sends more than min(1 s, 0.5% of the schedule's span) late, which a pacer that doesn't resync (replay) works through after spawn; `Send Lateness Max After Drain Ms` is the largest lateness after it, the figure to check a step on. A step that never catches up reports `Startup Drain Complete` 0, with the after-drain figure set to the drain's peak. `Client In Flight Samples` records the client's paced requests not yet completed (sent on the wire, still waiting or streaming) every `--client-in-flight-interval` seconds (default 15, 0 off), with a least-squares `Client In Flight Slope Per Min` fitted over the window's second half (past the warm-up ramp from 0 to rate × request time) and its 95% interval (`... Ci95 Low` / `High`): a slope whose interval excludes zero means requests are accumulating, including ones stuck before they reach the server. The interval accounts for the samples' autocorrelation (counts of requests lasting about a sample interval are correlated, which made a plain interval read a healthy server as rising about a third of the time). It needs the fitted half to hold at least about 20 samples (a 10-minute step at the default 15 s), and fitting the second half assumes requests shorter than half the step; shorter steps get a wide interval or none. Also reported: the client's CPU (`Client Cpu Percent`, `Client Cpu Peak 5S Percent`): a client above about 90% sends late and evenly, so results from that run are unreliable. With `--qps` or `--replay-schedule`, an `Arrivals so far: N sends, delivered gap CV2 (idle rule) X` line is logged every `--arrival-log-interval` seconds (default 60, 0 off), and the summary reports `Delivered Gap Cv2 Idle Rule` and `Intended Gap Cv2 Idle Rule` (serving's idle-restart rule). Each log line recomputes over every send so far, so its cost grows with the sends: milliseconds at a few QPS, about half a second per line at 1,000 QPS over ten minutes, which would show as send lateness. Raise the interval or pass 0 for very high-rate steps...
   - `--qps` is per process: a second process sends the full rate again. Sends are open loop and not capped in flight, so against a stalled server at a high rate the client can run out of file descriptors (raise `ulimit -n` or lower the rate). `Timed Sends` can be higher than the requests counted at the end of a run, because it includes sends still waiting for a response.
   - `--summary-file` appends take a lock (`<file>.lock`), so parallel runs can share one file. A run with columns the file lacks (an older client's file, or `--qps` columns after a plain run) extends the header and leaves earlier rows blank there, with a warning; code that reads these files by column position should read by name.

### Workload

Input is read from --dataset, which is either:
- `limerics`: default dataset. Requires --tokenizer to be passed. Will be used to auto-generate realistic prompts.
- `@`-prefixed, specifies a path to JSONL file, used to read contents of each request.

The number of tokens to generate is sampled on every request from a given distribution:
- `-o`/`--max-tokens`: maximum number of tokens to generate. If --max-tokens-distribution is non-constant this is going to be the mean of the distribution.
- `--max-tokens-distribution`: specifies probability distribution to use.
- `--max-tokens-range` Specifies "the width" of the distribution (e.g. stddev for "normal" distribution). Specified value `alpha` is relative to `max-tokens`. Default is 0.3 so most of the range falls in "3 sigma" region.
- `--max-tokens-cap`: specify upper bound to "truncate" the probability distribution. The lower bound is always 1 token. This allows to sample from "truncated normal" or "truncated exponential" distributions.

Based on the above settings the following distributions are supported:
- `constant`: use `--max_tokens` value on every request
- `uniform`: sample from the range `[max_tokens - max_tokens * alpha, max_tokens + max_tokens * alpha]`
- `normal`: sample from gaussian distribution `N(max_tokens, max_tokens * alpha)`
- `exponential`: sample from exponential distribution with the mean `max_tokens`. `alpha` is ignored

The benchmark makes the best effort to ensure the desired `max_tokens` number is respected:
- for providers that support it, it passes `ignore_eos` or `min_tokens` parameter to avoid early stopping
- the default prompt is a lengthy code generation request that usually doesn't stop early
- it verifies the number of tokens actually generated and prints warnings on mismatch. Different providers use varying mechanisms of returning generated number of tokens. For some of them `--logprobs` might be needed in the streaming mode.
- optionally, `--tokenizer` can be passed specifying Huggingface tokenizer to be used to count the output tokens on client side.

Class-mix workload (`--shape-mix`), for traffic shaped like a production profile instead of a single prompt length with one shared prefix:
- `--shape-mix '<json>'` or `--shape-mix @profile.json`. The JSON is an object with `seed` (default 0), optional `request_seed` (defaults to `seed`), optional `arrivals.cv2` and a `classes` list; a traffic profile can be passed as is, other top-level fields are ignored. Each class has `weight`, `prompt_tokens`, `cached_prompt_tokens` (default 0), `generated_tokens`, `prefix_pool_size` (default 1, with a warning for classes that have cached tokens) and an optional `name`. Token counts may be fractional (profile means) and are rounded. Every class needs `prompt_tokens` of at least the chat template overhead (0 with `--no-chat`) plus 32 for the unique tail; shorter classes are rejected at startup, before any request is sent. Without `name`, the class is named from `cache_class` and `prompt_band` (`marginal` when both are null, `class<i>` when absent), made unique with a `_2`, `_3` suffix; names appear in the summary columns and the affinity key. Other unknown class fields are ignored with a warning.
- `arrivals.cv2` is the default for `--qps-cv2` when the flag isn't given (it only applies with `--qps-distribution gamma`).
- Each request picks a class by weight. Its prompt is one of the class's `prefix_pool_size` shared prefixes, `cached_prompt_tokens` long, followed by a unique tail up to exactly `prompt_tokens` (sized with `--tokenizer`, including the chat template in `--chat` mode). The tail starts with a per-request nonce, so a full prompt never repeats and a class with `cached_prompt_tokens: 0` stays cold.
- `--prefix-reuse-overhang N`: tokens before a divergence point that the server can't reuse from its prefix cache. DeepSeek V4 keeps sliding-window KV only for a sequence's last 128 tokens, so where a prompt diverges from a cached one it recomputes the 128 tokens before that point and reports 126 to 129 fewer cached tokens than were shared. Shared prefixes are made `N` tokens longer (and the unique tail `N` shorter), so the server reuses exactly `cached_prompt_tokens`. Defaults to the model config's `sliding_window` for `deepseek_v4` models, else 0.
- `--prefix-restore-boundary N`: the block (tokens) the server restores cached KV in. GLM 5.x (`glm_moe_dsa`, DSA) restores only whole 2,048-token blocks, so a shared prefix of N tokens reuses floor(N / 2048) × 2048. Cached lengths (a class's `cached_prompt_tokens`, a replay row's `cached_length`) are aligned down to a block, and shared prefixes are laid out 16 tokens past it, so a cut a few tokens short of the boundary keeps the block; `Reusable Tokens` and `Laid Out Cached Tokens` report the aligned values. Defaults to the model config's `index_topk` for `glm_moe_dsa` models, else 0 (token-exact reuse); `Prefix Restore Boundary` is in the summary.
- Requests of classes with `prefix_pool_size` of 2 or more send `x-session-affinity: shape-<seed>-<class>-<prefix index>`, so routing keeps each prefix on one replica as in production. Cold classes and single-prefix classes send no affinity header, since one key would pin the whole class to one replica.
- Output length is forced per class: `max_tokens = min_tokens = generated_tokens` with `ignore_eos: true`. With `--shape-mix-output-mode max`, only `max_tokens = generated_tokens` is sent (no `min_tokens`, and no `ignore_eos`, even for providers that add it), so the model stops where it would; the per-class `Completion Tokens` shows what it generated.
- Prompts of different prefix slots, and cold prompts, share no leading text beyond the chat template head, the tail start and rarely one nonce word (at most 7 tokens with the DeepSeek V4 template), so they diverge inside the first KV block.
- Prefixes depend only on `seed`, so every worker and every run uses the same prefix working set. Every prefix is built at startup, before pacing starts, so no build work lands inside the measured window; the log reports the build time, the pool's resident size and process RSS. Each prefix is a window of a cyclic word corpus, so the pool stores 8 bytes per prefix (offset and width) and slices the text per request; a 250M-token pool needs well under 1 MiB, with building memory bounded by a 1M-token batch.
- `--shape-mix-tail-variants N` (default 16): precomputed fillers per class after each request's nonce; `0` fits a fresh filler for every request (tokenizer work per request). On a Qwen3.6-35B-A3B deployment with 3-token MTP speculation, 16 fillers measured 1.5 to 2.8 points higher acceptance per draft position than fresh fillers (two paired runs each).
- A tail is a nonce of fixed token length (single-token words encoding random bits, the worker index and the request counter) followed by one of 16 precomputed fillers of the class, so building a request needs no tokenizer call (about 0.03 ms) and doesn't delay other sends. If the tokenizer can't give the nonce a fixed length, tails are fitted per request instead.
- Class choice, prefix choice and the tail nonce come from `request_seed`, overridable with `--request-seed`; with `--qps`, the arrival times are seeded from it too. Give each run against the same deployment its own request seed, or later runs replay earlier runs' tails and hit them in the cache. Reuse a request seed to send identical traffic, e.g. to compare an incumbent and a candidate. Distributed workers get different tails.
- Replaces `-p`, `-o`, `--dataset` and `-pcml`, which can't be combined with it. Works with `--chat` and `--no-chat`, streaming or not.
- `--shape-mix-corpus NAME|PATH`: the text prefixes and tails are cut from: `prose` (default: 1.7 MB, about 370k tokens, of public-domain novels, Pride and Prejudice, The Adventures of Sherlock Holmes and Frankenstein, shipped as `prose_corpus.txt.gz` with `prose_corpus.LICENSE.txt`), `limericks` (about 4.9k tokens, so longer prompts repeat it), `code` (1.76 MB, about 393k Llama-3 tokens, of CPython 3.12 standard-library source under the PSF license, shipped as `code_corpus.txt.gz` with `code_corpus.LICENSE.txt`), or a text file, optionally gzipped (paragraphs separated by blank lines, shuffled by seed). Part of the `--prefix-pool-cache` key.
- Prefix popularity: by default every prefix of a pool is equally likely. A class's optional `prefix_popularity` is either `{"zipf": s}` (prefix j drawn with weight 1 / (j + 1)^s) or a list of `prefix_pool_size` weights; `--prefix-popularity-zipf S` sets a Zipf default for classes without one. Draws use a cumulative-weight array (fast for million-prefix pools); `--prefix-fill` still sends every prefix once, and `--replay-schedule` entries without a `prefix_index` draw by popularity.
- `--shape-mix-sample-lengths` (off by default): draw each request's uncached prompt tokens and output tokens from the class's optional `uncached_tokens_quantiles` and `generated_tokens_quantiles` (`{"p10": n, "p50": n, "p90": n}` or `[[0.1, n], [0.5, n]]`), instead of the class means. The cached share stays `cached_prompt_tokens`. Lengths follow log(value) linear in the normal z of the quantile between the given points, extrapolated past them (capped at p100, else 4x the top point), so every given quantile is kept. Only what the quantiles don't pin moves to keep the class mean (`prompt_tokens` minus the reusable share for uncached tokens, `generated_tokens` for output): the upper tail's slope, then a blend toward linear interpolation; when no such choice reaches the mean (or a class gives fewer than two interior quantiles), all values are scaled, with a warning. Uncached lengths take 19 strata (16 equal ones with the top one split into four down to the top 0.3%, plus the given quantile levels), each at its exact quantile average and drawn by its width, so the top quantile is reached and requests still need no tokenizer call; output lengths are drawn continuously (forced in `exact` mode, a cap in `max` mode). Tails are floored at what the nonce and instruction need (counted in the mean), with a warning. Classes without quantiles keep their means, and with the flag off the quantiles are ignored entirely.
- `--shape-mix-instruction TEXT|@file`: an instruction appended after a blank line to every prompt (e.g. `Summarize the text above.`), useful with `--shape-mix-output-mode max` so the model has something to answer. It counts toward `prompt_tokens`, the unique tail is shortened to fit, and classes too short for it are rejected at startup.
- `--prefix-pool-cache DIR`: keep built prefix windows in `DIR`, so later llm_bench processes (e.g. one per load step) load the pool instead of rebuilding it; a 40.6M-token pool loads in about 0.1 s instead of being rebuilt in about 22 s (8 cores). Files are keyed by seed, class, prefix length, tokenizer and corpus, not by request seed or pool size: a larger pool loads what's cached and builds only the missing prefixes. Writes are atomic, so concurrent processes can share the directory; unreadable files are rebuilt. The summary reports `Prefix Pool Prefixes`, `Tokens`, `Loaded From Cache` and `Build S`.
- `--replay-schedule FILE.jsonl`: send at recorded times instead of `--qps`. One request per line: `{"timestamp": <ms>, "class": "<class name>"}` (AIPerf-style millisecond timestamps; `"time_s"` in seconds also works), optionally `"prefix_index"` (a prefix of the class pool, drawn otherwise) and `"output_length"` (overrides the class's output length). Times are replayed at recorded speed (no speed-up) relative to the first entry, open loop with the same lateness handling and delivered-arrival stats as `--qps`. Prompts come from the class prefix pools and tails as usual. The run ends once every request is sent and answered (or at `-t`). Classes and prefix indexes are checked against `--shape-mix` at startup.
  - Row fields (task 15's format): `input_length` (alias `prompt_tokens`), `cached_length` (alias `cached_tokens`; if any row has it, every row must), `output_length` (alias `generated_tokens`), `cancelled`, `temperature`, `top_p`, `reasoning_effort`, `session_id`, `prefix_index`, `pod`. An invalid optional setting (`temperature`, `top_p`, `reasoning_effort`) is dropped for that row and an `output_length` of 0 is sent as 1, both counted (`Replay Rows Defaulted`), instead of rejecting the schedule. Recorded lengths are authoritative: a row sends exactly `input_length` prompt tokens as the server counts them (the chat template is subtracted from the content; `--no-replay-template-subtraction` counts content only), of which `cached_length` are reusable, with `output_length` forced. Per-row `temperature`, `top_p` and `reasoning_effort` override the payload.
  - The reusable text is the first `cached_length` tokens of a source: the class pool prefix (a shorter `cached_length` reuses a byte prefix of the same prefix) for rows outside a session and a session's first row; for later session rows, the most recent earlier prompt of the session that covers `cached_length` (the server holds prompts, not our synthetic replies), else the previous prompt plus a synthetic reply of its `output_length`, else the class pool prefix. Rows are taken in time order, so branching and concurrent rows within a session work; nothing waits for responses. When no source is long enough, as much as there is is reused, and the shortfall is counted (`Replay Cached Shortfall Rows/Tokens`). A row whose `input_length` is below the template + nonce + instruction is sent at that minimum and counted (`Replay Input Overflow Rows/Tokens`). A fresh tail (unique nonce + filler, one of 8 filler variants per length chosen by row; a bare newline + filler when the uncached room is too small for the nonce) fills the prompt to `input_length`. Layout runs before traffic; `Replay Layout S` and `Replay Layout Ms Per Row` report its cost (about 4 to 6 ms per row with long session prompts). Segment boundaries are measured where text merges across them, so lengths stay exact.
  - Note: the reply part of a session's source is synthetic, so a server that caches its own generated tokens can't reuse it; compare `Class <name> Cached Tokens` (server) with `Class <name> Recorded Cached Tokens` and `Laid Out Cached Tokens`, and `Delivered Over Recorded Cached`.
  - Late rows (more than `--qps-max-lateness` behind) are counted and still sent (`--replay-late-rows send`, default) or counted and skipped (`skip`); the summary reports `Replay Late Rows`, `Replay Late Share`, `Replay Rows Skipped`, `Replay Max Lateness Ms` and `Replay Cancelled Rows` (cancelled rows are sent with their recorded output length, the work the server did before the cancellation).
  - `--replay-retime poisson|gamma:CV2` keeps every row (lengths, class, session, sampling fields) and redraws only the send times of conversation starts (rows outside a session and each session's first row): seeded (`--request-seed` / the mix's `request_seed`) Poisson or gamma gaps at their own mean rate over their own span, applied to the replayed stream (a recorded pod keeps its own rate). Every follow-up keeps its recorded gap to the previous turn of its session, so none overtakes its parent; lead-in rows keep their times. The summary reports `Replay Retime` and, as for `--qps`, intended and delivered gap CV² and 10 s dispersion, so a replay can be compared with the same replay retimed.
  - Recorded pods: when rows carry `"pod"`, one test replica replays one recorded pod. `--replay-pod ID` replays that pod's rows at their exact times (offsets from the start of the whole schedule); `--replay-pods N --replay-pod-index I` (default `$REPLAY_POD_INDEX`) gives replica I the I-th of the first N recorded pods (natural order, or of a `--replay-pod` list). A schedule recording several pods must choose. Merging pods into one stream (`--replay-pod a,b` or `all`) is only for tiers above 1x and is labelled (`MERGED` in `Concurrency`, `Replay Pods Merged`). Session keys are scoped to their pod. Schedules without `"pod"` are split across `--replay-pods` generators by session.
  - `--replay-start-at UNIX_SECONDS` (default `$REPLAY_START_AT`): a wall-clock start shared by every generator, so replicas started at different moments replay in step. A generator ready early waits with its stats, QPS and `-t` starting at the shared start; one that starts after it logs a warning with the number of rows already due, which count as late (and go out at once with `--replay-late-rows send`).
  - Lead-in rows (`"lead_in": true`, task 15's warm-up for keys whose first row already has cached tokens) are laid out with the rest, so the first scored row reuses them, and sent unscored before time 0 (in setup, before users spawn): each session in order, `max_tokens` 1, skipping any an earlier prompt already covers. Time 0 is the first scored row; lead-ins aren't paced, scored or retimed. The summary reports `Replay Lead In Rows`, `Sent`, `Covered`, `Failed` and `S`. Bare-tail rows are confirmed by encoding the whole prompt; any still off are counted in `Replay Length Mismatch Rows`.
  - Session affinity keys use a new random salt per run (`Replay Session Salt`; `--replay-session-salt` fixes it).
- `--prefix-fill [--prefix-fill-concurrency N]`: prefill-only cache fill for a test's warm-up phase. Sends every prefix of every class once (classes interleaved), each as a request shaped like live traffic (the prefix plus a unique tail, so the cache ends up as live traffic leaves it) with `max_tokens` 1, keeping `N` (default 16) in flight, then exits. The summary reports `Prefix Fill Requests`, `Prompt Tokens Sent`, `Shared Prefix Tokens Sent`, `Server Prompt Tokens`, `Server Cached Tokens`, `Seconds` and `Prompt Tokens Per S`. Run it in one process and without `-t`.
- Every run's summary reports `Client Cpu Percent` (this process's CPU over the run window) and `Client Cpu Peak 5s Percent` (its busiest 5 s), with a warning above 90%: a saturated client sends late and evenly, which voids send-timing results. Text datasets (including `-pcml` prompts) are built before users spawn, and their prompt token count comes from the dataset once per process, so no tokenizer work lands in the run window. One process sustains about 100 req/s with 16k-token `-pcml` prompts and 300-token outputs (69% CPU, delivered gap CV² 1.78 against 1.86 intended), and 50 req/s with 100k-token prompts (53%); streaming costs about 20 µs of CPU per output token, so longer outputs lower that.
- The summary adds, per class: `Class <name> Requests`, `Reusable Tokens` (the tokens laid out for the server to reuse: `cached_prompt_tokens` unless the prompt is too short), average `Time To First Token`, `Total Latency`, `Prompt Tokens`, `Cached Tokens` (as reported by the server) and `Completion Tokens`, plus P50/P90/P99 TTFT and latency.

```
locust -t 10min -u 200 -r 200 --qps 4 --qps-distribution gamma --qps-cv2 4 \
  --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct --summary-file results.csv \
  --shape-mix '{"seed": 0, "classes": [
    {"name": "chat", "weight": 0.7, "prompt_tokens": 6000, "cached_prompt_tokens": 5000, "generated_tokens": 200, "prefix_pool_size": 32},
    {"name": "rag", "weight": 0.2, "prompt_tokens": 12000, "cached_prompt_tokens": 2000, "generated_tokens": 400, "prefix_pool_size": 4},
    {"name": "cold", "weight": 0.1, "prompt_tokens": 2000, "generated_tokens": 100}]}'
```

Generation options:
- `--chat` / `--no-chat`: chat API or raw completions. The mode is required (it used to default to chat, which silently kept runs that omitted the flag in chat mode).
- `--stream`: stream the result back. Enabling this gives "time to first token" and "time per token" metrics
- `--request-timeout S` (default 60): seconds to wait for a connection and, while streaming, between reads; 0 waits forever. Timed-out requests are failures. The summary counts `Requests Timeout` (including read timeouts mid-stream), `Requests Http 299` (a proxy answering for a cancelled request), `Requests Cancel Rate` (`--cancel-rate`) and `Requests Other Error`, the `Client Cancelled Fraction` (timeouts and 299s over requests sent) and, with `--shape-mix`, timeouts and 299s per class. `--max-fail-ratio` divides failed requests by requests sent (not by the custom metric rows, which used to dilute it about 15-fold); when a run fails it, the error line carries the same counts.
- (optional) `--logprobs`: corresponds to `logprobs` API parameter. For some providers, it's needed for output token counting in streaming mode.

Embeddings and rerank options:
- `--embeddings`: use the `/v1/embeddings` API instead of completions
- `--rerank`: use the `/v1/rerank` API. The generated prompt text is split into documents (by paragraph), and `--rerank-query` is used as the query.
- `--rerank-query`: the search query string for rerank requests. Defaults to a generic query if not specified.
- `--rerank-top-n`: number of top results to return from the rerank endpoint.
- `--rerank-return-documents` / `--no-rerank-return-documents`: whether to include document text in the rerank response (default: true).

SystemOne options:
- `--systemone`: use the Fireworks `/v1/systemone` API (always non-streaming). `--model` is sent verbatim, e.g. `accounts/<account>/deployments/<id>`. Requests are synthetic: a JSON state built from the limericks dataset plus `q0..q{N-1}` choice questions with `option_a..` criteria.
- `--systemone-num-questions`: questions per request (default 1).
- `--systemone-num-options`: criteria per choice question (default 4).
- `--systemone-state-tokens`: approximate state size in tokens (default 768). Sized with `--tokenizer` if given, otherwise ~4 chars/token.
- `--systemone-question-tokens`: approximate instructions size per question in tokens (default 48).
- `--systemone-unique-state` / `--no-systemone-unique-state`: by default each state starts with a random nonce so every request misses the prefix cache; with `--no-systemone-unique-state` each worker reuses one fixed state (warm-cache path).

A request fails if the status isn't 200 or `answers` doesn't contain exactly the asked question ids. A response with `usage.output_tokens == 0` came from the packed single-pass path; the summary reports `Packed Fraction`, `Questions Per S` (QPS x questions), mean `Input Tokens` and total latency percentiles instead of generation metrics.

### Warm-up and A/B comparison

Comparing two deployments (A without a change, B with it) needs both warmed up the way production is, then sent the same measured traffic. With `--shape-mix` (and `--qps`, or `--replay-schedule`), llm_bench warms the server up before the run window, unscored (nothing from the warm-up is in the stats, `-t` or the send-timing figures):

- `--warmup-hit-rate R` warms up until the client-measured prompt-cache hit rate (cached prompt tokens / prompt tokens from response `usage`, over the responses completed in the last `--warmup-window-s`, default 30) is at least `R - --warmup-tolerance` (default 0.02). `--warmup-cap-s C` is required with a `--shape-mix` target: the warm-up stops after C seconds even if the target is not reached (`Warmup Reason: cap`, a warning, not a failure). A `--replay-schedule` target needs no cap, its warm-up rows bound it (`Warmup Reason: exhausted` if the target is never reached). `--warmup-cap-s` with `--warmup-requests` is refused, so B is never cut short of A's count.
- `--warmup-requests N` is a fixed warm-up of exactly N requests, mutually exclusive with a target (`0` is no warm-up: the same run as one with no warm-up flag, and a schedule's warm-up rows are dropped). **The handoff from A to B is the count, not the duration**: at the end of warm-up the run prints `--warmup-requests <N>`; give B that and A's `--request-seed`, and it sends A's warm-up requests and A's measured requests (repeats are pairs, each with its own seed, run A1, B1, A2, B2 ...: see `compare` below) (on a slower server, open-loop pacing can fall behind, and an equal duration would mean different traffic).
- `--no-warmup` ignores the workload file's `warmup` block. With no flag and no block, nothing changes: no warm-up.
- A mix file may carry `"warmup": {"hit_rate": 0.62, "source": "free text"}`, a schedule's meta `"warmup": {"hit_rate": ..., "hit_rate_by_pod": {"<pod>": ...}, "source": ..., "rows": <int>}` (`hit_rate_by_pod` is used when one pod is replayed; either hit rate may be absent when unknown). They set the default target (a mix file's block may also say `"sized_for": {"qps": ..., "warmup_cap_s": ...}`, the warm-up its prefix pool was sized for: a run whose `--qps` or `--warmup-cap-s` differs (beyond the 6-digit rounding the profile writes) warns at startup and the summary records `Warmup Cap S`, `Warmup Sized For Qps` and `Warmup Sized For Cap S`; either hit rate may be absent: a replayed pod's `hit_rate_by_pod` figure wins over `hit_rate`; a mix file's block needs `--warmup-cap-s` and `--qps`, and without either it is left alone with a warning, so a mix that gains a block never stops an existing command line (explicit warm-up flags still require them); `--prefix-fill` ignores the block).

With `--shape-mix --qps` the warm-up is the mix at the run's rate (same arrival process, its own seed stream), drawn from a stream of its own derived from the request seed: the first N requests are the same whatever N is, and the measured requests are the same whatever the warm-up was. Each warm-up request is a non-streaming request with the class's `generated_tokens`; with `--lora-adapters` the warm-up requests pick the adapters round robin, in the order they are issued (their own counter: the measured requests' round robin does not depend on the warm-up's length). `--prefix-fill` stays a separate step (and refuses the warm-up flags).

With `--replay-schedule`, the schedule may carry warm-up rows (`"warmup": true`, recorded just before the window, at negative times): they are replayed at their recorded times, unscored, in schedule order with the `lead_in` rows (sent one token each, each held until the next warm-up row goes out and sent just ahead of it; a lead-in finishes before its session's next row is sent, sessions are concurrent). A lead-in is sent only if what was actually sent before it does not cover it: an earlier sent prompt of its session at least as long, or the session's last sent prompt plus its reply (a lead-in of 7002 + 20 tokens after a 7002-token warm-up row is covered); one that is not covered continues the text the session was sent where there is one, rather than starting fresh. A hit-rate target stops the warm-up early, before a warm-up row; `--warmup-requests N` sends the first N warm-up rows in schedule order. So a run given N makes the same decisions as the run that reported N: the rows left are dropped, with the lead-ins ahead of a dropped row of their session; the lead-ins still ahead (a scored session's earlier context, which the dropped rows would have provided) go out unless covered, in schedule order and one after another (after the warm-up requests still in flight), since they share pool prefixes. A scored row whose session's sent text holds its recorded cached length (the bar a lead-in is covered by) continues from that text, not the class pool prefix. A scored row is one that is neither `lead_in` nor `warmup`, and time 0 is the first scored row. If every warm-up row was sent (a target met on the last row, exhaustion, `--warmup-requests` of all of them) the recorded gap to time 0 is waited out; otherwise the measured window starts at once. After the warm-up the client waits for its in-flight warm-up requests before the measured phase starts (so the server is idle at time 0), and the measured phase may start about a second after the recorded gap, because the measured rows are laid out after the warm-up. A schedule with warm-up rows and no flag or target replays all of them (`Warmup Mode: requests`). A schedule without warm-up rows is as before (lead-ins only). Do not combine with `--replay-start-at` unless the start leaves room for the warm-up (a hit-rate target's length is not known ahead). A warm-up with `--processes N` (N > 1), `--master` or `--worker` is refused: each process would warm up on its own, and no single `--warmup-requests N` reproduces it.

The summary reports `Warmup Mode` (target, requests or none), `Warmup Reason` (target, cap, requests or exhausted), `Warmup Requests`, `Warmup Prompt Tokens`, `Warmup Duration S`, `Warmup Hit Rate` (the stop decision's window), `Warmup Target Hit Rate`, `Warmup Failed Requests` and `Measured Hit Rate` (the whole measured phase), and the workload identity `compare` needs: `Workload Sha256` (the mix file, schedule and meta), `Workload Options` and `Workload Options Sha256` (every option that changes the traffic: the replayed pod and retiming, the arrival process, `-t`, the corpus, tokenizer, output mode, chat mode, length and sampling flags), `Request Seed`, `Target Qps` and `Llm Bench Version`. A warm-up's draws leave the measured counters alone, and `Warmup Hit Rate` is over the full `--warmup-window-s` ending at the last warm-up response.

`--fail-on-delivery` exits 3 when the measured phase did not deliver its traffic: delivered send rate below `--qps-min-delivered` (default 0.95) of the target, more than `--max-late-share` (default 0.05) of sends late or skipped, or a failed-request share above `--max-fail-ratio` (already exit 1). The rate is judged against the run's own intended schedule, not the nominal `--qps` (a Poisson run's realized rate scatters around it): the sends' span on the intended schedule over their span on the wire, times the share of due sends that went out; the late share is over the sends that were due when the run ended. A stalled run fails it too: the monotonic clock stops while a host sleeps, so a paused schedule counted no due rows, so the window's wall-clock time is checked against it (`Clock Stall S`; a stall over 5 s fails, and a replay that sent under `--qps-min-delivered` of the rows due by the wall clock fails), a schedule the pacer moved later by more than max(5 s, 1% of the window) in total (`Schedule Shift S`: a stalled client shifts the rest of the schedule and its intended times with it, so the rate alone reads 1) fails, and a rate above 1 / `--qps-min-delivered` times intended (a backlog released in a burst) is not delivery either. Default: on when the run warms up, off otherwise, so existing `--qps` and `--replay-schedule` runs keep their exit codes; `--no-fail-on-delivery` turns it off. Reported as `Delivered Qps`, `Intended Qps`, `Target Qps`, `Delivered Over Intended`, `Late Send Share`, `Delivery Check`, `Delivery Failures`.

`python -m llm_bench.compare --a A1.csv [A2.csv ...] --b B1.csv [B2.csv ...] [--min-noise 0.03] [--min-repeats 3] [--json out.json] [--markdown out.md]` compares paired runs from `--summary-file` CSVs (every row is a run; the rows of `--a` and of `--b` are paired in order, A_i with B_i, so both sides need as many). **Repeats are paired, each pair with its own seed**: run i uses its own `--request-seed s_i` on both sides, A_i warms up to the target and prints its N_i, B_i gets `--request-seed s_i --warmup-requests N_i`, and the runs go A1, B1, A2, B2, ... A repeat with the same seed would resend the first run's exact prompts, unique tails included, into the cache that run left, and read as a better hit rate and lower TTFT. It refuses (exit 2) when a pair's request seed or warm-up requests differ, when a seed repeats across pairs, or when the workload identity differs between any two runs (workload sha, the options that change the traffic and which of them differ, rate, llm_bench version; the seed is not among the options), ignores rows that are not measured runs (a `--prefix-fill` step) and says so, else reports per-metric results from the paired relative deltas (throughput, TTFT, per-token and total latency percentiles, measured hit rate, failures): the mean delta against a noise floor of max(`--min-noise`, the spread of the per-pair deltas), with every pair's delta listed. A verdict needs 3 pairs (`--min-repeats` counts pairs); with fewer, every metric reads `inconclusive (too few repeats)` (the deltas are still reported). A delta inside the floor reads `same`, or `inconclusive (noisy pairs)` when the floor is over twice `--min-noise`; a metric that is 0 on A and not on B reads `different` (the absolute difference is reported). Failures (`Fail Ratio`, `Requests Failed`) are one rule: judged exactly, pair by pair, with no noise floor (any increase on B, or on A, is reported as `worse` / `better`, mixed pairs as `mixed`; with fewer than `--min-repeats` pairs it is still reported, labelled `(fact; too few pairs for the noise judgement)`), and a pair whose baseline is 0 is not a reason to judge the whole metric by a raw difference (the metric reads `different: better|worse where A was 0 (B higher, pair N); other pairs: <their verdict>`, the other pairs keep their relative deltas). A delta exactly at the floor is not beyond it (float noise at `+0.03` against a `0.03` floor reads `same`). Identity (seeds, counts, shas, versions, options hashes) is compared as exact strings; only the target rate takes a tolerance. A run that failed its delivery check or `--max-fail-ratio` makes compare exit 3 and name the run (`--allow-flagged` compares anyway: the run is then flagged in the output and is still in the figures). A percentile's floor is at least one bin of Locust's response-time rounding (exact below 100 ms, 10 ms below 1,000, 100 ms below 10,000, 1,000 ms above) over the value, reported as `percentile bin N ms` in the floor's source: P90 TTFT 1,500 vs 1,400 ms is one bin, not a change. Exit 0 compared, 2 refused, 3 flagged runs, 1 error. For a trustworthy comparison run at least 3 pairs, alternating A and B, not alongside other load; P99 is the noisiest figure.

**Iterating on changes.** To try one change after another against one baseline: run the baseline A once (pairs A1, A2, A3 with their own seeds and warm-up targets, so the summaries record each pair's seed and N_i). For every later variant, `python -m llm_bench.compare --recipe A1.csv A2.csv A3.csv` prints, per pair in order, the flags the variant run must use (`--request-seed <s_i> --warmup-requests <N_i>`, or `--request-seed <s_i> --no-warmup` for a baseline run that had no warm-up) and the workload identity it must match (workload sha, options hash, rate, llm_bench version) as one line to check against; it refuses a baseline with a repeated or missing seed or count or with mixed workloads (`--json` for a script). Run B1..B3 with those flags (later C1..C3 the same), and compare any two variants: A with B, A with C, B with C. Rerun the baseline from time to time (A') and compare A with A' to see drift: a delta beyond the floor means the baseline is stale and should be refreshed. A llm_bench version change means a new baseline (compare refuses a mix). If the server was restarted before every run, say so with `--restarted-between-runs` and the same-endpoint warning below becomes a note.

**A replay's hit rates, and a cold server.** For a replay, `Measured Hit Rate` mostly reflects the recorded cached lengths (the server caches what the schedule designs): it moves between A and B only when a server evicts what production kept, so a cache difference usually shows in latency first, not in the hit rate. A cold server needs a longer warm-up: on a first GPU run all six cold runs needed N=76, while on a warm server the same schedule gave 76, 19, 19. That is why the runs of an A/B restart the server first (and why a restart is stated with `--restarted-between-runs`). `Warmup Hit Rate` is the rate over the window the stop decision used (a target warm-up stops when it is within the tolerance of the target, with requests still in flight); `Warmup Hit Rate End` is the rate over the last window once those had completed, and can be lower when the requests in flight were cold.

**Same endpoint is not a valid A/A.** B_i sends A_i's exact requests (same seed, same warm-up), so on one server B_i hits the cache A_i filled: on a first GPU A/A, B's hit rate was 0.9993 against A's 0.972 and TTFT p50 was 14% lower in every pair. The summary records the target host (`Target Host`; not part of the workload identity, it is what A and B differ by) and `compare` warns in every output (stderr, the table, the markdown and JSON `warnings`) when A and B were sent to the same host: "same endpoint: B reuses A's cache unless the server was restarted between runs; not a valid A/A". The valid options are two servers (A and B each on their own), or restarting the server before every run (the same applies to an A/B of a flag on one deployment: restart between runs). The warning is about the host recorded, so a restart is yours to ensure.

**Fail fast.** The options that need nothing built are checked at startup, before prefixes are built (a warm-up without `--provider` or a `--host` that names it, `--replay-schedule` without `--shape-mix`, `--prefix-fill` with `--qps`, burst with `--qps`, an arrival process the pacer refuses, and the like), so a bad command line fails at once.

**`--prefix-reuse-probe`** prints `Prefix reuse probe: pass these flags to the scored runs: --prefix-reuse-overhang W [--prefix-restore-boundary B]` (also the summary's `Prefix Reuse Probe Suggested`). It sends no scored request, which used to read as a failed run: it now exits 0 when it measured something and 1 when it measured nothing or a measurement failed.

### Writing results

Locust prints out the detailed summary including quantiles of various metrics. Additionally, the script prints out the summary block at the very end of the output that includes the model being tested.

When comparing multiple configurations, it's useful to aggregate results together:

- `--summary-file`: Append the line with the summary to the specified CSV file. Useful for generating a spreadsheet with perf sweep results. If the file doesn't exist, it writes out the header first. For Fireworks text runs, the summary also includes server-side time to first token and total latency (average and P50 to P99.9). Client-side values minus server-side values is roughly the network and gateway time between the load generator and the deployment.
- `-t`: duration (e.g. `5min`) for which to run the test (standard Locust option). It's particularly useful when scripting multiple runs. By default, the test runs without a limit until Ctrl+C is pressed.

The typical workflow would be to run benchmark several times appending to the same CSV file. The resulting file can be imported into a spreadsheet or pandas for further analysis.

### Custom prompts

Sometimes it's necessary to replay exact prompts, for example in the case of embedding images.
`--dataset` option can be used in this case to specify a file with .jsonl extension (starting with an ampersand, e.g. `@prompt.jsonl`.).
jsonl files will be read line-by-line. Each line has to have a valid JSON object, which will be used to form the resulting API request.
Examples:

Chat dataset (--chat option):
```
{"messages": [{"role": "user", "content": "Write a poem about a cat"}], "temperature": 0.9}
{"messages": [{"role": "user", "content": "Write a poem about a dog"}], "temperature": 1}
```

Non-chat dataset (--no-chat option):
```
{"prompt": "One two three four"}
{"prompt": "Five six seven eight"}
```


## Examples

Download tokenizer for the model being benchmarked from Huggingface.
```
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /models/Meta-Llama-3-8B-Instruct   --include '*.json'
export TOKENIZER=/models/Meta-Llama-3-8B-Instruct
```

Maintain fixed 8 requests concurrency against local deployment:

```bash
locust -u 8 -r 2 -p 512 -o 128
```

Call streaming chat API locally with the request issued every 2 seconds. Run for 1 minute and save results to `results.csv`:

```bash
locust -t 1min -u 100 -r 100 -p 512 -o 128 --stream --chat --qps 0.5 --summary-file results.csv
```

Benchmark Fireworks public deployment deployment with 1 request only:

```bash
locust -u 1 -H https://api.fireworks.ai/inference -p 128 -o 200 --api-key $FIREWORKS_API_KEY --model=accounts/fireworks/models/llama-v3p1-8b-instruct
```

Benchmark Fireworks public deployment with 1 request and 2 images (1024w x 1024h and 3084w x 1080h):

```bash
locust -u 1  -H https://api.fireworks.ai/inference -p 128 -o 200 --api-key $FIREWORKS_API_KEY --model=accounts/fireworks/models/llama-v3p1-8b-instruct --chat --prompt-images-with-resolutions 1024x1024 3084x1080
```

Benchmark Fireworks rerank deployment with a single request and 115k prompt tokens:

```bash
locust -u 1 -r 2 -H https://api.fireworks.ai/inference --api-key $FIREWORKS_API_KEY -m "accounts/fireworks/models/qwen3-reranker-8b" --rerank --prompt-tokens 115000 -t 3min --tokenizer /path/to/tokenizer --summary-file rerank_results.csv
```

Benchmark Fireworks rerank with a custom query and top-5 results:

```bash
locust -u 1 -r 2 -H https://api.fireworks.ai/inference --api-key $FIREWORKS_API_KEY -m "accounts/fireworks/models/qwen3-reranker-8b" --rerank --rerank-query "How do I reset my password?" --rerank-top-n 5 --prompt-tokens 4096 -t 1min --tokenizer /path/to/tokenizer
```

Benchmark Fireworks SystemOne with 8 concurrent workers, 4 questions of 6 options each over a ~2k-token state:

```bash
locust -u 8 -r 8 -t 2min -H https://api.fireworks.ai/inference --api-key $FIREWORKS_API_KEY -m "accounts/pyroworks/deployments/pkrpl2xp" --systemone --systemone-num-questions 4 --systemone-num-options 6 --systemone-state-tokens 2048 --tokenizer /path/to/tokenizer --summary-file systemone_results.csv
```

Benchmark OpenAI deployment reading prompts from a file at 1 QPS:

```bash
locust --dataset '@input.jsonl' -u 1 -H https://api.openai.com -o 200 --api-key $OPENAI_API_KEY --model=gpt-3.5-turbo --chat
```

## UI mode

Instead of relying on textual data, it's also possible to plot the results in Grafana.

```bash
pip install locust locust-plugins
locust-compose up
```

This starts your local Postgre and Grafana. Grafana is available at http://127.0.0.1:3000 (sometimes logs don't print out).

Then run the test as specified above with an additional argument:

```bash
locust --config locust-grafana.conf ...
```

This starts the load test locally and pushes results into Grafana in real-time. Besides the actual requests, we push additional metrics (e.g. time per token) as separate fake requests to get stats aggregation. Make sure to remove them from aggregation when viewing the graphs.

Other settings for Locust are in `./locust.conf`. You may start Locust in non-headless mode, but its UI is very basic and misses advanced stats aggregation capabilities.
