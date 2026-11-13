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
