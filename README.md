# Lambda.ai API Poller

Grab a 1x Blackwell GPU on Lambda Cloud the moment one frees up.

Lambda's single-GPU Blackwell capacity is effectively never idle. When someone
terminates a `gpu_1x_b200_sxm6`, it shows as available for a few seconds and then
someone else takes it. `lambda_watch.py` polls Lambda's API and fires the launch
request the instant capacity appears, so you win those races while asleep.

Stdlib only — Python 3.10+, no virtualenv, no dependencies, no install step.

## Setup

The API key comes from the first of these that exists:

1. `--api-key <key>`
2. `--api-key-file <path>` (a file that may contain labels; the `secret_...` line is used)
3. `$LAMBDA_API_KEY`
4. `~/.lambda/api_key`

```bash
mkdir -p ~/.lambda && chmod 700 ~/.lambda
cp /path/to/key.txt ~/.lambda/api_key && chmod 600 ~/.lambda/api_key
```

You also need an SSH key **already registered with Lambda** — pass its Lambda
name, not a file path. `--list` shows which keys the account has.

## Use

```bash
# what does the account look like, and what has capacity right now?
./lambda_watch.py --list

# watch without launching, to confirm the filters target what you expect
./lambda_watch.py --ssh-key my-key --dry-run

# the real thing: hunt indefinitely, speak the alert when it lands
./lambda_watch.py --ssh-key my-key --name my-b200 --say

# leave it running after you close the terminal
nohup ./lambda_watch.py --ssh-key my-key --name my-b200 \
    > ~/lambda-poll.log 2>&1 &
tail -f ~/lambda-poll.log
```

By default it targets **every 1-GPU Blackwell instance type** — today just
`gpu_1x_b200_sxm6` ($6.99/hr). The match is on GPU description, so if Lambda adds
a 1x GB200 or RTX PRO 6000 SKU it is picked up with no code change. To pin one
exactly, use `--instance-type gpu_1x_b200_sxm6`.

### Options worth knowing

| Flag | Why |
|---|---|
| `--dry-run` | Report capacity, never launch. Always start here. |
| `--region us-west-1 --region us-east-1` | Restrict to regions; repeat order = preference order. |
| `--max-price 8.00` | Refuse to launch above this $/hr. A cheap guard against a surprise SKU. |
| `--timeout 6h` | Give up eventually instead of running forever. |
| `--interval 15s` | Poll faster. Lambda allows ~1 req/s; the default 30s is polite. |
| `--allow-duplicate` | Launch even if you already own a matching instance. |
| `--webhook <url>` | POST `{"text": ...}` on success or fatal error (Slack-compatible). |
| `--user-data setup.sh` | cloud-init script to run on first boot. |
| `--image-family lambda-stack-24-04` | Pick the image. |
| `--gpu-count 8` / `--gpu-pattern H100` | Hunt something other than a single Blackwell. |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Launched (or, with `--dry-run`/duplicate found, nothing to do) |
| 1 | `--once` and no capacity |
| 2 | Fatal: bad key, unknown SSH key name, quota exceeded, bad filters |
| 3 | `--timeout` elapsed with no capacity |
| 4 | A launch may have gone through but couldn't be confirmed; stopped so it can't launch twice. Check the dashboard. |
| 130 | Ctrl-C |

## Behavior that matters

- **It will spend money.** A successful launch bills at the instance's hourly
  rate until *you* terminate it. This tool never terminates anything. Losing
  track of a won B200 costs ~$168/day.
- **Losing the race is normal.** When two watchers see the same freed GPU, the
  loser gets `insufficient-capacity`; that is logged and polling resumes.
- **A launch that looks like a failure is checked.** A timeout, a 5xx, or a
  garbled response can hide a launch that actually went through. Unless the
  launch response returns an instance id, the watcher compares `/instances`
  against the IDs it saw at startup, waiting up to 3 minutes for the instance to
  show up (one check for a plain `insufficient-capacity`). If a new instance
  shows up, that counts as success. If `/instances` can't be read at all, the
  watcher stops with exit 4 rather than risk a second launch.
- **It won't double-launch.** On startup, if you already own an instance of a
  target type (`active`/`booting`/`unhealthy`), it reports it and exits rather
  than adding a second one. Override with `--allow-duplicate`.
- **Preflight fails fast.** The key and SSH key names are validated before the
  loop starts, so a typo surfaces immediately rather than at the one moment
  capacity appears.
- **Rate limits are respected.** ≥1.1s between any two requests, ≥13s between
  launch attempts (Lambda documents 1/s and 1 per 12s respectively).
- **Cloudflare needs a User-Agent.** `cloud.lambda.ai` returns 403 (error 1010)
  to urllib's default agent, so the client sends its own.

## API notes

Built against `https://cloud.lambda.ai/api/v1`, whose OpenAPI spec is public at
`https://cloud.lambda.ai/api/v1/openapi.json` — useful if you extend this. Auth
is HTTP Basic with the API key as the username and an empty password. The two
endpoints that matter:

- `GET /instance-types` — every type, with `regions_with_capacity_available`
- `POST /instance-operations/launch` — `region_name`, `instance_type_name` and
  `ssh_key_names` are required; returns `data.instance_ids`

The codes in `FATAL_CODES` / `RETRYABLE_CODES` were taken from that spec rather
than guessed, which is what makes "retry on a lost race, stop on a quota problem"
reliable.

## Tests

```bash
python3 -m pytest tests/ -q
```

94 tests, no network: the selection/classification logic, the launch payload, and
the poll loop driven against a stubbed client (capacity appearing, races lost,
quota exceeded, duplicate guard, timeout, launches that land despite an
error response).
