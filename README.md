# pair-check-kit

One script that checks a fine-tuning dataset carries the exact system prompt your service sends. Run it before you pay for a training run, and on every change after that. One file, standard library only, built for the conversation format Amazon Bedrock uses for fine-tuning.

## Why it exists

A fine-tuned model is trained against one system prompt. Every row of the training file carries it, and the service has to send the same text on every call. Change it on one side and you are running a combination that was never trained.

Nothing raises an error when that happens. Amazon Bedrock validates the file's format when a job starts, but it has no way to know what your service sends. Four of the checks below come from mistakes made in a real pipeline: a dataset builder that baked the wrong stub into every row, a re-bake tool that wrote the raw file into the rows, a check that read the wrong line of a config, and a test that sorted versions as text. The other two make sure no row slips out of the count.

## What you need

- Python 3.12. The script itself uses only the standard library.
- `pip install -r requirements-dev.txt`, only to run the tests.

## Run it

```
python pair_check.py --train sample/train_ok.jsonl --stub sample/prompts/stub_v2.1.txt --config sample/service.yaml --var STUB_FILE --expect-newest
python pair_check.py --train sample/train_mixed.jsonl --stub sample/prompts/stub_v2.1.txt
```

`sample/` holds three versions of a small help-desk prompt and two three-row training files. `train_ok.jsonl` carries `stub_v2.1.txt` on every row. `train_mixed.jsonl` is the same file with row 2 built against `stub_v2.txt`.

## Result

The first command, exit code 0:

```
pair check: sample/train_ok.jsonl against sample/prompts/stub_v2.1.txt
PASS  rows          3 rows from 3 lines (0 blank, 0 unreadable)
      proves: every line was read and counted; none was skipped silently
PASS  system        3 of 3 rows carry a system prompt
      proves: no row was built without its partner
PASS  one prompt    1 distinct system prompt across 3 rows
      proves: every row was built against the same stub
PASS  matches stub  3 of 3 rows equal stub_v2.1.txt after strip(): 188 chars (raw file 189)
      proves: the rows carry the exact text the service sends
PASS  config        service.yaml sets STUB_FILE = stub_v2.1.txt
      proves: the stub checked here is the one the service is set to load
PASS  newest        stub_v2.1.txt is the highest of 3 versions (v1, v2, v2.1), compared as numbers
      proves: no newer stub sits in the folder unused by this build
6 checks: 6 pass, 0 fail
```

The second command, exit code 1. It names the file each stray prompt came from:

```
pair check: sample/train_mixed.jsonl against sample/prompts/stub_v2.1.txt
PASS  rows          3 rows from 3 lines (0 blank, 0 unreadable)
      proves: every line was read and counted; none was skipped silently
PASS  system        3 of 3 rows carry a system prompt
      proves: no row was built without its partner
FAIL  one prompt    2 distinct system prompts across 3 rows
                    2 rows    188 chars  sha256 c414e1db3bf9  matches stub_v2.1.txt  (first at line 1)
                    1 row     167 chars  sha256 9f99a95bac46  matches stub_v2.txt  (first at line 2)
      not proven: every row was built against the same stub
FAIL  matches stub  2 of 3 rows equal stub_v2.1.txt after strip(): 188 chars (raw file 189)
      not proven: the rows carry the exact text the service sends
4 checks: 2 pass, 2 fail
```

## The checks

| Check | Proves | Why it is there |
|---|---|---|
| rows | every line was read and counted; none was skipped silently | A line that fails to parse is a row the job never trains on. Blank lines are counted and shown, not failed. |
| system | no row was built without its partner | A row with no system prompt would otherwise drop out of every count below it. |
| one prompt | every row was built against the same stub | A dataset builder once walked a list of fallback file names, had never heard of the newer stubs, and baked an old one into every row. No error was raised. |
| matches stub | the rows carry the exact text the service sends | The comparison applies the same `strip()` the loaders apply. Compare the raw file instead and a healthy dataset fails on the trailing newline. Bake the raw file into the rows and the dataset carries a character no loader sends. |
| config (opt-in) | the stub checked here is the one the service is set to load | The value is read by the variable's name. A check that took the first matching line of a config once reported a different model's setting. |
| newest (opt-in) | no newer stub sits in the folder unused by this build | Versions are compared as numbers. Sorted as text, `stub_v6.txt` ranks above `stub_v6.4.txt`, because `t` comes after `4`. |

## Use it on your data

```
python pair_check.py --train <training file> --stub <the prompt file your service loads> [--config <service config> --var <variable>] [--expect-newest] [--no-strip]
```

- **Format.** Rows in the conversation format: `"schemaVersion": "bedrock-conversation-2024"`, a `system` list of text blocks, then `messages`. Bedrock's user guide lists this format for the models it fine-tunes through the Converse API, Meta Llama 3.3 70B among them. The check also accepts a `system` given as a plain string, the shape the same page shows for Anthropic Claude 3 Haiku. (Amazon Bedrock User Guide, [Prepare data for fine-tuning your models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html), read 2026-11-15.)
- **Not covered: the prompt and completion format**, which the same page lists for Meta Llama 3.1. It has no `system` field, so the prompt sits inside every `prompt` and there is no separate text to compare.
- **Strip.** The check strips the stub because the loaders it was written for strip it on both sides, training and serving. If yours do not, pass `--no-strip`.
- **Blocks.** A row with two `system` blocks counts as a different prompt from one block. The check does not guess how blocks would be joined.
- **Config.** `--config` reads a YAML env list: a `- name: VAR` line followed by a `value:` line. The match is on the variable's name, so other variables and commented-out lines are never read. A variable set twice fails instead of picking one. The file names are compared, not the folders.
- **Newest.** `--expect-newest` looks at the `.txt` files in the stub's folder that share its stem, such as `stub_v1.txt` and `stub_v2.1.txt`. Files with no version in the name are listed as ignored. Use it when you build data for a new model. Leave it off for a rollback, where the older stub is the right one.

Exit codes: 0 every check passed, 1 a check failed, 2 the check could not run, for example a missing file or a bad flag. A missing stub is always exit 2, with no fallback to another file.

## CI

`.github/workflows/pair-check.yml` runs the tests, then the gate: the matching sample must pass every check, and the mixed sample must exit with exactly 1. Exactly 1 matters. A typo in a path exits 2, and a step that accepted any failure would let it through as "failed as expected".

To gate your own repository, add a step with your paths:

```yaml
      - name: Training data and system prompt are a pair
        run: python pair_check.py --train data/train.jsonl --stub prompts/stub_v3.txt --config service.yaml --var STUB_FILE
```

GitHub looks for workflow files in the `.github/workflows` directory at the root of the repository, so copy the file there ([GitHub Docs, Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows), read 2026-11-15).

## What it does not do

- It does not validate the rest of the file. Bedrock does that when the job starts.
- It does not measure how much a mismatch changes the model's output. No number for that is claimed here.
- It makes no AWS calls and needs no account.

## Status

- 29 tests pass offline (`python -m pytest -q`, Python 3.12.10, pytest 9.0.2, 2026-11-15).
- Run read-only against a real 3,422-row training file and its 10,306-character stub: all six checks passed in 0.3 seconds.
- The CI workflow has not run on GitHub yet. It runs once this folder is its own repository. Its two gate steps were replayed locally under `bash -e`, together with a mistyped path and a swapped file, and each failed where it should.

## Layout

| Path | What it is |
|---|---|
| `pair_check.py` | the check, one file |
| `tests/test_pair_check.py` | the tests, including the trailing-newline and text-sort traps |
| `sample/` | three prompt versions, a matching and a mixed training file, a service config |
| `.github/workflows/pair-check.yml` | the CI gate |

## License

MIT. See `LICENSE`.
