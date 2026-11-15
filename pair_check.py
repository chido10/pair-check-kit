"""Check that a fine-tuning dataset carries the exact system prompt your service sends.

A fine-tuned model is trained against one system prompt, and every row of the training file carries it. The
service has to send that same text on every call. When the two drift apart, nothing raises an error: Amazon
Bedrock validates the file's format when a job starts, but it has no way to know what your service sends.
Run this before you pay for a training run, and on every change after that.

    python pair_check.py --train sample/train_ok.jsonl --stub sample/prompts/stub_v2.1.txt
    python pair_check.py --train sample/train_ok.jsonl --stub sample/prompts/stub_v2.1.txt \\
        --config sample/service.yaml --var STUB_FILE --expect-newest

Exit codes: 0 every check passed, 1 a check failed, 2 the check could not run.
Standard library only.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# "stub_v2.1.txt" -> stem "stub", version "2.1". Versions are compared as numbers, never as text.
VERSIONED = re.compile(r"^(?P<stem>.+?)[_-]?v(?P<version>\d+(?:\.\d+)*)\.txt$")

PROVES = {
    "rows": "every line was read and counted; none was skipped silently",
    "system": "no row was built without its partner",
    "one prompt": "every row was built against the same stub",
    "matches stub": "the rows carry the exact text the service sends",
    "config": "the stub checked here is the one the service is set to load",
    "newest": "no newer stub sits in the folder unused by this build",
}


class CannotRun(Exception):
    """An input is missing or unreadable, so no check result would mean anything."""


def load_stub(path, strip=True):
    """Read the stub the way the loaders read it. A missing file raises: there is no fallback to another stub."""
    text = Path(path).read_text(encoding="utf-8")
    return text.strip() if strip else text


def version_key(name):
    """("stub", (2, 1)) for "stub_v2.1.txt", or None when the name carries no version."""
    m = VERSIONED.match(name)
    if not m:
        return None
    return m.group("stem"), tuple(int(part) for part in m.group("version").split("."))


def newest(names):
    """The highest version among the names, each part compared as a number.

    Sorting the names as text gets this wrong: "stub_v2.txt" sorts after "stub_v2.1.txt" because "t" comes after "1".
    """
    versioned = [(version_key(name)[1], name) for name in names if version_key(name)]
    return max(versioned)[1] if versioned else None


def stub_family(stub_path):
    """The .txt files in the stub's folder that share its stem: (stem, versioned names, other names)."""
    key = version_key(stub_path.name)
    stem = key[0] if key else stub_path.stem
    versioned, other = [], []
    for path in sorted(stub_path.parent.iterdir()):
        if not path.is_file() or path.suffix != ".txt" or not path.name.startswith(stem):
            continue
        k = version_key(path.name)
        (versioned if k and k[0] == stem else other).append(path.name)
    return stem, versioned, other


def config_values(text, var):
    """Every value set for var in a YAML env list: a "- name: VAR" line followed by a "value: ..." line.

    The match is anchored on the variable's name, so another variable or a commented-out line is never read by
    mistake. Reading the first matching value in the file instead is how a check ends up reporting a different
    model's setting.
    """
    pattern = re.compile(
        r"^[ \t]*(?:-[ \t]+)?name:[ \t]*(?P<q>[\"']?)" + re.escape(var) + r"(?P=q)[ \t]*(?:#.*)?\r?\n"
        r"[ \t]*value:[ \t]*(?P<value>\"[^\"\n]*\"|'[^'\n]*'|[^\s#]+)",
        re.M,
    )
    return [m.group("value").strip("\"'") for m in pattern.finditer(text)]


def system_of(row):
    """The row's system prompt as a tuple of texts, or None when the row has none.

    Bedrock's conversation format holds it as a list of {"text": ...} blocks, and a plain string is accepted
    too. Blocks are kept apart, never joined: two blocks are a different shape from the one block a service
    sends, and the check does not guess how they would be combined.
    """
    s = row.get("system")
    if isinstance(s, str):
        texts = (s,)
    elif isinstance(s, list) and s and all(isinstance(b, dict) and isinstance(b.get("text"), str) for b in s):
        texts = tuple(b["text"] for b in s)
    else:
        return None
    return texts if any(t.strip() for t in texts) else None


def read_rows(path):
    """Every line, counted. Returns (rows as (line number, row), lines, blank line numbers, unreadable lines)."""
    rows, blank, bad = [], [], []
    lines = 0
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            lines = number
            if not line.strip():
                blank.append(number)
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                bad.append((number, e.msg))
                continue
            if not isinstance(row, dict):
                bad.append((number, "not a JSON object"))
                continue
            rows.append((number, row))
    return rows, lines, blank, bad


def first(numbers, limit=5):
    shown = ", ".join(str(n) for n in numbers[:limit])
    return shown + (f" and {len(numbers) - limit} more" if len(numbers) > limit else "")


def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def identify(texts, stub_path, strip):
    """Name the stub file in the stub's folder whose text equals this prompt, so a mismatch says where it came from."""
    if len(texts) != 1:
        return f"{len(texts)} blocks, so no single stub file"
    _, versioned, other = stub_family(stub_path)
    for name in versioned + other:
        if load_stub(stub_path.parent / name, strip) == texts[0]:
            return f"matches {name}"
    return f"matches no stub file in {stub_path.parent.name or '.'}"


def run(args):
    """Run the checks. Returns a list of (name, ok, detail, extra lines)."""
    train, stub_path = Path(args.train), Path(args.stub)
    strip = not args.no_strip
    if not train.is_file():
        raise CannotRun(f"training file not found: {train}")
    if not stub_path.is_file():
        raise CannotRun(f"stub not found: {stub_path}. There is no fallback to another stub.")
    if args.config and not Path(args.config).is_file():
        raise CannotRun(f"config not found: {args.config}")
    if args.expect_newest and not version_key(stub_path.name):
        raise CannotRun(f"--expect-newest needs a version in the stub's name, like stub_v2.1.txt; got {stub_path.name}")
    try:
        stub = load_stub(stub_path, strip)
        raw = load_stub(stub_path, strip=False)
        rows, lines, blank, bad = read_rows(train)
    except UnicodeDecodeError as e:
        raise CannotRun(f"a file is not UTF-8 text: {e}")

    results = []

    extra = [f"unreadable line {n}: {why}" for n, why in bad[:5]]
    if len(bad) > 5:
        extra.append(f"and {len(bad) - 5} more unreadable lines")
    if blank:
        extra.append(f"blank lines (counted, not rows): {first(blank)}")
    results.append(("rows", bool(rows) and not bad,
                    f"{plural(len(rows), 'row')} from {plural(lines, 'line')} "
                    f"({len(blank)} blank, {len(bad)} unreadable)", extra))

    prompts = {}  # prompt texts -> [rows carrying it, first line number]
    missing = []
    for number, row in rows:
        texts = system_of(row)
        if texts is None:
            missing.append(number)
            continue
        prompts.setdefault(texts, [0, number])[0] += 1
    carried = len(rows) - len(missing)
    extra = [f"no system prompt on line{'s' if len(missing) > 1 else ''} {first(missing)}"] if missing else []
    results.append(("system", bool(rows) and not missing,
                    f"{carried} of {plural(len(rows), 'row')} carry a system prompt", extra))

    extra = []
    if len(prompts) > 1:
        for texts, (count, line) in sorted(prompts.items(), key=lambda item: -item[1][0]):
            joined = "\n".join(texts)
            digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
            extra.append(f"{plural(count, 'row'):<9} {len(joined)} chars  sha256 {digest}  "
                         f"{identify(texts, stub_path, strip)}  (first at line {line})")
    results.append(("one prompt", len(prompts) == 1,
                    f"{plural(len(prompts), 'distinct system prompt')} across {plural(carried, 'row')}", extra))

    equal = prompts.get((stub,), [0])[0]
    how = "after strip()" if strip else "as raw text, no strip"
    sizes = f"{len(stub)} chars (raw file {len(raw)})" if strip else f"{len(stub)} chars"
    extra = []
    if len(prompts) == 1 and equal == 0:
        (only,) = prompts
        if len(only) == 1 and strip and only[0] == raw:
            extra.append("the rows carry the raw file, trailing whitespace included, which a loader that strips never sends")
        elif len(only) == 1 and only[0].strip() == stub.strip():
            extra.append("the rows differ from the stub only in leading or trailing whitespace")
        else:
            extra.append(f"the rows' prompt {identify(only, stub_path, strip)}")
    results.append(("matches stub", bool(rows) and equal == len(rows),
                    f"{equal} of {plural(len(rows), 'row')} equal {stub_path.name} {how}: {sizes}", extra))

    if args.config:
        config = Path(args.config)
        try:
            values = config_values(config.read_text(encoding="utf-8"), args.var)
        except UnicodeDecodeError as e:
            raise CannotRun(f"config is not UTF-8 text: {e}")
        if not values:
            ok, detail = False, f"{args.var} is not set in {config.name}"
        elif len(values) > 1:
            ok, detail = False, (f"{args.var} is set {len(values)} times in {config.name} "
                                 f"({', '.join(values)}); the check will not pick one")
        else:
            ok = Path(values[0]).name == stub_path.name
            detail = f"{config.name} sets {args.var} = {values[0]}"
            if not ok:
                detail += f", but the stub checked is {stub_path.name}"
        results.append(("config", ok, detail, []))

    if args.expect_newest:
        stem, versioned, other = stub_family(stub_path)
        top = newest(versioned)
        order = sorted(versioned, key=lambda name: version_key(name)[1])
        labels = ", ".join("v" + ".".join(str(p) for p in version_key(name)[1]) for name in order)
        if top == stub_path.name:
            ok, detail = True, f"{stub_path.name} is the highest of {len(order)} versions ({labels}), compared as numbers"
        else:
            ok, detail = False, f"a newer stub exists: {top} (versions compared as numbers: {labels})"
        extra = []
        if other:
            extra.append(f"ignored {len(other)} other {'file' if len(other) == 1 else 'files'} "
                         f"starting with '{stem}': {', '.join(other)}")
        results.append(("newest", ok, detail, extra))

    return results


def report(results, train, stub):
    print(f"pair check: {train} against {stub}")
    for name, ok, detail, extra in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<13} {detail}")
        for line in extra:
            print(f"{'':20}{line}")
        print(f"      {'proves' if ok else 'not proven'}: {PROVES[name]}")
    passed = sum(1 for _, ok, _, _ in results if ok)
    print(f"{plural(len(results), 'check')}: {passed} pass, {len(results) - passed} fail")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="the training file, one JSON object per line")
    ap.add_argument("--stub", required=True, help="the system prompt file your service loads")
    ap.add_argument("--config", help="the service config that names the stub file")
    ap.add_argument("--var", help="the variable in --config that holds the stub's file name")
    ap.add_argument("--expect-newest", action="store_true",
                    help="fail when a newer stub version sits in the stub's folder (for new builds, not rollbacks)")
    ap.add_argument("--no-strip", action="store_true",
                    help="compare the raw file; use only if your loaders do not strip the stub")
    args = ap.parse_args(argv)
    if bool(args.config) != bool(args.var):
        ap.error("--config and --var go together")
    try:
        results = run(args)
    except CannotRun as e:
        print(f"pair check could not run: {e}", file=sys.stderr)
        return 2
    report(results, args.train, args.stub)
    return 0 if all(ok for _, ok, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
