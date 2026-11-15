import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import pair_check as pc

KIT = Path(__file__).resolve().parent.parent
SAMPLE = KIT / "sample"
STUB = SAMPLE / "prompts" / "stub_v2.1.txt"
OK = SAMPLE / "train_ok.jsonl"
MIXED = SAMPLE / "train_mixed.jsonl"


def row(system, user="question", assistant="answer"):
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": system,
        "messages": [
            {"role": "user", "content": [{"text": user}]},
            {"role": "assistant", "content": [{"text": assistant}]},
        ],
    }


def blocks(text):
    return [{"text": text}]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def check(capsys, *argv):
    code = pc.main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def all_six(stub=STUB, train=OK):
    return ("--train", train, "--stub", stub, "--config", SAMPLE / "service.yaml", "--var", "STUB_FILE",
            "--expect-newest")


# The samples


def test_the_matching_sample_passes_every_check(capsys):
    code, out = check(capsys, *all_six())
    assert code == 0, out
    assert out.count("PASS") == 6 and "FAIL" not in out
    assert "6 checks: 6 pass, 0 fail" in out


def test_the_mixed_sample_fails_and_names_the_stray_stub(capsys):
    code, out = check(capsys, "--train", MIXED, "--stub", STUB)
    assert code == 1
    assert "2 distinct system prompts across 3 rows" in out
    assert "matches stub_v2.txt  (first at line 2)" in out
    assert "2 of 3 rows equal stub_v2.1.txt after strip()" in out


def test_every_check_prints_what_it_proves(capsys):
    _, out = check(capsys, *all_six())
    assert out.count("      proves: ") == 6
    _, out = check(capsys, "--train", MIXED, "--stub", STUB)
    assert out.count("      not proven: ") == len(re.findall(r"^FAIL", out, re.M)) == 2


# The trailing-newline trap


def test_trailing_newlines_on_the_stub_do_not_fake_a_mismatch(tmp_path, capsys):
    stub = tmp_path / "stub_v1.txt"
    stub.write_text("Answer briefly.\n\n\n", encoding="utf-8")
    train = write_jsonl(tmp_path / "train.jsonl", [row(blocks("Answer briefly."))] * 3)
    code, out = check(capsys, "--train", train, "--stub", stub)
    assert code == 0, out
    assert "15 chars (raw file 18)" in out


def test_comparing_the_raw_file_reports_a_mismatch_that_is_not_real(capsys):
    # Same rows as the passing sample. The only difference is the newline every text file ends with.
    assert STUB.read_bytes().endswith(b"\n")
    code, out = check(capsys, "--train", OK, "--stub", STUB, "--no-strip")
    assert code == 1
    assert "0 of 3 rows equal stub_v2.1.txt as raw text, no strip" in out
    assert "differ from the stub only in leading or trailing whitespace" in out


def test_a_dataset_that_baked_in_the_raw_file_fails(tmp_path, capsys):
    stub = tmp_path / "stub_v1.txt"
    stub.write_text("Answer briefly.\n", encoding="utf-8")
    train = write_jsonl(tmp_path / "train.jsonl", [row(blocks("Answer briefly.\n"))] * 3)
    code, out = check(capsys, "--train", train, "--stub", stub)
    assert code == 1
    assert "the rows carry the raw file, trailing whitespace included" in out


# The string-sort trap


def test_text_sort_picks_the_wrong_newest_and_numeric_sort_does_not():
    names = ["stub_v1.txt", "stub_v2.txt", "stub_v2.1.txt"]
    assert max(names) == "stub_v2.txt"  # the trap: "t" sorts after "1"
    assert pc.newest(names) == "stub_v2.1.txt"


def test_version_parts_compare_as_numbers():
    names = ["stub_v2.9.txt", "stub_v2.10.txt"]
    assert max(names) == "stub_v2.9.txt"  # the same trap one digit further on
    assert pc.newest(names) == "stub_v2.10.txt"


def test_expect_newest_fails_when_a_newer_stub_sits_unused(capsys):
    code, out = check(capsys, "--train", OK, "--stub", SAMPLE / "prompts" / "stub_v2.txt", "--expect-newest")
    assert code == 1
    assert "a newer stub exists: stub_v2.1.txt (versions compared as numbers: v1, v2, v2.1)" in out


def test_files_without_a_version_are_counted_not_dropped(tmp_path, capsys):
    (tmp_path / "stub_v1.txt").write_text("First.\n", encoding="utf-8")
    (tmp_path / "stub_v2.txt").write_text("Second.\n", encoding="utf-8")
    (tmp_path / "stub_notes.txt").write_text("not a stub\n", encoding="utf-8")
    train = write_jsonl(tmp_path / "train.jsonl", [row(blocks("Second."))])
    code, out = check(capsys, "--train", train, "--stub", tmp_path / "stub_v2.txt", "--expect-newest")
    assert code == 0, out
    assert "ignored 1 other file starting with 'stub': stub_notes.txt" in out


def test_expect_newest_needs_a_version_in_the_name(tmp_path, capsys):
    stub = tmp_path / "system.txt"
    stub.write_text("Answer briefly.\n", encoding="utf-8")
    train = write_jsonl(tmp_path / "train.jsonl", [row(blocks("Answer briefly."))])
    code, out = check(capsys, "--train", train, "--stub", stub, "--expect-newest")
    assert code == 2
    assert "needs a version in the stub's name" in out


# Nothing is skipped, guessed or substituted


def test_load_stub_raises_instead_of_falling_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        pc.load_stub(tmp_path / "stub_v9.txt")


def test_a_missing_stub_is_exit_2_with_no_fallback(tmp_path, capsys):
    code, out = check(capsys, "--train", OK, "--stub", tmp_path / "stub_v9.txt")
    assert code == 2
    assert "There is no fallback to another stub." in out


def test_a_row_without_a_system_prompt_is_counted(tmp_path, capsys):
    bare = row(blocks("Answer briefly."))
    del bare["system"]
    stub = tmp_path / "stub_v1.txt"
    stub.write_text("Answer briefly.\n", encoding="utf-8")
    train = write_jsonl(tmp_path / "train.jsonl", [row(blocks("Answer briefly.")), bare, row(blocks("Answer briefly."))])
    code, out = check(capsys, "--train", train, "--stub", stub)
    assert code == 1
    assert "2 of 3 rows carry a system prompt" in out
    assert "no system prompt on line 2" in out


def test_an_unreadable_line_fails_with_its_line_number(tmp_path, capsys):
    good = json.dumps(row(blocks("Answer briefly.")))
    train = tmp_path / "train.jsonl"
    train.write_text(good + "\n{not json\n" + good + "\n", encoding="utf-8")
    stub = tmp_path / "stub_v1.txt"
    stub.write_text("Answer briefly.\n", encoding="utf-8")
    code, out = check(capsys, "--train", train, "--stub", stub)
    assert code == 1
    assert "2 rows from 3 lines (0 blank, 1 unreadable)" in out
    assert "unreadable line 2:" in out


def test_blank_lines_are_counted_and_shown(tmp_path, capsys):
    good = json.dumps(row(blocks("Answer briefly.")))
    train = tmp_path / "train.jsonl"
    train.write_text(good + "\n\n" + good + "\n", encoding="utf-8")
    stub = tmp_path / "stub_v1.txt"
    stub.write_text("Answer briefly.\n", encoding="utf-8")
    code, out = check(capsys, "--train", train, "--stub", stub)
    assert code == 0, out
    assert "2 rows from 3 lines (1 blank, 0 unreadable)" in out
    assert "blank lines (counted, not rows): 2" in out


def test_an_empty_training_file_fails(tmp_path, capsys):
    train = tmp_path / "train.jsonl"
    train.write_text("", encoding="utf-8")
    code, out = check(capsys, "--train", train, "--stub", STUB)
    assert code == 1
    assert "0 rows from 0 lines" in out


# Reading the stub's name from the service config


def test_config_reads_the_named_variable_not_the_first_stub_in_the_file():
    text = (SAMPLE / "service.yaml").read_text(encoding="utf-8")
    naive = re.search(r"value:\s*(\S+\.txt)", text).group(1)
    assert naive == "stub_v2.txt"  # the trap: the first stub in the file belongs to another variable
    assert pc.config_values(text, "STUB_FILE") == ["stub_v2.1.txt"]


def test_a_commented_out_pair_is_never_read():
    text = "env:\n  # - name: STUB_FILE\n  #   value: old.txt\n  - name: STUB_FILE\n    value: new.txt\n"
    assert pc.config_values(text, "STUB_FILE") == ["new.txt"]


def test_quoted_names_and_values_are_read():
    text = 'env:\n  - name: "STUB_FILE"\n    value: "stub_v3.txt"\n'
    assert pc.config_values(text, "STUB_FILE") == ["stub_v3.txt"]


def test_a_variable_set_twice_fails_instead_of_picking_one(tmp_path, capsys):
    config = tmp_path / "service.yaml"
    config.write_text("env:\n  - name: STUB_FILE\n    value: stub_v2.1.txt\n"
                      "  - name: STUB_FILE\n    value: stub_v2.txt\n", encoding="utf-8")
    code, out = check(capsys, "--train", OK, "--stub", STUB, "--config", config, "--var", "STUB_FILE")
    assert code == 1
    assert "STUB_FILE is set 2 times in service.yaml (stub_v2.1.txt, stub_v2.txt); the check will not pick one" in out


def test_a_variable_missing_from_the_config_fails(capsys):
    code, out = check(capsys, "--train", OK, "--stub", STUB, "--config", SAMPLE / "service.yaml", "--var", "PROMPT_FILE")
    assert code == 1
    assert "PROMPT_FILE is not set in service.yaml" in out


def test_a_config_naming_another_stub_fails(capsys):
    code, out = check(capsys, "--train", OK, "--stub", STUB, "--config", SAMPLE / "service.yaml",
                      "--var", "BASELINE_STUB_FILE")
    assert code == 1
    assert "service.yaml sets BASELINE_STUB_FILE = stub_v2.txt, but the stub checked is stub_v2.1.txt" in out


def test_config_without_var_is_refused():
    with pytest.raises(SystemExit) as refused:
        pc.main(["--train", str(OK), "--stub", str(STUB), "--config", str(SAMPLE / "service.yaml")])
    assert refused.value.code == 2


# Formats


def test_two_system_blocks_are_not_joined_into_one(tmp_path, capsys):
    stub = tmp_path / "stub_v1.txt"
    stub.write_text("Answer briefly. Be kind.\n", encoding="utf-8")
    split = row([{"text": "Answer briefly."}, {"text": " Be kind."}])
    train = write_jsonl(tmp_path / "train.jsonl", [split, split])
    code, out = check(capsys, "--train", train, "--stub", stub)
    assert code == 1
    assert "0 of 2 rows equal stub_v1.txt" in out
    assert "2 blocks, so no single stub file" in out


def test_a_plain_string_system_prompt_is_accepted(tmp_path, capsys):
    stub = tmp_path / "stub_v1.txt"
    stub.write_text("Answer briefly.\n", encoding="utf-8")
    train = write_jsonl(tmp_path / "train.jsonl", [row("Answer briefly.")] * 2)
    code, out = check(capsys, "--train", train, "--stub", stub)
    assert code == 0, out


# The real command line


@pytest.mark.parametrize("train, expected", [("train_ok.jsonl", 0), ("train_mixed.jsonl", 1), ("missing.jsonl", 2)])
def test_the_command_line_exit_codes(train, expected):
    proc = subprocess.run(
        [sys.executable, str(KIT / "pair_check.py"), "--train", str(SAMPLE / train), "--stub", str(STUB)],
        capture_output=True, text=True,
    )
    assert proc.returncode == expected, proc.stdout + proc.stderr
