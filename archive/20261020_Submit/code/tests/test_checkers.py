"""Unit tests for the checker DSL.

The battery is 1,000 items whose entire judgement lives in a declared string.
A wrong predicate is therefore invisible without a test that pins the semantics
of every predicate in ``bench/CONTRACT.md`` §4 down with a true case and a false
case.  That is what this file is.

    python tests/test_checkers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.checkers import (Ctx, CheckerError, compile_checker,  # noqa: E402
                            known_predicates, lint_checker, predicate_names)

FAILED: list[str] = []


def check(spec: str, reply: str, want: bool, *, prompt: str = "", users=(),
          system=None, n_new: int = 0):
    try:
        fn = compile_checker(spec)
    except CheckerError as exc:
        FAILED.append(f"{spec!r}: does not compile: {exc}")
        return
    got = bool(fn(Ctx(reply=reply, prompt=prompt, users=users or (prompt,),
                      system=system, n_new=n_new)))
    if got != want:
        FAILED.append(f"{spec!r} on {reply[:48]!r}: got {got}, want {want}")


def expect_error(spec: str):
    try:
        compile_checker(spec)
    except CheckerError:
        return
    FAILED.append(f"{spec!r}: expected CheckerError, compiled fine")


def test_substring():
    check("any:北京", "中国的首都是北京。", True)
    check("any:北京", "中国的首都是上海。", False)
    check("any:北京,上海", "首都是上海", True)
    check("all:甲,乙", "甲乙", True)
    check("all:甲,乙", "甲", False)
    check("none:喵,汪", "你好呀", True)
    check("none:喵,汪", "你好喵", False)
    check("!any:喵,汪", "你好呀", True)
    check("!any:喵,汪", "你好喵", False)
    check("contains_all:甲,乙", "甲乙丙", True)
    check("!all:甲,乙", "甲", True)


def test_len_and_bounds():
    check("len:1..5", "abc", True)
    check("len:1..5", "abcdef", False)
    check("startswith:好的", "好的，没问题", True)
    check("startswith:好的", "没问题，好的", False)
    check("endswith:。", "结束了。", True)
    check("endswith:！,?", "真的?", True)
    check("len:40..", "x" * 50, True)                  # open-ended upper bound
    check("len:40..", "x" * 10, False)
    expect_error("len:0..100000")                      # unreachable upper bound
    expect_error("re:^")


def test_regex():
    check(r"re:(?<!\d)42(?!\d)", "答案是 42。", True)
    check(r"re:(?<!\d)42(?!\d)", "答案是 420。", False)
    check(r"not_re:[。，！？]", "没有标点", True)
    check("regex_count:你好,3", "你好你好你好", True)
    check("regex_count:你好,3", "你好你好", False)
    # `regex_count` is a single-value predicate: its comma is its own separator
    check("regex_count:abc,2", "abc abc", True)


def test_lists():
    check("numbered:3", "1. 甲\n2. 乙\n3. 丙", True)
    check("numbered:3", "1. 甲\n2. 乙", False)
    check("numbered:2", "（1）甲\n（2）乙", True)
    check("numbered:2", "一、甲\n二、乙", True)
    check("numbered:2", "i. one\nii. two", True)
    check("dash_list:2", "- 苹果\n- 香蕉", True)
    check("dash_list:2", "苹果 香蕉", False)
    check("bullets:3", "- 甲\n2. 乙\n三、丙", True)
    check("steps:步骤一,步骤二,步骤三", "步骤一 甲，步骤二 乙，步骤三 丙", True)
    check("steps:步骤一,步骤二", "步骤二 乙，步骤一 甲", False)


def test_json():
    check("json:name,age", '{"name": "小明", "age": 12}', True)
    check("json:name,age", '{"name": "小明"}', False)
    check("json:name", '好的，这是结果：\n```json\n{"name": "小明"}\n```', True)
    check("json_nested:user.name", '{"user": {"name": "小明"}}', True)
    check("json_nested:user.name", '{"user": "小明"}', False)
    check("json_array:3", '["a", "b", "c"]', True)
    check("json_array:3", '["a", "b"]', False)
    check("json_array_min:2", '["a", "b", "c"]', True)
    check("json_has:name=小明", '{"name": "小明"}', True)
    check("json_has:name=小明", '{"name": "小红"}', False)
    check("json:name", '{"name": "小明",}', False)      # trailing comma：应判失败


def test_tabular():
    check("csv:2,2", "name,age\n小明,12", True)
    check("csv:3,2", "name,age\n小明,12", False)
    check("md_table", "| 颜色 | 含义 |\n| --- | --- |\n| 红 | 热情 |", True)
    check("md_table", "红：热情", False)
    check("code_block", "```python\nprint(1)\n```", True)
    check("code_block", "print(1)", False)
    check("yaml:name,age", "name: 小明\nage: 12", True)
    check("yaml:name,age", "name: 小明", False)
    check("xml:name", "<name>小明</name>", True)
    check("xml:name", "<name>小明", False)
    check("kv:name,age", "name=小明\nage=12", True)
    check("kv:name,age", "小明 12", False)


def test_numbers():
    check("num:42", "答案是 42", True)
    check("num:42", "答案是 420", False)
    check("num:42", "答案是 42.5", False)
    check("num:42.5", "3.50 米", False)
    check("num:3.5", "3.50 米", True)
    check("num_any:42,43", "43 才对", True)
    check("num:42,43", "43 才对", True)
    check("number_only:123,10", "123", True)
    check("number_only:123,10", "答案是 123 哦", False)      # strict by default
    check("number_only:123,10,delims", "答案是 123。", True)
    check("number_only:123,10,delims", "答案是 123456 哦", False)
    check("number_only:123,10", "123。", True)
    check("is_int_in:1,10", "7", True)
    check("is_int_in:1,10", "7 个", False)
    check("num_f:3.5", "3.50 米", True)
    check("eq:cba", "cba", True)
    check("eq:cba", "cba ", True)
    check("eq:cba", "cba d", False)


def test_style_guards():
    check("ascii:0.8,5", "Hello there, friend!", True)
    check("ascii:0.8,5", "你好，朋友，很高兴见到你", False)
    check("ascii:0.8,5", "Hi", False)                    # below min length
    check("zh_ratio:0.5,3", "你好朋友", True)
    check("zh_ratio:0.5,3", "hello", False)
    check("dedup:0.65,4,40", "这是一段正常的话，内容各不相同，读起来没有重复。", True)
    check("dedup:0.65,4,40", "哈哈哈哈" * 30, False)
    # MINLEN gates *judging*, not passing: a short reply abstains, so the length
    # requirement must come from len:.  This is deliberate — see CONTRACT §4.
    check("dedup:0.65,4,40", "喵", True)
    check("dedup:0.65,4,40;len:40..", "喵", False)
    check("punct_free", "你好世界", True)
    check("punct_free", "你好，世界", False)
    check("caps:hello", "HELLO WORLD", True)
    check("sentences:2", "第一句。第二句！", True)
    check("sentences:3", "只有一句。", False)
    # ASCII terminators must count, or every English probe scores 0
    check("sentences:2", "This is one. And this is two.", True)
    check("sentences:3", "One. Two.", False)
    check("sentences:2", "First clause; second clause.", True)


def test_think_guard():
    check("not_think", "正常回答", True)
    check("not_think", "<think>在想……", False)
    check("not_think", "<think>想完了</think>答案是 42", True)
    check("think_leak", "<think>在想……", True)
    check("think_leak", "正常回答", False)


def test_not_echo():
    check("not_echo_user", "这是全新的回答内容", True)
    check("not_echo_user", "请介绍一下你自己吧", False,
          prompt="请介绍一下你自己吧")


def test_and_composition():
    spec = "any:喵;len:1..60;dedup:0.65,4,40;not_think"
    check(spec, "主人你回来啦喵～今天过得怎么样呀喵？", True)
    check(spec, "主人你回来啦", False)
    check(spec, "喵" * 80, False)
    check(spec, "<think>喵喵喵喵喵喵喵喵喵喵喵喵喵喵喵喵喵喵喵喵", False)


def test_escaping():
    """Only two escape sequences exist: \\x2c -> ',' and \\x3b -> ';'.

    Every other backslash is left alone, so regex escapes such as ``\\d``
    survive verbatim and ``re:\\d{2\\x2c3}`` matches 12, 123, ... (the ``\\x2c``
    decodes to a comma *after* the split decision, so the quantifier stays put).
    """
    check(r"re:\d\d\d?", "数字 123", True)
    check(r"re:\d\d\d?", "数字 1", False)
    check(r"re:\d{2\x2c3}", "数字 123", True)
    check(r"re:\d{2\x2c3}", "数字 12", True)
    check(r"re:\d{2\x2c3}", "数字 1", False)
    check(r"any:a\x2cb", "x a,b y", True)
    check(r"any:a\x2cb", "x a b y", False)
    check(r"eq:a\x2cb", "a,b", True)
    check("len:1..5;any:甲", "甲", True)
    expect_error("nosuchpred:x")
    expect_error("")
    expect_error("dedup:notanumber")
    expect_error("number_only:abc")
    check("true;any:甲", "甲", True)
    # a two-argument predicate must still split
    assert compile_checker("num:1,2")(Ctx(reply="2")) is True
    # a regex may contain a semicolon via \x3b, which must not end the spec
    assert compile_checker(r"re:a\x3bb;len:2..5")(Ctx(reply="a;b")) is True


def test_lint():
    assert lint_checker("any:北京") != [], "bare any: should be flagged"
    # anchor-does-not-discriminate analysis is not lint's job: it needs the turn
    # structure and the reviewed exception list, both of which live in registry
    assert lint_checker("any:北京;len:1..50") == [], \
        lint_checker("any:北京;len:1..50")
    assert lint_checker("any:喵;len:1..60;dedup:0.65,4,40") == [], \
        lint_checker("any:喵;len:1..60;dedup:0.65,4,40")
    assert lint_checker("not_empty") != []
    assert lint_checker("len:1..5") == []


def test_registry_consistency():
    """Every predicate named in the contract's DSL table must exist, and the
    contract must document everything that is implemented."""
    text = (ROOT / "bench" / "CONTRACT.md").read_text(encoding="utf-8")
    # only the "### 谓词表" table lists predicates; the schema table above it
    # lists item fields and must not be mistaken for predicates
    marker = "### 谓词表"
    assert marker in text, "CONTRACT §4 谓词表 heading changed"
    table = text.split(marker, 1)[1].split("###", 1)[0]
    documented = set()
    for line in table.splitlines():
        if line.startswith("| `"):
            name = line.split("`")[1].split(":")[0].split("[")[0].strip()
            if name and name.islower():
                documented.add(name)
    implemented = set(known_predicates())
    missing = sorted(documented - implemented - {"true"})
    extra = sorted(implemented - documented)
    if missing:
        FAILED.append(f"documented but not implemented: {missing}")
    if extra:
        FAILED.append(f"implemented but undocumented: {extra}")
    if len(documented) < 30:
        FAILED.append(f"only parsed {len(documented)} predicates from CONTRACT; "
                      f"the table format probably changed")


def test_predicate_names():
    assert set(predicate_names("any:甲;!len:1..5;re:x")) == {"any", "len", "re"}


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURES:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print(f"PASS: checker DSL ({len(tests)} groups, "
          f"{len(known_predicates())} predicates registered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
