"""
Unit tests for sieval/cli/_filter_spec.py.

Covers: key_function_spec, check_arg_names, check_by, check_values_source,
check_by_digest, resolve_values_path, relative_values_files,
compute_values_digest, compute_key_function_digest, and pin_filter_digests
(inject / verify / reject, for both digests).

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import functools
import hashlib
import importlib
import inspect
import linecache
import sys

import pytest

from sieval.cli._filter_spec import (
    BY_DIGEST_KEY,
    VALUES_DIGEST_KEY,
    check_arg_names,
    check_by,
    check_by_digest,
    check_values_source,
    compute_key_function_digest,
    compute_values_digest,
    key_function_spec,
    pin_filter_digests,
    relative_values_files,
    resolve_values_path,
)


def sample_key(row):
    """A key function the pin tests can name by dotted path."""
    return row["id"]


def other_key(row):
    """A second one, so a digest can be shown to distinguish them."""
    return row["id"].lower()


_SAMPLE_KEY_PATH = "tests.unit.cli.test_filter_spec.sample_key"


def _cfg(op_args: dict, dataset: str = "ds") -> dict:
    return {"datasets": {dataset: {"operations": [{"filter": op_args}]}}}


def _op(cfg: dict, dataset: str = "ds") -> dict:
    return cfg["datasets"][dataset]["operations"][0]["filter"]


@pytest.fixture
def loguru_caplog(caplog):
    """Bridge loguru warnings into pytest's caplog for the test duration."""
    from loguru import logger as _logger

    sink_id = _logger.add(caplog.handler, level="WARNING", format="{message}")
    yield caplog
    _logger.remove(sink_id)


@pytest.fixture
def temp_key_module(tmp_path, monkeypatch):
    """A real importable module whose source the test can rewrite in place.

    Editing a function on disk is the only way to exercise what the pin exists
    for; comparing two different functions would pass even if the digest read
    the name rather than the body.
    """
    names = ("sieval_test_keys", "sieval_test_keys.keys")
    package = tmp_path / "sieval_test_keys"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(body: str) -> None:
        (package / "keys.py").write_text(body, encoding="utf-8")
        for name in names:
            sys.modules.pop(name, None)
        linecache.clearcache()
        importlib.invalidate_caches()

    yield write

    for name in names:
        sys.modules.pop(name, None)


class TestKeyFunctionSpec:
    def test_reads_the_callable_form(self):
        assert key_function_spec({"callable": "pkg.mod.fn"}) == "pkg.mod.fn"

    @pytest.mark.parametrize(
        "by",
        ["col", ["a", "b"], None, 3, {}, {"callable": 3}, {"callable": "f", "x": 1}],
    )
    def test_returns_none_for_anything_else(self, by):
        assert key_function_spec(by) is None


class TestCheckBy:
    def test_accepts_the_three_forms(self):
        assert check_by("subset") is None
        assert check_by(["subset", "key"]) is None
        assert check_by({"callable": "pkg.mod.fn"}) is None

    def test_missing_by(self):
        assert check_by(None) == "'filter' requires 'by'"

    @pytest.mark.parametrize("by", [[], ["a", 2], [None]])
    def test_rejects_a_bad_list(self, by):
        assert "must name one or more columns as strings" in (check_by(by) or "")

    @pytest.mark.parametrize(
        "by", [{}, {"callable": 3}, {"callable": "f", "extra": 1}, {"fn": "f"}]
    )
    def test_rejects_a_bad_mapping(self, by):
        assert "exactly one key, 'callable'" in (check_by(by) or "")

    @pytest.mark.parametrize("by", [3, 1.5, True, object()])
    def test_rejects_other_types(self, by):
        assert "must be a column name" in (check_by(by) or "")


class TestCheckArgNames:
    def test_every_documented_key_is_accepted(self):
        assert (
            check_arg_names(
                {
                    "by": {"callable": "pkg.mod.fn"},
                    "values_file": "picked.json",
                    VALUES_DIGEST_KEY: f"sha256:{'0' * 64}",
                    BY_DIGEST_KEY: f"sha256:{'1' * 64}",
                    "require_all": True,
                    "split": "validation",
                }
            )
            == []
        )

    def test_rejects_an_unknown_key(self):
        # A typo in an *optional* key is otherwise silent: this one reads as
        # require_all simply left at its default, and the assertion is lost.
        problems = check_arg_names({"by": "tag", "require_all_keys": True})
        assert len(problems) == 1
        assert "unknown key(s) ['require_all_keys']" in problems[0]

    def test_names_every_unknown_key_at_once(self):
        problems = check_arg_names({"by": "tag", "zzz": 1, "aaa": 2})
        assert "['aaa', 'zzz']" in problems[0]

    @pytest.mark.parametrize("split", [3, ["test"], {"name": "test"}, True])
    def test_rejects_a_non_string_split(self, split):
        problems = check_arg_names({"by": "tag", "split": split})
        assert any("'split' must be a split name" in p for p in problems)

    def test_an_absent_split_is_fine(self):
        assert check_arg_names({"by": "tag", "value": "a"}) == []


class TestRelativeValuesFiles:
    def test_reports_only_the_relative_ones(self):
        cfg = {
            "datasets": {
                "d1": {"operations": [{"filter": {"values_file": "picked.json"}}]},
                "d2": {"operations": [{"filter": {"values_file": "/abs/picked.json"}}]},
                "d3": {"operations": [{"filter": {"by": "tag", "value": "a"}}]},
            }
        }
        assert relative_values_files(cfg) == ["picked.json"]

    def test_no_filter_operations_is_empty(self):
        assert relative_values_files({"datasets": {"d1": {}}}) == []


class TestCheckValuesSource:
    def test_value_alone_is_fine(self):
        assert check_values_source({"value": "a"}) == []

    def test_values_file_alone_is_fine(self):
        assert check_values_source({"values_file": "k.json"}) == []

    def test_neither_and_both_are_rejected(self):
        assert "exactly one of" in check_values_source({})[0]
        assert (
            "exactly one of"
            in check_values_source({"value": "a", "values_file": "k.json"})[0]
        )

    def test_falsy_value_is_still_a_value(self):
        # Presence, not truthiness: `value: 0` / `value: false` are real keys.
        assert check_values_source({"value": 0}) == []
        assert check_values_source({"value": False}) == []

    def test_values_file_must_be_a_path(self):
        problems = check_values_source({"values_file": 3})
        assert any("must be a path" in p for p in problems)

    def test_require_all_must_be_a_bool(self):
        problems = check_values_source({"value": "a", "require_all": "yes"})
        assert any("must be a boolean" in p for p in problems)

    @pytest.mark.parametrize("digest", ["sha256:abc", "abc", "sha256:" + "z" * 64, 7])
    def test_rejects_a_malformed_digest(self, digest):
        problems = check_values_source(
            {"values_file": "k.json", VALUES_DIGEST_KEY: digest}
        )
        assert any("sha256:<64 hex>" in p for p in problems)

    def test_rejects_a_digest_with_no_file_to_pin(self):
        problems = check_values_source(
            {"value": "a", VALUES_DIGEST_KEY: "sha256:" + "a" * 64}
        )
        assert any("but none is given" in p for p in problems)

    def test_collects_every_problem_rather_than_the_first(self):
        # The accumulate-vs-raise split is the whole reason these return
        # messages instead of raising.
        problems = check_values_source({"values_file": 3, "require_all": "yes"})
        assert len(problems) == 2


class TestCheckByDigest:
    def test_absent_is_fine(self):
        assert check_by_digest({"by": "id", "value": "a"}) == []

    def test_accepts_a_pin_on_a_callable_by(self):
        op_args = {
            "by": {"callable": "pkg.mod.fn"},
            BY_DIGEST_KEY: "sha256:" + "a" * 64,
        }
        assert check_by_digest(op_args) == []

    @pytest.mark.parametrize("digest", ["sha256:abc", "abc", "sha256:" + "z" * 64, 7])
    def test_rejects_a_malformed_digest(self, digest):
        problems = check_by_digest({"by": {"callable": "f.g"}, BY_DIGEST_KEY: digest})
        assert any("sha256:<64 hex>" in p for p in problems)

    @pytest.mark.parametrize("by", ["id", ["a", "b"], None])
    def test_rejects_a_pin_with_no_key_function_to_pin(self, by):
        problems = check_by_digest({"by": by, BY_DIGEST_KEY: "sha256:" + "a" * 64})
        assert any("does not name one" in p for p in problems)


class TestComputeValuesDigest:
    def test_matches_sha256_of_the_bytes(self):
        data = b'["a", "b"]'
        assert compute_values_digest(data) == (
            "sha256:" + hashlib.sha256(data).hexdigest()
        )

    def test_differs_on_a_one_byte_change(self):
        assert compute_values_digest(b'["a"]') != compute_values_digest(b'["b"]')


class TestComputeKeyFunctionDigest:
    def test_matches_sha256_of_the_source(self):
        source = inspect.getsource(sample_key).encode("utf-8")
        assert compute_key_function_digest(sample_key) == (
            "sha256:" + hashlib.sha256(source).hexdigest()
        )

    def test_distinguishes_two_functions(self):
        assert compute_key_function_digest(sample_key) != (
            compute_key_function_digest(other_key)
        )

    @pytest.mark.parametrize("fn", [len, dict.get, functools.partial(sample_key)])
    def test_unreadable_source_is_none_rather_than_an_error(self, fn):
        # A builtin, a C-level descriptor and a partial are all legitimate keys
        # from Python; refusing to run them would be worse than not pinning.
        assert compute_key_function_digest(fn) is None


class TestResolveValuesPath:
    def test_relative_resolves_against_the_config_dir(self, tmp_path):
        assert (
            resolve_values_path("sub/k.json", tmp_path)
            == (tmp_path / "sub" / "k.json").resolve()
        )

    def test_absolute_is_left_alone(self, tmp_path):
        target = tmp_path / "k.json"
        assert resolve_values_path(str(target), tmp_path / "other") == target


class TestPinValuesFile:
    def test_injects_the_digest(self, tmp_path):
        (tmp_path / "k.json").write_text('["a"]', encoding="utf-8")
        cfg = _cfg({"by": "id", "values_file": "k.json"})

        pin_filter_digests(cfg, tmp_path)

        assert _op(cfg)[VALUES_DIGEST_KEY] == compute_values_digest(b'["a"]')

    def test_the_digest_tracks_the_contents(self, tmp_path):
        # The point of the whole mechanism: two configs identical on disk must
        # not compare equal once the file behind them differs.
        path = tmp_path / "k.json"
        path.write_text('["a"]', encoding="utf-8")
        first = _cfg({"by": "id", "values_file": "k.json"})
        pin_filter_digests(first, tmp_path)

        path.write_text('["a", "b"]', encoding="utf-8")
        second = _cfg({"by": "id", "values_file": "k.json"})
        pin_filter_digests(second, tmp_path)

        assert first != second

    def test_an_existing_matching_digest_is_left_alone(self, tmp_path):
        (tmp_path / "k.json").write_text('["a"]', encoding="utf-8")
        digest = compute_values_digest(b'["a"]')
        cfg = _cfg({"by": "id", "values_file": "k.json", VALUES_DIGEST_KEY: digest})

        pin_filter_digests(cfg, tmp_path)

        assert _op(cfg)[VALUES_DIGEST_KEY] == digest

    def test_a_stale_digest_raises_and_names_the_dataset(self, tmp_path):
        # Re-running a persisted effective_config.yaml whose values file moved
        # under it. Silently re-pinning would score a different sample set.
        (tmp_path / "k.json").write_text('["a", "b"]', encoding="utf-8")
        cfg = _cfg(
            {
                "by": "id",
                "values_file": "k.json",
                VALUES_DIGEST_KEY: "sha256:" + "0" * 64,
            },
            dataset="curated",
        )

        with pytest.raises(ValueError, match="Dataset 'curated'") as exc:
            pin_filter_digests(cfg, tmp_path)
        assert "has changed since this config recorded it" in str(exc.value)

    def test_a_missing_file_raises(self, tmp_path):
        cfg = _cfg({"by": "id", "values_file": "gone.json"})
        with pytest.raises(ValueError, match="'values_file' not found"):
            pin_filter_digests(cfg, tmp_path)

    def test_an_inline_value_is_untouched(self, tmp_path):
        cfg = _cfg({"by": "id", "value": "a"})
        pin_filter_digests(cfg, tmp_path)
        assert VALUES_DIGEST_KEY not in _op(cfg)

    def test_pins_every_dataset_and_every_operation(self, tmp_path):
        (tmp_path / "a.json").write_text('["a"]', encoding="utf-8")
        (tmp_path / "b.json").write_text('["b"]', encoding="utf-8")
        cfg = {
            "datasets": {
                "one": {
                    "operations": [{"filter": {"by": "id", "values_file": "a.json"}}]
                },
                "two": {
                    "operations": [
                        {"slice": {"n": 2}},
                        {"filter": {"by": "id", "values_file": "b.json"}},
                    ]
                },
            }
        }

        pin_filter_digests(cfg, tmp_path)

        assert _op(cfg, "one")[VALUES_DIGEST_KEY] == compute_values_digest(b'["a"]')
        assert cfg["datasets"]["two"]["operations"][1]["filter"][VALUES_DIGEST_KEY] == (
            compute_values_digest(b'["b"]')
        )

    @pytest.mark.parametrize(
        "cfg",
        [
            {},
            {"datasets": None},
            {"datasets": {"ds": None}},
            {"datasets": {"ds": {"operations": "nope"}}},
            {"datasets": {"ds": {"operations": [None]}}},
            {"datasets": {"ds": {"operations": [{"a": 1, "b": 2}]}}},
            {"datasets": {"ds": {"operations": [{"filter": "nope"}]}}},
            {"datasets": {"ds": {"operations": [{"filter": {"values_file": 3}}]}}},
            {"datasets": {"ds": {"operations": [{"filter": {"by": {"callable": 3}}}]}}},
        ],
    )
    def test_malformed_configs_are_left_for_the_validator(self, cfg, tmp_path):
        # A shape error raised here would pre-empt cli.validation's better one.
        pin_filter_digests(cfg, tmp_path)


class TestPinKeyFunction:
    def test_injects_the_digest(self, tmp_path):
        cfg = _cfg({"by": {"callable": _SAMPLE_KEY_PATH}, "value": "a"})

        pin_filter_digests(cfg, tmp_path)

        assert _op(cfg)[BY_DIGEST_KEY] == compute_key_function_digest(sample_key)

    @pytest.mark.parametrize("by", ["id", ["a", "b"]])
    def test_a_column_by_has_nothing_to_pin(self, by, tmp_path):
        cfg = _cfg({"by": by, "value": "a"})
        pin_filter_digests(cfg, tmp_path)
        assert BY_DIGEST_KEY not in _op(cfg)

    def test_an_existing_matching_digest_is_left_alone(self, tmp_path):
        digest = compute_key_function_digest(sample_key)
        cfg = _cfg(
            {"by": {"callable": _SAMPLE_KEY_PATH}, "value": "a", BY_DIGEST_KEY: digest}
        )

        pin_filter_digests(cfg, tmp_path)

        assert _op(cfg)[BY_DIGEST_KEY] == digest

    def test_a_stale_digest_raises_and_names_the_dataset(self, tmp_path):
        cfg = _cfg(
            {
                "by": {"callable": _SAMPLE_KEY_PATH},
                "value": "a",
                BY_DIGEST_KEY: "sha256:" + "0" * 64,
            },
            dataset="curated",
        )

        with pytest.raises(ValueError, match="Dataset 'curated'") as exc:
            pin_filter_digests(cfg, tmp_path)
        assert "has changed since this config recorded it" in str(exc.value)

    def test_an_unresolvable_path_is_left_for_the_session(self, tmp_path):
        # The session's message names the dataset and what could not be
        # imported; raising here would replace it with a worse one.
        cfg = _cfg({"by": {"callable": "no.such.module.fn"}, "value": "a"})

        pin_filter_digests(cfg, tmp_path)

        assert BY_DIGEST_KEY not in _op(cfg)

    def test_an_unpinnable_callable_warns_rather_than_blocking(
        self, tmp_path, loguru_caplog
    ):
        cfg = _cfg({"by": {"callable": "builtins.len"}, "value": "a"})

        pin_filter_digests(cfg, tmp_path)

        assert BY_DIGEST_KEY not in _op(cfg)
        assert "no readable source" in loguru_caplog.text

    def test_an_unpinnable_callable_under_an_existing_pin_raises(self, tmp_path):
        # The pin cannot be checked, and an unverifiable pin is the one thing
        # the resume contract cannot wave through.
        cfg = _cfg(
            {
                "by": {"callable": "builtins.len"},
                "value": "a",
                BY_DIGEST_KEY: "sha256:" + "0" * 64,
            }
        )

        with pytest.raises(ValueError, match="cannot be checked"):
            pin_filter_digests(cfg, tmp_path)

    def test_both_digests_are_pinned_on_one_operation(self, tmp_path):
        (tmp_path / "k.json").write_text('["a"]', encoding="utf-8")
        cfg = _cfg({"by": {"callable": _SAMPLE_KEY_PATH}, "values_file": "k.json"})

        pin_filter_digests(cfg, tmp_path)

        assert _op(cfg)[VALUES_DIGEST_KEY] == compute_values_digest(b'["a"]')
        assert _op(cfg)[BY_DIGEST_KEY] == compute_key_function_digest(sample_key)


class TestTheKeyFunctionPinTracksTheSource:
    def test_editing_the_body_changes_the_digest(self, tmp_path, temp_key_module):
        path = "sieval_test_keys.keys.key"
        temp_key_module("def key(row):\n    return row['id']\n")
        first = _cfg({"by": {"callable": path}, "value": "a"})
        pin_filter_digests(first, tmp_path)

        temp_key_module("def key(row):\n    return row['id'].lower()\n")
        second = _cfg({"by": {"callable": path}, "value": "a"})
        pin_filter_digests(second, tmp_path)

        assert _op(first)[BY_DIGEST_KEY] != _op(second)[BY_DIGEST_KEY]

    def test_an_edited_function_is_rejected_against_the_recorded_pin(
        self, tmp_path, temp_key_module
    ):
        # What the mechanism is for: a persisted effective_config.yaml re-run
        # after the function it names was edited must not resume silently.
        path = "sieval_test_keys.keys.key"
        temp_key_module("def key(row):\n    return row['id']\n")
        cfg = _cfg({"by": {"callable": path}, "value": "a"})
        pin_filter_digests(cfg, tmp_path)

        temp_key_module("def key(row):\n    return row['id'].lower()\n")

        with pytest.raises(ValueError, match="has changed since this config"):
            pin_filter_digests(cfg, tmp_path)

    def test_an_untouched_function_still_matches(self, tmp_path, temp_key_module):
        # The other half: re-importing the same source must not trip the pin,
        # or every resume of a callable filter would fail.
        path = "sieval_test_keys.keys.key"
        body = "def key(row):\n    return row['id']\n"
        temp_key_module(body)
        cfg = _cfg({"by": {"callable": path}, "value": "a"})
        pin_filter_digests(cfg, tmp_path)
        pinned = _op(cfg)[BY_DIGEST_KEY]

        temp_key_module(body)
        pin_filter_digests(cfg, tmp_path)

        assert _op(cfg)[BY_DIGEST_KEY] == pinned
