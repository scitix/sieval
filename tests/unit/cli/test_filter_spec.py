"""
Unit tests for sieval/cli/_filter_spec.py.

Covers: check_by, check_values_source, resolve_values_path,
compute_values_digest, and pin_values_files (inject / verify / reject).

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import hashlib

import pytest

from sieval.cli._filter_spec import (
    DIGEST_KEY,
    check_by,
    check_values_source,
    compute_values_digest,
    pin_values_files,
    resolve_values_path,
)


def _cfg(op_args: dict, dataset: str = "ds") -> dict:
    return {"datasets": {dataset: {"operations": [{"filter": op_args}]}}}


def _op(cfg: dict, dataset: str = "ds") -> dict:
    return cfg["datasets"][dataset]["operations"][0]["filter"]


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
        problems = check_values_source({"values_file": "k.json", DIGEST_KEY: digest})
        assert any("sha256:<64 hex>" in p for p in problems)

    def test_rejects_a_digest_with_no_file_to_pin(self):
        problems = check_values_source({"value": "a", DIGEST_KEY: "sha256:" + "a" * 64})
        assert any("but none is given" in p for p in problems)

    def test_collects_every_problem_rather_than_the_first(self):
        # The accumulate-vs-raise split is the whole reason these return
        # messages instead of raising.
        problems = check_values_source({"values_file": 3, "require_all": "yes"})
        assert len(problems) == 2


class TestComputeValuesDigest:
    def test_matches_sha256_of_the_bytes(self):
        data = b'["a", "b"]'
        assert compute_values_digest(data) == (
            "sha256:" + hashlib.sha256(data).hexdigest()
        )

    def test_differs_on_a_one_byte_change(self):
        assert compute_values_digest(b'["a"]') != compute_values_digest(b'["b"]')


class TestResolveValuesPath:
    def test_relative_resolves_against_the_config_dir(self, tmp_path):
        assert (
            resolve_values_path("sub/k.json", tmp_path)
            == (tmp_path / "sub" / "k.json").resolve()
        )

    def test_absolute_is_left_alone(self, tmp_path):
        target = tmp_path / "k.json"
        assert resolve_values_path(str(target), tmp_path / "other") == target


class TestPinValuesFiles:
    def test_injects_the_digest(self, tmp_path):
        (tmp_path / "k.json").write_text('["a"]', encoding="utf-8")
        cfg = _cfg({"by": "id", "values_file": "k.json"})

        pin_values_files(cfg, tmp_path)

        assert _op(cfg)[DIGEST_KEY] == compute_values_digest(b'["a"]')

    def test_the_digest_tracks_the_contents(self, tmp_path):
        # The point of the whole mechanism: two configs identical on disk must
        # not compare equal once the file behind them differs.
        path = tmp_path / "k.json"
        path.write_text('["a"]', encoding="utf-8")
        first = _cfg({"by": "id", "values_file": "k.json"})
        pin_values_files(first, tmp_path)

        path.write_text('["a", "b"]', encoding="utf-8")
        second = _cfg({"by": "id", "values_file": "k.json"})
        pin_values_files(second, tmp_path)

        assert first != second

    def test_an_existing_matching_digest_is_left_alone(self, tmp_path):
        (tmp_path / "k.json").write_text('["a"]', encoding="utf-8")
        digest = compute_values_digest(b'["a"]')
        cfg = _cfg({"by": "id", "values_file": "k.json", DIGEST_KEY: digest})

        pin_values_files(cfg, tmp_path)

        assert _op(cfg)[DIGEST_KEY] == digest

    def test_a_stale_digest_raises_and_names_the_dataset(self, tmp_path):
        # Re-running a persisted effective_config.yaml whose values file moved
        # under it. Silently re-pinning would score a different sample set.
        (tmp_path / "k.json").write_text('["a", "b"]', encoding="utf-8")
        cfg = _cfg(
            {"by": "id", "values_file": "k.json", DIGEST_KEY: "sha256:" + "0" * 64},
            dataset="curated",
        )

        with pytest.raises(ValueError, match="Dataset 'curated'") as exc:
            pin_values_files(cfg, tmp_path)
        assert "has changed since this config recorded it" in str(exc.value)

    def test_a_missing_file_raises(self, tmp_path):
        cfg = _cfg({"by": "id", "values_file": "gone.json"})
        with pytest.raises(ValueError, match="'values_file' not found"):
            pin_values_files(cfg, tmp_path)

    def test_an_inline_value_is_untouched(self, tmp_path):
        cfg = _cfg({"by": "id", "value": "a"})
        pin_values_files(cfg, tmp_path)
        assert DIGEST_KEY not in _op(cfg)

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

        pin_values_files(cfg, tmp_path)

        assert _op(cfg, "one")[DIGEST_KEY] == compute_values_digest(b'["a"]')
        assert cfg["datasets"]["two"]["operations"][1]["filter"][DIGEST_KEY] == (
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
        ],
    )
    def test_malformed_configs_are_left_for_the_validator(self, cfg, tmp_path):
        # A shape error raised here would pre-empt cli.validation's better one.
        pin_values_files(cfg, tmp_path)
