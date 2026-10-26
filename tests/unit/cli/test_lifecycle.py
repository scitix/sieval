"""Model lifecycles, implied barriers, phases and placement (``sieval.cli.lifecycle``).

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import pytest

from sieval.cli.lifecycle import (
    Lifecycle,
    LifecycleError,
    LifecycleFitError,
    Placement,
    check_iteration_barriers,
    check_lifecycles_launched,
    effective_barriers,
    model_lifecycles,
    parse_lifecycle,
    place,
    placed_commands,
    plan_phases,
    role_models,
    runs_before_stop,
    visible_devices,
)
from sieval.infer.backends.translator import BackendCommand


class TestParseLifecycle:
    def test_from_and_until(self) -> None:
        assert parse_lifecycle("m", {"until": "inference"}) == Lifecycle(
            None, "inference"
        )
        assert parse_lifecycle(
            "m", {"from": "preprocess", "until": "feedback"}
        ) == Lifecycle("preprocess", "feedback")
        assert parse_lifecycle("m", {}) == Lifecycle()

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ("inference", "must be a mapping"),
            ({"until": "data"}, "until must be one of"),
            ({"from": "inference", "until": "preprocess"}, "is not before until"),
            ({"from": "inference", "until": "inference"}, "is not before until"),
            ({"from": "report"}, "no stage after it"),
            ({"start": "inference"}, "unknown key"),
        ],
    )
    def test_refusals_name_the_model(self, raw: object, message: str) -> None:
        with pytest.raises(LifecycleError, match=f"Model 'm'.*{message}"):
            parse_lifecycle("m", raw)

    def test_a_derived_model_cannot_set_one(self) -> None:
        config = {
            "models": {
                "root": {"path": "/m"},
                "child": {"base": "root", "lifecycle": {"until": "inference"}},
            }
        }
        with pytest.raises(LifecycleError, match="derived models"):
            model_lifecycles(config)


class TestEffectiveBarriers:
    def test_union_of_stop_after_and_every_boundary_with_what_implied_each(
        self,
    ) -> None:
        config = {
            "runner_config": {"stop_after": "feedback"},
            "models": {
                "subject": {"path": "/s", "lifecycle": {"until": "inference"}},
                "judge": {"path": "/j", "lifecycle": {"from": "inference"}},
            },
        }
        assert effective_barriers(config) == {
            "inference": ["judge.lifecycle.from", "subject.lifecycle.until"],
            "feedback": ["stop_after"],
        }

    def test_none_without_stop_after_or_lifecycle(self) -> None:
        assert effective_barriers({"models": {"m": {"path": "/m"}}}) == {}


class TestChecks:
    def test_a_lifecycle_on_a_model_not_launched_is_refused(self) -> None:
        with pytest.raises(LifecycleError, match="api"):
            check_lifecycles_launched({"api": Lifecycle(None, "inference")}, ["m"])

    def test_an_iterating_task_refuses_an_inference_barrier(self) -> None:
        config = {
            "runner_config": {"max_iterations": 1},
            "tasks": {"loop": {"runner_config": {"max_iterations": 3}}, "once": {}},
        }
        with pytest.raises(LifecycleError, match="loop \\(max_iterations 3\\)"):
            check_iteration_barriers(config, ["inference"])
        check_iteration_barriers(config, ["feedback"])

    def test_role_models_reads_grader_and_extractor_names(self) -> None:
        config = {
            "models": {"s": {}, "j": {}},
            "tasks": {
                "a": {"args": {"grader": "j"}},
                "b": {"args": {"extractor": "self"}},
                "c": {"args": {"grader": {"model": "remote"}}},
            },
        }
        assert role_models(config) == {"a": {"j"}}


class TestPhases:
    def test_sequential_lifecycles_split_the_run(self) -> None:
        lifecycles = {
            "subject": Lifecycle(None, "inference"),
            "judge": Lifecycle("inference", None),
        }
        phases = plan_phases(("inference",), lifecycles, ["subject", "judge"])
        assert [(p.start, p.stop, p.models) for p in phases] == [
            (None, "inference", ("subject",)),
            ("inference", None, ("judge",)),
        ]

    def test_a_stop_ends_the_phases(self) -> None:
        lifecycles = {"judge": Lifecycle("inference", None)}
        phases = plan_phases(
            ("inference",), lifecycles, ["subject", "judge"], "inference"
        )
        assert [p.models for p in phases] == [("subject",)]

    def test_runs_before_stop(self) -> None:
        judge = Lifecycle("inference", None)
        assert not runs_before_stop(judge, "inference")
        assert not runs_before_stop(judge, "preprocess")
        assert runs_before_stop(judge, "feedback")
        assert runs_before_stop(Lifecycle(None, "feedback"), "preprocess")


class TestPlace:
    def test_a_model_alone_takes_every_gpu_on_the_default_port(self) -> None:
        phases = plan_phases(
            ("inference",),
            {"s": Lifecycle(None, "inference"), "j": Lifecycle("inference", None)},
            ["s", "j"],
        )
        placements = place(phases, {"s": 8, "j": 8}, 8)
        assert placements == {
            "s": Placement(8, None, 0),
            "j": Placement(8, None, 0),
        }

    def test_models_alive_together_get_disjoint_blocks_and_ports(self) -> None:
        phases = plan_phases(
            ("inference",),
            {"s": Lifecycle(None, "inference")},
            ["s", "j", "x"],
        )
        placements = place(phases, {"s": 4, "j": 2, "x": 2}, 8)
        devices = {name: p.devices for name, p in placements.items()}
        assert devices == {"j": (0, 1), "s": (2, 3, 4, 5), "x": (6, 7)}
        ports = {p.port_offset for p in placements.values()}
        assert len(ports) == 3

    def test_too_many_gpus_in_one_phase_is_refused_naming_it(self) -> None:
        phases = plan_phases((), {}, ["s", "j"])
        with pytest.raises(LifecycleFitError, match="start → end: j 6, s 4 need 10"):
            place(phases, {"s": 4, "j": 6}, 8)

    def test_a_model_with_many_commands_widens_the_next_port_offset(self) -> None:
        phases = plan_phases((), {}, ["a", "b", "c"])
        placements = place(phases, {"a": 1, "b": 1, "c": 1}, 8, ports={"a": 12, "b": 1})
        offsets = {name: p.port_offset for name, p in placements.items()}
        assert offsets == {"a": 0, "b": 12, "c": 22}

    def test_unknown_gpus_check_nothing(self) -> None:
        phases = plan_phases((), {}, ["s", "j"])
        placements = place(phases, {"s": 8, "j": 8}, None)
        assert all(p.devices is None for p in placements.values())


class TestPlacedCommands:
    def test_sets_devices_and_moves_the_port_and_health_url(self) -> None:
        command = BackendCommand(
            cli_args=["sglang", "serve", "--port", "30000"],
            backend="sglang",
            health_url="http://localhost:30000/health",
        )
        (placed,) = placed_commands([command], Placement(2, (2, 3), 10))
        assert placed.cli_args[-1] == "30010"
        assert placed.health_url == "http://localhost:30010/health"
        assert placed.env["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert command.cli_args[-1] == "30000"

    def test_device_indices_select_from_the_inherited_visible_devices(self) -> None:
        command = BackendCommand(cli_args=["vllm"], backend="vllm")
        (placed,) = placed_commands(
            [command], Placement(2, (0, 1), 0), visible=["4", "5", "6", "7"]
        )
        assert placed.env["CUDA_VISIBLE_DEVICES"] == "4,5"

    @pytest.mark.parametrize(
        ("environ", "expected"),
        [({}, None), ({"CUDA_VISIBLE_DEVICES": "4, 5,,6"}, ["4", "5", "6"])],
    )
    def test_visible_devices(self, environ: dict, expected: list | None) -> None:
        assert visible_devices(environ) == expected

    def test_a_user_set_device_list_wins(self) -> None:
        command = BackendCommand(
            cli_args=["vllm"], env={"CUDA_VISIBLE_DEVICES": "7"}, backend="vllm"
        )
        (placed,) = placed_commands([command], Placement(1, (0,), 0))
        assert placed.env["CUDA_VISIBLE_DEVICES"] == "7"

    def test_unchanged_on_every_gpu_and_the_default_port(self) -> None:
        command = BackendCommand(cli_args=["vllm", "--port", "8000"])
        assert placed_commands([command], Placement(8, None, 0)) == [command]
