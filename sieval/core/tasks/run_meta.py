"""Stage identities and run history recorded in a task directory's ``meta.json``.

``meta.json`` is the one record of where a task directory came from. Each run
rewrites it atomically, so this module only builds the document; the saver
writes it.

The four stage identities describe the run that produced the current
``report.json``:

=============  ==============================================================
``data``       Every selected sample, keyed by ``(sample_id, repeat_index)``
               and digested over its whole raw content.
``prompt``     Every sample's ``PromptRecord.prompt``, before the model's chat
               template.
``inference``  The candidate's resolved request arguments, dialect and engine,
               the task's own inference arguments (``n``), the result-changing
               runner keys and the serving recipe, plus ``requests``: every
               sample's request parameters as the dialect sent them, so a
               ``stop`` or ``max_tokens`` a task passes at call time counts.
               Only what defines a request counts there: its body and whether
               it streamed, never the method, path, query or model name.
               The model under test, the engine build and the SiEval version
               are left out.
``grader``     The source of every module the task class reaches by a static
               import walk under :data:`GRADER_PACKAGES` and the ``sieval_*``
               plugin package of each class in its MRO, and each judge's model,
               version, sampling arguments and prompt.
=============  ==============================================================

Every digest is ``sha256:`` over canonical JSON (sorted keys, no whitespace).
A stage whose input could not be read is ``None``: unknown, never a guess. A
per-sample stage with one sample it could not digest is unknown as a whole,
and says how many in ``unknown``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import ast
import hashlib
import importlib.util
import json
import os
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, NotRequired, TypedDict, cast

from sieval.core.models import Model
from sieval.core.types import JSONValue

#: Version of the ``identity`` block's layout.
IDENTITY_SCHEMA = 1

#: A per-sample digest of a value that was legitimately absent: the prompt of a
#: sample whose ``preprocess`` failed, the requests of one never inferred. Kept
#: apart from ``None``, which is a digest that could not be taken.
ABSENT = "absent"

#: Stage names in pipeline order; ``runs[].reran`` lists them in this order.
STAGES: tuple[str, ...] = ("data", "prompt", "inference", "grader")

#: Package roots the grader walk follows. ``sieval.core`` and
#: ``sieval.datasets`` are left out on purpose: data has its own identity, and a
#: core change is told by the SiEval version and the replay a release runs.
#: ``sieval_shared`` is the package sieval-plugins variants share.
GRADER_PACKAGES: tuple[str, ...] = (
    "sieval.tasks",
    "sieval.community",
    "sieval_plugins",
    "sieval_shared",
)

#: Environment variable naming the served judge, set by the orchestrator that
#: serves it. A plain value applies to every judge role; a JSON object maps role
#: to reference. Checked once by :func:`validate_judge_model_ref`.
JUDGE_MODEL_REF_ENV = "SIEVAL_JUDGE_MODEL_REF"

#: The runner keys whose value changes the samples a run produces.
INFERENCE_RUNNER_KEYS: tuple[str, ...] = (
    "deterministic",
    "max_iterations",
    "max_retries",
)


class DigestedStage(TypedDict):
    """A per-sample stage: an aggregate digest and how many samples it covers.

    ``digest`` is ``None`` when ``unknown`` samples (one or more) could not be
    digested: a fold that left them out, or spelled them alike, would claim
    two different sample sets are the same.
    """

    digest: str | None
    samples: int
    unknown: NotRequired[int]


class InferenceIdentity(TypedDict):
    """``digest`` covers the configuration; ``requests`` what was sent."""

    digest: str
    requests: NotRequired[DigestedStage]


class JudgeIdentity(TypedDict):
    """One model a task calls while grading.

    ``version`` is what the judge API reported as its model (``response_model``)
    once it was called, else ``None``. ``prompt`` is the digest of the template
    the task declares through :meth:`~sieval.core.tasks.task.Task.judge_prompts`,
    else ``None``; a template in code is already covered by ``modules``. ``ref``
    is present only when :data:`JUDGE_MODEL_REF_ENV` names the served model.
    """

    role: str
    model: str
    version: str | None
    sampling: dict[str, JSONValue]
    prompt: str | None
    ref: NotRequired[str]


class GraderIdentity(TypedDict):
    digest: str
    modules: dict[str, str]
    judges: list[JudgeIdentity]


class RunIdentity(TypedDict):
    """The ``identity`` block. A stage is ``None`` while unknown.

    Two runs share a stage only when every digest in it matches: for
    ``inference`` that is ``digest`` *and* ``requests.digest``, and a ``None``
    digest on either side matches nothing.
    """

    schema: int
    data: DigestedStage | None
    prompt: DigestedStage | None
    inference: InferenceIdentity | None
    grader: GraderIdentity | None


class RunFrom(TypedDict):
    dir: str
    identity: RunIdentity


# Functional form: ``from`` is a keyword.
RunRecord = TypedDict(
    "RunRecord",
    {
        "sieval": str,
        "at": str,
        "start": Literal["fresh", "resume", "from"],
        "ended": NotRequired[str],
        "reran": NotRequired[list[str]],
        "reported": NotRequired[bool],
        "interrupted": NotRequired[bool],
        "config": NotRequired[list[str]],
        "from": NotRequired[RunFrom],
    },
)
"""One ``runs[]`` entry, appended when a run starts.

While the run is in flight the entry has neither ``ended`` nor ``reran``.
A run that saves its ``report.json`` writes both: ``ended`` (ISO 8601 UTC, like
``at``) and ``reran``, the stages it executed, ``[]`` for a fast resume.
``reported: true`` says the run wrote ``report.json``, which a fast resume does
not. ``report()`` has no identity, so a run that only re-aggregated finished
samples is ``reran: []`` with ``reported: true``. A run
stopped by an interrupt writes ``ended`` and ``interrupted: true``, and no
``reran``. An entry with no ``ended`` is a run that is still going, or was
killed or failed before it could say so. ``config`` is the resume-mutable config
diff, present when non-empty. ``from`` names the directory a resume across
tasks copied; nothing writes it until that resume exists.
"""


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def _default(value: object) -> Any:
    """Encode the non-JSON values a raw sample or a model default may hold."""
    if isinstance(value, (set, frozenset)):
        return sorted(canonical_json(v) for v in value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, bytes):
        return {"__bytes__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return dict(value)
    tolist = getattr(value, "tolist", None)  # numpy arrays and scalars
    if callable(tolist):
        return tolist()
    tobytes = getattr(value, "tobytes", None)  # PIL images
    if callable(tobytes):
        raw = tobytes()
        if isinstance(raw, bytes):
            return {"__bytes__": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot digest a {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Canonical JSON: sorted keys, no whitespace, ASCII only.

    NaN and the infinities are written as the bare tokens ``NaN``,
    ``Infinity`` and ``-Infinity``. That is not strict JSON, but this text is
    only ever hashed: refusing them would leave a sample holding a missing
    float with no digest, and spelling them as strings would make them equal
    to the string ``"NaN"``.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
        default=_default,
    )


def digest(value: object) -> str:
    """``sha256:`` over :func:`canonical_json` of *value*."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def prompt_of(preprocess_result: object) -> object:
    """The part of a ``preprocess`` result the prompt identity covers.

    ``PromptRecord.prompt`` for a record; the whole value for a task that still
    returns a bare prompt.
    """
    if isinstance(preprocess_result, dict) and "prompt" in preprocess_result:
        return cast(dict[str, object], preprocess_result)["prompt"]
    return preprocess_result


def _sample_key(sample_id: str | int, repeat_index: int | None) -> tuple:
    if isinstance(sample_id, int):
        return (0, sample_id, "", -1 if repeat_index is None else repeat_index)
    return (1, 0, str(sample_id), -1 if repeat_index is None else repeat_index)


def aggregate_digest(
    entries: Iterable[tuple[str | int, int | None, str | None]],
) -> DigestedStage:
    """Fold per-sample ``(sample_id, repeat_index, digest)`` into one stage.

    :data:`ABSENT` stays in the fold: a sample whose ``preprocess`` failed has
    no prompt, which is still a fact about the run. ``None`` is a digest that
    could not be taken, and makes the whole stage unknown.
    """
    rows = sorted(entries, key=lambda e: _sample_key(e[0], e[1]))
    unknown = sum(d is None for _, _, d in rows)
    if unknown:
        return {"digest": None, "samples": len(rows), "unknown": unknown}
    return {
        "digest": digest([[sid, rep, d] for sid, rep, d in rows]),
        "samples": len(rows),
    }


#: ``ModelMeta`` keys that say where a model was reached, not which model it is.
_CALL_KEY_EXCLUDED = frozenset({"api_base", "provenance"})


def call_key(model_meta: Mapping) -> str:
    """Which model made a call: its ``ModelMeta`` without endpoint or provenance.

    Both are left out because a resume may reach the same model at another
    address or deployment. Two models alike in name, sampling and ``extra`` are
    the same model as far as a run's results go.
    """
    return canonical_json(
        {k: v for k, v in model_meta.items() if k not in _CALL_KEY_EXCLUDED}
    )


#: Request-parameter keys that say how a request was sent, or name the model,
#: rather than what was asked: the keys a dialect's ``request_params`` actually
#: carries. ``stream_options`` only asks for usage.
_TRANSPORT_KEYS = frozenset(
    {"model", "method", "path", "query", "stream", "stream_options"}
)


def request_definition(params: Mapping[Any, object]) -> dict[str, object]:
    """What defines a request, out of a dialect's ``request_params``.

    Dialects record two shapes: a flat body (OpenAI, Anthropic, SGLang) and an
    HTTP envelope with ``body``, ``path`` and ``query`` (Google, whose path
    names the model and whose method and ``alt`` query say whether it
    streamed). Both reduce to ``{"body": …, "stream": bool}``, so neither the
    model under test nor the gateway prefix reaches the digest.
    """
    body = params.get("body")
    if isinstance(body, Mapping) and "path" in params:
        path = params.get("path")
        query = params.get("query")
        streamed_path = isinstance(path, str) and path.endswith(
            ":streamGenerateContent"
        )
        sse = (
            isinstance(query, Mapping)
            and cast(Mapping[Any, object], query).get("alt") == "sse"
        )
        stream = streamed_path or sse
        source: Mapping[Any, object] = cast(Mapping[Any, object], body)
    else:
        stream = params.get("stream") is True
        source = params
    return {
        "body": {k: v for k, v in source.items() if k not in _TRANSPORT_KEYS},
        "stream": bool(stream),
    }


def request_digest(
    model_calls: Iterable[Mapping[str, object]], candidate: str
) -> str | None:
    """Digest what the calls *candidate* made asked for.

    *candidate* is the model under test's :func:`call_key`; a judge called
    from the same stage is left out. Each call counts by
    :func:`request_definition`. ``None`` when no candidate call was recorded,
    or one has no ``request_params``: what was sent is then unknown, not
    "nothing".
    """
    sent = []
    for call in model_calls:
        model = call.get("model")
        if not isinstance(model, Mapping) or call_key(model) != candidate:
            continue
        params = call.get("request_params")
        if not isinstance(params, Mapping):
            return None
        sent.append(request_definition(params))
    if not sent:
        return None
    try:
        return digest(sent)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def model_request_args(model: Model) -> dict[str, JSONValue]:
    """A model's resolved request arguments without its name or endpoint.

    Reads the builder defaults and ``extra`` from ``Model.meta()``, plus the
    runtime plan's capability defaults and the dialect.
    """
    meta = model.meta()
    args: dict[str, JSONValue] = {"params": dict(meta["default_params"])}
    if "extra" in meta:
        args["extra"] = dict(meta["extra"])
    plan = model.runtime_plan
    if plan is not None:
        if plan.request_defaults.values:
            args["request_defaults"] = plan.request_defaults.to_json_value()
        args["dialect"] = plan.dialect_id
    return args


def inference_identity(
    *,
    model: Model,
    task_args: Mapping[str, JSONValue],
    runner_keys: Mapping[str, JSONValue],
    recipe: Mapping[str, JSONValue] | None,
) -> InferenceIdentity:
    """Digest the inference stage's configuration, leaving out the model under test.

    What the candidate was actually sent is folded per sample into
    ``requests`` once the run ends; this digest is the part known up front.
    """
    payload = {
        "model_args": model_request_args(model),
        "engine": model.deployment.engine.engine_id,
        "task_args": dict(task_args),
        "runner": {key: runner_keys.get(key) for key in INFERENCE_RUNNER_KEYS},
        "recipe": dict(recipe) if recipe is not None else None,
    }
    return {"digest": digest(payload)}


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------


def _followed(name: str, roots: Iterable[str] = GRADER_PACKAGES) -> bool:
    return any(name == root or name.startswith(root + ".") for root in roots)


def _own_root(module: str) -> str | None:
    """The top-level package of *module* when it is a SiEval plugin.

    A plugin need not be named ``sieval_plugins``: a task class in
    ``sieval_sysbench_plugin.task`` makes ``sieval_sysbench_plugin`` part of its
    grader. Only ``sieval_*`` packages with source on disk count, so a test
    module, a third-party base class or ``sieval`` itself (whose ``core`` and
    ``datasets`` stay out) never widens the walk.
    """
    top = module.partition(".")[0]
    if not top.startswith("sieval_"):
        return None
    located = _locate(top)
    if located is None or located[0] is None:
        return None
    return top


def _root_dirs(top: str) -> list[str]:
    """Search locations of top-level package *top*, without importing it."""
    module = sys.modules.get(top)
    if module is not None:
        return list(getattr(module, "__path__", ()))
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return []
    return list(spec.submodule_search_locations or ()) if spec is not None else []


def _locate(name: str) -> tuple[str | None, bool] | None:
    """Find module *name*'s source file by path alone: ``(file, is_package)``.

    Nothing is imported, so walking a module never runs it. ``file`` is
    ``None`` for a namespace package. ``None`` when *name* is not a module (an
    attribute named by ``from pkg import name``, or something not installed).
    """
    top, *rest = name.split(".")
    dirs = _root_dirs(top)
    if not dirs:
        return None
    if not rest:
        init = next(
            (
                os.path.join(d, "__init__.py")
                for d in dirs
                if os.path.isfile(os.path.join(d, "__init__.py"))
            ),
            None,
        )
        return init, True
    for i, part in enumerate(rest):
        last = i == len(rest) - 1
        package_dirs = [
            os.path.join(d, part) for d in dirs if os.path.isdir(os.path.join(d, part))
        ]
        if last:
            for d in package_dirs:
                init = os.path.join(d, "__init__.py")
                if os.path.isfile(init):
                    return init, True
            for d in dirs:
                module_file = os.path.join(d, part + ".py")
                if os.path.isfile(module_file):
                    return module_file, False
            return (None, True) if package_dirs else None
        dirs = package_dirs
        if not dirs:
            return None
    return None


def _resolve_relative(current: str, is_package: bool, level: int, target: str) -> str:
    base = current if is_package else current.rpartition(".")[0]
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    return f"{base}.{target}" if target else base


def _imports_of(
    name: str, source: bytes, is_package: bool, roots: Sequence[str]
) -> set[str]:
    """Module names *source* imports, read from its AST.

    ``from pkg import x`` names the submodule ``pkg.x`` when there is one, else
    ``pkg`` itself, whose ``__init__`` then defines ``x``.
    """
    with warnings.catch_warnings():
        # Vendored code under `sieval.community` carries invalid escapes.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                module = _resolve_relative(
                    name, is_package, node.level, node.module or ""
                )
            else:
                module = node.module or ""
            if not module or not _followed(module, roots):
                continue
            for alias in node.names:
                submodule = f"{module}.{alias.name}"
                found.add(submodule if _locate(submodule) is not None else module)
    return found


def grader_modules(task_cls: type) -> dict[str, str]:
    """Digest the modules reachable from *task_cls*'s grading code.

    The walk follows :data:`GRADER_PACKAGES` and the ``sieval_*`` plugin
    package of each class in the MRO (see :func:`_own_root`). It starts at the
    module of every class in the MRO and follows imports by a static AST walk
    over source files found by path, so nothing is imported or run. Returns
    ``{module: "sha256:…"}`` over each module's source bytes.
    """
    own = {
        root
        for klass in task_cls.__mro__
        if (root := _own_root(klass.__module__)) is not None
    }
    roots = (*GRADER_PACKAGES, *sorted(own - set(GRADER_PACKAGES)))
    pending = [
        klass.__module__
        for klass in task_cls.__mro__
        if _followed(klass.__module__, roots)
    ]
    seen: set[str] = set()
    modules: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if name in seen or not _followed(name, roots):
            continue
        seen.add(name)
        located = _locate(name)
        if located is None:
            continue
        path, is_package = located
        if path is None:
            continue
        with open(path, "rb") as f:
            source = f.read()
        modules[name] = "sha256:" + hashlib.sha256(source).hexdigest()
        pending.extend(_imports_of(name, source, is_package, roots))
    return dict(sorted(modules.items()))


def _parse_judge_model_ref(raw: str) -> str | dict[str, str] | None:
    """Parse :data:`JUDGE_MODEL_REF_ENV`: ``None`` unset, a plain ref, or by role.

    A value starting with ``{`` or ``[`` is meant as JSON, so it must parse as
    an object of role to string; anything else is one plain reference.
    """
    raw = raw.strip()
    if not raw:
        return None
    if not raw.startswith(("{", "[")):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{JUDGE_MODEL_REF_ENV} is not valid JSON ({exc}); set a plain "
            'reference, or a JSON object such as {"grader": "<ref>"}'
        ) from None
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise ValueError(
            f"{JUDGE_MODEL_REF_ENV} must be a plain reference or a JSON object "
            f"mapping each judge role to a string, got {raw!r}"
        )
    return parsed


def validate_judge_model_ref(environ: Mapping[str, str] | None = None) -> None:
    """Fail on a malformed :data:`JUDGE_MODEL_REF_ENV` before any work starts.

    Raises ``ValueError`` naming the variable. Called when a run starts, so a
    bad value stops it rather than leaving every grader identity unknown.
    """
    _parse_judge_model_ref(
        (os.environ if environ is None else environ).get(JUDGE_MODEL_REF_ENV, "")
    )


def judge_model_refs(
    roles: Iterable[str], environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Read :data:`JUDGE_MODEL_REF_ENV` into ``{role: ref}`` for *roles*."""
    parsed = _parse_judge_model_ref(
        (os.environ if environ is None else environ).get(JUDGE_MODEL_REF_ENV, "")
    )
    if parsed is None:
        return {}
    if isinstance(parsed, str):
        return dict.fromkeys(roles, parsed)
    return {role: parsed[role] for role in roles if role in parsed}


def judge_model_id(model: str) -> str:
    """A judge's model id without Google's ``models/`` resource prefix.

    ``models/gemini-x`` and ``gemini-x`` name one model.
    """
    return model.removeprefix("models/")


def judge_identities(
    judges: Mapping[str, Model],
    prompts: Mapping[str, JSONValue],
    versions: Mapping[str, str | None] | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[JudgeIdentity]:
    """Describe each judge by model, version, sampling and prompt, never endpoint."""
    refs = judge_model_refs(judges, environ)
    out: list[JudgeIdentity] = []
    for role in sorted(judges):
        meta = judges[role].meta()
        sampling: dict[str, JSONValue] = dict(meta["default_params"])
        if "extra" in meta:
            sampling["extra"] = dict(meta["extra"])
        entry: JudgeIdentity = {
            "role": role,
            "model": judge_model_id(meta["model"]),
            "version": (versions or {}).get(role),
            "sampling": sampling,
            "prompt": digest(prompts[role]) if role in prompts else None,
        }
        if role in refs:
            entry["ref"] = refs[role]
        out.append(entry)
    return out


def grader_identity(
    modules: Mapping[str, str], judges: Sequence[JudgeIdentity]
) -> GraderIdentity:
    return {
        "digest": digest({"modules": dict(modules), "judges": list(judges)}),
        "modules": dict(modules),
        "judges": list(judges),
    }


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def empty_identity() -> RunIdentity:
    return {
        "schema": IDENTITY_SCHEMA,
        "data": None,
        "prompt": None,
        "inference": None,
        "grader": None,
    }


def carried_identity(existing: Mapping[str, Any] | None) -> RunIdentity:
    """The previous run's identity when it has this schema, else an empty one."""
    identity = empty_identity()
    previous = (existing or {}).get("identity")
    if isinstance(previous, dict) and previous.get("schema") == IDENTITY_SCHEMA:
        for stage in STAGES:
            if isinstance(previous.get(stage), dict):
                identity[stage] = previous[stage]  # type: ignore[literal-required]
    return identity


def carried_runs(existing: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The previous ``runs[]``; a ``meta.json`` older than the field has none."""
    runs = (existing or {}).get("runs")
    return [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []


def new_run_record(
    version: str,
    start: Literal["fresh", "resume", "from"],
    config: Sequence[str] = (),
    now: datetime | None = None,
) -> RunRecord:
    record: RunRecord = {
        "sieval": version,
        "at": now.isoformat() if now is not None else utc_now(),
        "start": start,
    }
    if config:
        record["config"] = list(config)
    return record


def utc_now() -> str:
    """The timestamp format of ``runs[].at`` and ``runs[].ended``."""
    return datetime.now(UTC).isoformat()


def ordered_stages(stages: Iterable[str]) -> list[str]:
    chosen = set(stages)
    return [stage for stage in STAGES if stage in chosen]
