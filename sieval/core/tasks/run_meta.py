"""Stage identities and run history recorded in a task directory's ``meta.json``.

``meta.json`` is the one record of where a task directory came from. Each run
rewrites it atomically, so this module only builds the document; the saver
writes it.

The six stage identities, one per pipeline stage, describe the run that
produced the current ``report.json``:

===============  ============================================================
``data``         Every selected sample, keyed by ``(sample_id, repeat_index)``
                 and digested over its whole raw content.
``preprocess``   Every sample's ``PromptRecord.prompt``, before the model's
                 chat template.
``inference``    The candidate's resolved request arguments, dialect and
                 engine, the task's own inference arguments (``n``), the
                 result-changing runner keys and the serving recipe, plus
                 ``requests``: every sample's request parameters as the dialect
                 sent them, so a ``stop`` or ``max_tokens`` a task passes at
                 call time counts. Only what defines a request counts there:
                 its body and whether it streamed, never the method, path,
                 query or model name. The model under test, the engine build
                 and the SiEval version are left out.
``postprocess``  The code ``postprocess`` reaches (:func:`stage_code`), the
                 task class and its ``grader_args()``.
``feedback``     The same for ``feedback``, plus every judge and service.
``report``       The same for ``report`` and ``problem_groups``.
===============  ============================================================

A change to one stage reruns it and every stage after it.

Identity schema 1 (0.10.0 to 0.11.x) named ``preprocess`` ``prompt`` and
folded the grading stages into one ``grader``. :func:`legacy_grader_keys`
recomputes it with the walk that wrote it: equal means nothing changed.

Every digest is ``sha256:`` over canonical JSON (sorted keys, no whitespace).
A stage whose input could not be read is ``None``: unknown, never a guess. A
per-sample stage with one sample it could not digest is unknown as a whole,
and says how many in ``unknown``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import ast
import codecs
import hashlib
import importlib.util
import json
import os
import secrets
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal, NotRequired, TypedDict, cast

from loguru import logger

from sieval.core.models import REQUEST_ENVELOPE_HTTP, REQUEST_ENVELOPE_KEY, Model
from sieval.core.types import JSONValue

from .services import SERVICE_ENDPOINT_ENV, ServiceIdentity

#: Version of the task result directory a resume reads (shard records and their
#: ``.idx``, ``meta.json``, the ``report.json`` shape), written as ``meta.json``
#: ``result_schema_version``; a resume refuses any other. Bump it for a layout
#: change older directories cannot be read under, or for a core change that
#: alters results: the grader walk skips ``sieval.core``, so nothing else sees it.
RESULT_SCHEMA_VERSION = 1

#: Version of the ``identity`` block's layout. A digest computed differently is
#: not a layout change: it reruns one stage, a new schema would discard them all.
#: Schema 1 is still read (:func:`read_identity`).
IDENTITY_SCHEMA = 2

#: A per-sample digest of a value that was legitimately absent: the prompt of a
#: sample whose ``preprocess`` failed, the requests of one never inferred. Kept
#: apart from ``None``, which is a digest that could not be taken.
ABSENT = "absent"

#: Stage names in pipeline order; ``runs[].reran`` lists them in this order.
STAGES: tuple[str, ...] = (
    "data",
    "preprocess",
    "inference",
    "postprocess",
    "feedback",
    "report",
)

#: The stages that grade, each named by the Task method that runs it.
GRADING_STAGES: tuple[str, ...] = ("postprocess", "feedback", "report")

#: Schema-1 stage names that carry over to schema 2, and their new names. The
#: schema-1 ``grader`` covered all of :data:`GRADING_STAGES` and has none.
SCHEMA_1_STAGES: Mapping[str, str] = MappingProxyType(
    {"data": "data", "prompt": "preprocess", "inference": "inference"}
)

#: Package roots the grader walk follows. ``sieval.core`` and
#: ``sieval.datasets`` are left out on purpose: data has its own identity, and a
#: core change that alters results bumps :data:`RESULT_SCHEMA_VERSION`, checked
#: by the replay a release runs.
#: ``sieval_shared`` is the package sieval-plugins variants share.
GRADER_PACKAGES: tuple[str, ...] = (
    "sieval.tasks",
    "sieval.community",
    "sieval_plugins",
    "sieval_shared",
)

#: What computing a grading identity may raise: a walk reads files and parses
#: them, and a task hook may reject an argument. The identity is then unknown.
IDENTITY_ERRORS: tuple[type[Exception], ...] = (
    TypeError,
    ValueError,
    AttributeError,
    SyntaxError,
    OSError,
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
    """The identity-schema-1 ``grader`` block, which covered every grading stage.

    ``task`` is :func:`grader_task` and ``args`` the task's
    :meth:`~sieval.core.tasks.task.Task.grader_args`, both absent in a block
    written before they counted; ``services`` is present only for a task that
    declares one."""

    digest: str
    task: NotRequired[str]
    modules: dict[str, str]
    args: NotRequired[dict[str, JSONValue]]
    judges: list[JudgeIdentity]
    services: NotRequired[list[ServiceIdentity]]


class GradingIdentity(TypedDict):
    """One grading stage: ``postprocess``, ``feedback`` or ``report``.

    ``modules`` is that stage's code (:func:`stage_code`). ``feedback`` holds
    every judge and service; another stage holds the ones its code reads.
    ``services`` is present only for a task that declares one.
    """

    digest: str
    task: str
    modules: dict[str, str]
    args: dict[str, JSONValue]
    judges: NotRequired[list[JudgeIdentity]]
    services: NotRequired[list[ServiceIdentity]]


class RunIdentity(TypedDict):
    """The ``identity`` block. A stage is ``None`` while unknown.

    Two runs share a stage only when every digest in it matches: for
    ``inference`` that is ``digest`` *and* ``requests.digest``, and a ``None``
    digest on either side matches nothing.
    """

    schema: int
    data: DigestedStage | None
    preprocess: DigestedStage | None
    inference: InferenceIdentity | None
    postprocess: GradingIdentity | None
    feedback: GradingIdentity | None
    report: GradingIdentity | None


class LimiterPeak(TypedDict):
    """The most requests a limiter held at once, against its cap."""

    peak: int
    limit: int | None


class RequestCounts(TypedDict):
    """One phase x model's model requests in a run: calls, attempts, outcomes."""

    calls: int
    attempts: int
    outcomes: dict[str, int]
    in_flight: dict[str, LimiterPeak]


#: Request counts by phase, then model id.
type RunRequests = dict[str, dict[str, RequestCounts]]


class RunFrom(TypedDict):
    """``identity`` is the copied directory's block as it recorded it, of
    either schema, or an empty one when it recorded none."""

    dir: str
    identity: RunIdentity | dict[str, Any]


class ResumeOutcomes(TypedDict):
    """What a resume did with each recorded sample, as ``runs[].resume``.

    Keyed by the earliest stage that changed for it: ``kept`` (nothing, or
    only ``report``), ``regraded`` (``feedback``), ``repostprocessed``
    (``postprocess``), ``restarted`` (from ``preprocess``), ``dropped`` (no
    longer in the dataset). ``rereported`` is how many kept samples a changed
    ``report`` read.
    """

    kept: int
    rereported: int
    regraded: int
    repostprocessed: int
    restarted: int
    dropped: int


# Functional form: ``from`` is a keyword.
RunRecord = TypedDict(
    "RunRecord",
    {
        "id": NotRequired[str],
        "sieval": str,
        "at": str,
        "start": Literal["fresh", "resume", "from"],
        "ended": NotRequired[str],
        "reran": NotRequired[list[str]],
        "reported": NotRequired[bool],
        "interrupted": NotRequired[bool],
        "rules_hash": NotRequired[str],
        "config": NotRequired[list[str]],
        "model": NotRequired[str],
        "from": NotRequired[RunFrom],
        "requests": NotRequired[RunRequests],
        "resume": NotRequired[ResumeOutcomes],
    },
)
"""One ``runs[]`` entry, appended when a run starts.

``id`` (:func:`new_run_id`) sorts by start time. ``rules_hash`` is set when
the run wrote ``anomalies.json``.
While the run is in flight the entry has neither ``ended`` nor ``reran``.
A run that saves its ``report.json`` writes both: ``ended`` (ISO 8601 UTC, like
``at``) and ``reran``, the stages it executed, ``[]`` for a fast resume.
``reported: true`` says the run wrote ``report.json``, which a fast resume does
not. ``resume`` is :class:`ResumeOutcomes`. A run
stopped by an interrupt writes ``ended`` and ``interrupted: true``, and no
``reran``. An entry with no ``ended`` is a run that is still going, or was
killed or failed before it could say so. ``config`` is the config diff a resume
made, present when non-empty. ``model`` is this run's model under test, present
only when it differs from the directory's ``model``, which stays the one whose
samples are on disk. ``from`` names
the directory a resume across tasks copied, with that directory's identity when
it was copied (``None`` stages for a directory that recorded none).
``requests`` counts the run's model requests by phase and model; absent when
it made none or ``record_requests`` was off. It is the durable copy:
``requests.jsonl`` rotates, ``profile.json`` is rewritten by every run, and a
fast resume keeps ``report.json``'s ``sieval.requests`` from the run that
wrote it. Not part of any identity.
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


#: Keys of a flat request body that say how it was sent, or name the model,
#: rather than what was asked. ``stream_options`` only asks for usage. An HTTP
#: envelope's ``method``, ``path`` and ``query`` never reach the body.
_TRANSPORT_KEYS = frozenset({"model", "stream", "stream_options"})


def request_definition(params: Mapping[Any, object]) -> dict[str, object]:
    """What defines a request, out of a dialect's ``request_params``.

    Dialects record two shapes: a flat body (OpenAI, Anthropic, SGLang) and an
    HTTP envelope with ``body``, ``path`` and ``query`` (Google, whose path
    names the model and whose method and ``alt`` query say whether it
    streamed), marked by :data:`REQUEST_ENVELOPE_KEY`. Both reduce to
    ``{"body": …, "stream": bool}``, so neither the model under test nor the
    gateway prefix reaches the digest.
    """
    body = params.get("body")
    if params.get(REQUEST_ENVELOPE_KEY) == REQUEST_ENVELOPE_HTTP and isinstance(
        body, Mapping
    ):
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
    grader. Only ``sieval_*`` packages on disk count, namespace packages
    included, so a test module, a third-party base class or ``sieval`` itself
    (whose ``core`` and ``datasets`` stay out) never widens the walk.
    """
    top = module.partition(".")[0]
    if not top.startswith("sieval_") or not _root_dirs(top):
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


def _read_source(path: str) -> bytes:
    """A module's source bytes, as every walk here digests them."""
    with open(path, "rb") as f:
        return f.read()


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


def _patch_root(module: str) -> str | None:
    """The top-level package of *module* when it may patch a task class.

    A patch (``setattr(Task, "report", report)``) can live in a package of any
    name. Any package with source on disk counts, except ``sieval`` itself, the
    standard library and ``site-packages``.
    """
    top = module.partition(".")[0]
    if top in ("sieval", "builtins") or top in sys.stdlib_module_names:
        return None
    if _followed(module):
        return None
    located = _locate(top)
    if located is None or located[0] is None:
        return None
    parts = os.path.normpath(located[0]).split(os.sep)
    if "site-packages" in parts or "dist-packages" in parts:
        return None
    return top


def _function_modules(task_cls: type) -> set[str]:
    """Modules defining the functions each MRO class holds, its own or patched in."""
    found: set[str] = set()
    for klass in task_cls.__mro__:
        for value in vars(klass).values():
            func = getattr(value, "__func__", None) or getattr(value, "fget", None)
            func = func or value
            module = getattr(func, "__module__", None)
            if callable(func) and isinstance(module, str) and module:
                found.add(module)
    return found


def _package_files(top: str) -> list[str]:
    """Every module under top-level package *top*, in each of its locations.

    A package may load modules with ``importlib``, which no import statement
    shows, so the walk digests it whole.
    """
    found: list[str] = []
    for _, rel, filenames in _package_dirs(top):
        prefix = top if rel == "." else f"{top}.{rel.replace(os.sep, '.')}"
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            stem = filename[:-3]
            found.append(prefix if stem == "__init__" else f"{prefix}.{stem}")
    return found


def _package_dirs(top: str) -> Iterable[tuple[str, str, list[str]]]:
    """``(root, dir relative to root, file names)`` for each directory of *top*."""
    for root in _root_dirs(top):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith((".", "__pycache__"))
            )
            yield root, os.path.relpath(dirpath, root), filenames


def _absent_inits(top: str) -> dict[str, str]:
    """``{file key: empty digest}`` for each directory of *top* lacking ``__init__.py``.

    sieval-plugins' variant rewrite adds an empty one to each, so a renamed
    copy keys the same only if a missing one reads as empty.
    """
    empty = "sha256:" + hashlib.sha256(b"").hexdigest()
    return {
        "/".join(
            (OWN_PACKAGE, *([] if rel == "." else rel.split(os.sep)), "__init__.py")
        ): empty
        for _, rel, filenames in _package_dirs(top)
        if "__init__.py" not in filenames
    }


#: File-key label of the task class's own plugin package, whatever its name.
OWN_PACKAGE = "."


def _file_key(name: str, path: str, own: str | None) -> str:
    """``<label>/<path in package>`` for module *name* at *path*.

    The label is the top-level package's name, or :data:`OWN_PACKAGE` when
    that is *own*, so a plugin package copied under another name (as a runner
    merging several into one process does) keys the same.
    """
    top = name.partition(".")[0]
    label = OWN_PACKAGE if top == own else top
    for root in _root_dirs(top):
        rel = os.path.relpath(path, root)
        if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
            return "/".join((label, *rel.split(os.sep)))
    raise ValueError(f"{path} is outside package {top}")


#: The task's own plugin package as its imports are digested; no import spells it.
SELF_IMPORT = "<package>"


def _names_own(module: str | None, own: str) -> bool:
    return module is not None and (module == own or module.startswith(own + "."))


def _self_import_text(node: ast.Import | ast.ImportFrom, own: str) -> str | None:
    """*node* with *own* spelt :data:`SELF_IMPORT`, or ``None`` if it names none.

    An unaliased ``import own.x`` also binds ``own``, so it reads as
    ``import <package>.x, <package> as own``, the form sieval-plugins' variant
    rewrite writes; a following ``own as n`` is taken as that binding. A
    hand-written ``import own.x, own as n`` therefore still digests per name.
    """
    moved = len(own)
    if isinstance(node, ast.ImportFrom):
        if node.level or not _names_own(node.module, own):
            return None
        module = SELF_IMPORT + cast(str, node.module)[moved:]
        return ast.unparse(ast.ImportFrom(module=module, names=node.names, level=0))
    if not any(_names_own(alias.name, own) for alias in node.names):
        return None
    names: list[ast.alias] = []
    for i, alias in enumerate(node.names):
        if not _names_own(alias.name, own):
            names.append(alias)
            continue
        target = SELF_IMPORT + alias.name[moved:]
        if alias.asname or alias.name == own:
            names.append(ast.alias(name=target, asname=alias.asname or own))
            continue
        names.append(ast.alias(name=target))
        following = node.names[i + 1] if i + 1 < len(node.names) else None
        if following is None or following.name != own or not following.asname:
            names.append(ast.alias(name=SELF_IMPORT, asname=own))
    return ast.unparse(ast.Import(names=names))


def _own_name_lines(source: bytes, own: str) -> list[bytes]:
    """*source*'s lines with each absolute import of *own* as
    :func:`_self_import_text`, line for line: a rewritten span's later lines
    are empty, so the AST of *source* still indexes them.

    Nothing outside those spans changes; comments inside one drop out, as in
    the variant rewrite. A file that is not UTF-8 or does not parse is
    returned unchanged.
    """
    lines = source.splitlines(keepends=True)
    bom = codecs.BOM_UTF8 if source.startswith(codecs.BOM_UTF8) else b""
    body = source[len(bom) :]
    try:
        # Decoded first, so a coding cookie cannot move the UTF-8 byte offsets.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(body.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return lines
    edits = [
        (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, text)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (text := _self_import_text(node, own)) is not None
    ]
    if not edits:
        return lines
    out = body.splitlines(keepends=True)
    for row, col, end_row, end_col, text in sorted(edits, reverse=True):
        tail = out[cast(int, end_row) - 1][cast(int, end_col) :]
        out[row - 1] = out[row - 1][:col] + text.encode("utf-8") + tail
        for i in range(row, cast(int, end_row)):
            out[i] = b""
    out[0] = bom + out[0]
    return out


def _without_own_name(source: bytes, own: str) -> bytes:
    """*source* as :func:`grader_modules` digests a file of the plugin *own*."""
    return b"".join(_own_name_lines(source, own))


def _own_package(task_cls: type) -> str | None:
    """The plugin package of the first class in *task_cls*'s MRO that has one."""
    for klass in task_cls.__mro__:
        if (root := _own_root(klass.__module__)) is not None:
            return root
    return None


def grader_task(task_cls: type) -> str:
    """``<file key>:<qualname>``, or the import path for a class with no source."""
    module = task_cls.__module__
    located = _locate(module)
    if located is None or located[0] is None:
        return f"{module}.{task_cls.__qualname__}"
    key = _file_key(module, located[0], _own_package(task_cls))
    return f"{key}:{task_cls.__qualname__}"


def grader_modules(task_cls: type) -> dict[str, str]:
    """Digest the modules reachable from *task_cls*'s grading code.

    The walk starts at the module of every class in the MRO and of every
    function they hold, so a patched-in method counts. It follows
    :data:`GRADER_PACKAGES`, each class's plugin package (:func:`_own_root`)
    and each patch's package (:func:`_patch_root`) by a static AST walk, so
    nothing is imported or run. A package outside ``sieval`` that the walk
    reaches is digested whole (:func:`_package_files`); under ``sieval`` only
    what is reached counts. Returns ``{file key: "sha256:…"}`` over each
    module's source bytes (:func:`_file_key`). So that a renamed copy of the
    task's own plugin package digests the same, its imports of itself are
    digested without its name (:func:`_without_own_name`) and a missing
    ``__init__.py`` as an empty one (:func:`_absent_inits`).
    """
    starts = {klass.__module__ for klass in task_cls.__mro__}
    own = {root for module in starts if (root := _own_root(module)) is not None}
    patched = {
        root
        for module in _function_modules(task_cls) - starts
        if (root := _patch_root(module)) is not None
    }
    extra = sorted((own | patched) - set(GRADER_PACKAGES))
    roots = (*GRADER_PACKAGES, *extra)
    own_package = _own_package(task_cls)
    pending = [m for m in starts | _function_modules(task_cls) if _followed(m, roots)]
    seen: set[str] = set()
    modules: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if name in seen or not _followed(name, roots):
            continue
        seen.add(name)
        top = name.partition(".")[0]
        if top != "sieval":
            pending.extend(_package_files(top) if name == top else (top,))
        located = _locate(name)
        if located is None:
            continue
        path, is_package = located
        if path is None:
            continue
        source = _read_source(path)
        key = _file_key(name, path, own_package)
        hashed = _without_own_name(source, top) if top == own_package else source
        modules[key] = "sha256:" + hashlib.sha256(hashed).hexdigest()
        pending.extend(_imports_of(name, source, is_package, roots))
    if own_package is not None:
        modules = _absent_inits(own_package) | modules
    return dict(sorted(modules.items()))


def grader_modules_v0_10_0(task_cls: type) -> dict[str, str]:
    """:func:`grader_modules` as SiEval 0.10.0 computed it, keyed by import name.

    Kept unchanged so a 0.10.0 directory's ``grader`` identity can still be
    compared (:func:`legacy_grader_keys`). 0.10.1 keyed files by path instead.
    """
    starts = {klass.__module__ for klass in task_cls.__mro__}
    own = {root for module in starts if (root := _own_root_v0_10_0(module)) is not None}
    patched = {
        root
        for module in _function_modules(task_cls) - starts
        if (root := _patch_root(module)) is not None
    }
    extra = sorted((own | patched) - set(GRADER_PACKAGES))
    roots = (*GRADER_PACKAGES, *extra)
    pending = [m for m in starts | _function_modules(task_cls) if _followed(m, roots)]
    pending.extend(extra)
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
        source = _read_source(path)
        modules[name] = "sha256:" + hashlib.sha256(source).hexdigest()
        pending.extend(_imports_of(name, source, is_package, roots))
        if is_package and name not in GRADER_PACKAGES:
            pending.extend(_package_files_v0_10_0(name, path))
    return dict(sorted(modules.items()))


def _own_root_v0_10_0(module: str) -> str | None:
    """:func:`_own_root` as 0.10.0 had it: a package with an ``__init__.py``."""
    top = module.partition(".")[0]
    if not top.startswith("sieval_"):
        return None
    located = _locate(top)
    if located is None or located[0] is None:
        return None
    return top


def _package_files_v0_10_0(name: str, init: str) -> list[str]:
    """:func:`_package_files` as 0.10.0 had it: one location, from its ``__init__``."""
    root = os.path.dirname(init)
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith((".", "__pycache__"))
        )
        rel = os.path.relpath(dirpath, root)
        prefix = name if rel == "." else f"{name}.{rel.replace(os.sep, '.')}"
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            stem = filename[:-3]
            found.append(prefix if stem == "__init__" else f"{prefix}.{stem}")
    return found


def grader_task_v0_10_0(task_cls: type) -> str:
    """:func:`grader_task` as 0.10.0 had it: the class's import path."""
    return f"{task_cls.__module__}.{task_cls.__qualname__}"


# ---------------------------------------------------------------------------
# Stage code: which code each stage reaches
# ---------------------------------------------------------------------------

#: The Task methods each code-identified stage starts from.
STAGE_ENTRY_METHODS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "preprocess": ("preprocess",),
        "postprocess": ("postprocess",),
        "feedback": ("feedback",),
        "report": ("report", "problem_groups"),
    }
)

#: Methods every code-identified stage starts from as well: they build the
#: state a stage reads (``self._*``) and the sample (``ctx.raw_sample``).
SHARED_ENTRY_METHODS: tuple[str, ...] = (
    "__init__",
    "__post_init__",
    "setup",
    "make_context",
    "grader_args",
    "judge_models",
    "judge_prompts",
)

#: Builtins that run, or look up, code no name in the source shows.
_DYNAMIC_NAMES = frozenset({"eval", "exec", "globals", "locals", "vars", "__import__"})

#: Attributes that load or look up code no name in the source shows.
_DYNAMIC_ATTRS = frozenset(
    {
        "import_module",
        "spec_from_file_location",
        "exec_module",
        "run_path",
        "run_module",
        "__dict__",
        "__getattribute__",
    }
)

#: The dynamic names that may load a module by name. A module digested whole
#: holds no task method a lookup could reach, so only these matter there.
_LOADERS = frozenset(
    {"__import__", "import_module", "spec_from_file_location", "exec_module"}
    | {"run_path", "run_module", "exec", "eval"}
)

#: Attribute lookups that name the attribute in an argument.
_ATTR_CALLS = frozenset({"getattr", "setattr", "delattr", "hasattr"})


class _Unattributable(Exception):
    """A construct the static per-method walk cannot attribute to a stage."""


def _body_span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
    """A function body's lines; ``(0, -1)`` when it shares the ``def`` line."""
    start = node.body[0].lineno
    if start <= node.lineno:
        return (0, -1)
    return (start, cast(int, node.end_lineno))


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """What runs when the ``def`` itself runs: decorators, defaults, annotations."""
    out: list[ast.AST] = [*node.decorator_list, node.args]
    if node.returns is not None:
        out.append(node.returns)
    return out


type _Bindings = dict[str, list[tuple[str, str | None]]]


def _bindings(
    roots: Iterable[ast.AST], name: str, is_package: bool, *, nested: bool = True
) -> _Bindings:
    """``{bound name: [(module, attribute or None)]}`` for the imports in *roots*.

    A name may bind several targets (``try``/``except ImportError``, or two
    ``import a.x``/``import a.y`` binding ``a``). A star import binds
    ``*<module>``. *nested* ``False`` leaves out imports inside a ``def``,
    which bind only there.
    """
    out: _Bindings = {}

    def bind(key: str, target: tuple[str, str | None]) -> None:
        if target not in (targets := out.setdefault(key, [])):
            targets.append(target)

    stack = list(roots)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            for alias in node.names:
                bind(alias.asname or alias.name.partition(".")[0], (alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            module = (
                _resolve_relative(name, is_package, node.level, node.module or "")
                if node.level
                else node.module or ""
            )
            for alias in node.names:
                if alias.name == "*":
                    bind(f"*{module}", (module, None))
                else:
                    bind(alias.asname or alias.name, (module, alias.name))
        if nested or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(ast.iter_child_nodes(node))
    return out


class _Module:
    """A module the stage walk reads one function at a time.

    ``units`` holds each top-level function and each method of a top-level
    class, by qualname. ``shell`` is the source without their bodies: imports,
    decorators, signatures, class attributes and every other statement that
    runs at import, which counts in every stage reaching the module.
    """

    def __init__(
        self, name: str, path: str, source: bytes, is_package: bool, own: bool
    ):
        self.name = name
        self.path = path
        self.is_package = is_package
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            self.tree = ast.parse(source)
        self.units: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.top: dict[str, ast.AST] = {}
        shell_nodes: list[ast.AST] = []
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.top[node.name] = node
                self.units[node.name] = node
                shell_nodes.extend(_signature(node))
            elif isinstance(node, ast.ClassDef):
                self.top[node.name] = node
                shell_nodes.extend([*node.decorator_list, *node.bases, *node.keywords])
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.units[f"{node.name}.{item.name}"] = item
                        shell_nodes.extend(_signature(item))
                    else:
                        shell_nodes.append(item)
            elif not isinstance(node, (ast.Import, ast.ImportFrom)):
                # A top-level import binds; what reads the name reaches it.
                shell_nodes.append(node)
                for target in _assigned_names(node):
                    self.top.setdefault(target, node)
        self.shell_nodes = shell_nodes
        # Parsed as written; digested as grader_modules digests the file.
        top = name.partition(".")[0]
        lines = (
            _own_name_lines(source, top) if own else source.splitlines(keepends=True)
        )
        self._lines = lines
        stripped: set[int] = set()
        for unit in self.units.values():
            start, end = _body_span(unit)
            stripped.update(range(start, end + 1))
        self.shell = b"".join(
            line for i, line in enumerate(lines, start=1) if i not in stripped
        )
        #: Module-scope imports; a unit's own are :meth:`unit_bindings`.
        self.bindings = _bindings(self.tree.body, name, is_package, nested=False)

    def unit_bindings(self, qualname: str) -> _Bindings:
        return _bindings([self.units[qualname]], self.name, self.is_package)

    def unit_source(self, qualname: str) -> bytes:
        unit = self.units[qualname]
        start, end = _body_span(unit)
        return b"".join(self._lines[start - 1 : end]) if start else b""


def _assigned_names(node: ast.AST) -> list[str]:
    """Module-level names *node* binds outside a ``def`` or ``class``."""
    names: list[str] = []
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    elif isinstance(node, ast.TypeAlias):
        targets = [node.name]
    for target in targets:
        names.extend(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return names


class _Refs:
    """The bare names and attribute names some code reads."""

    def __init__(self, nodes: Iterable[ast.AST]):
        self.names: set[str] = set()
        self.attrs: set[str] = set()
        #: Attributes read rather than assigned.
        self.loaded: set[str] = set()
        self.dynamic: list[str] = []
        #: Objects of a ``getattr`` with a computed name: a bare name, or
        #: ``None`` for any other expression.
        self.computed: list[str | None] = []
        #: ``(attribute, attributes read, strings)`` for each ``<x>.<attr> =
        #: <value>``: what the stored value was built from.
        self.stores: list[tuple[str, frozenset[str], frozenset[str]]] = []
        #: String constants outside such a stored value: an endpoint variable
        #: read where it is used.
        self.strings: set[str] = set()
        stored: set[int] = set()
        for root in nodes:
            for node in ast.walk(root):
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value:
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    value = list(ast.walk(node.value))
                    reads = frozenset(
                        n.attr for n in value if isinstance(n, ast.Attribute)
                    )
                    texts = frozenset(
                        n.value
                        for n in value
                        if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    )
                    for target in targets:
                        if isinstance(target, ast.Attribute):
                            self.stores.append((target.attr, reads, texts))
                            stored.update(id(n) for n in value)
        for root in nodes:
            for node in ast.walk(root):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in stored
                ):
                    self.strings.add(node.value)
                if isinstance(node, ast.Name):
                    self.names.add(node.id)
                    if node.id in _DYNAMIC_NAMES:
                        self.dynamic.append(node.id)
                elif isinstance(node, ast.Attribute):
                    self.attrs.add(node.attr)
                    if isinstance(node.ctx, ast.Load):
                        self.loaded.add(node.attr)
                    if node.attr in _DYNAMIC_ATTRS:
                        self.dynamic.append(node.attr)
                elif isinstance(node, ast.Call):
                    self._attr_call(node)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        self.names.add(alias.asname or alias.name.partition(".")[0])

    def _attr_call(self, node: ast.Call) -> None:
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None
        if name not in _ATTR_CALLS or len(node.args) < 2:
            return
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
            self.attrs.add(attr.value)
            if name != "setattr":
                self.loaded.add(attr.value)
            return
        target = node.args[0]
        if isinstance(target, ast.Attribute) and target.attr != "__class__":
            # State the task holds (``self._dataset``), never the task itself.
            return
        if isinstance(target, ast.Name) and target.id not in ("self", "cls"):
            # A module bound by an import is checked by the caller; any other
            # name may hold the task or one of its classes.
            self.computed.append(target.id)
            return
        self.computed.append(None)


def _methods_of(value: object) -> list[object]:
    """The functions behind a class attribute: a method, a property's accessors."""
    if isinstance(value, property):
        return [f for f in (value.fget, value.fset, value.fdel) if f is not None]
    inner = getattr(value, "__func__", None)
    if callable(inner):
        return [inner]
    wrapped = getattr(value, "func", None)  # functools.partial / partialmethod
    if callable(wrapped) and not isinstance(value, type):
        return [wrapped]
    return [value]


class _StageWalker:
    """Attribute a task class's code to the stages that reach it.

    A module holding a class of the MRO, or a method patched onto one, is
    read a function at a time (:class:`_Module`); any other is digested whole,
    as :func:`grader_modules` does.
    """

    def __init__(self, task_cls: type):
        self.task_cls = task_cls
        starts = {klass.__module__ for klass in task_cls.__mro__}
        own = {root for module in starts if (root := _own_root(module)) is not None}
        patched = {
            root
            for module in _function_modules(task_cls) - starts
            if (root := _patch_root(module)) is not None
        }
        extra = sorted((own | patched) - set(GRADER_PACKAGES))
        self.roots: tuple[str, ...] = (*GRADER_PACKAGES, *extra)
        self.own_package = _own_package(task_cls)
        self.mro_classes = {
            (klass.__module__, klass.__qualname__) for klass in task_cls.__mro__
        }
        self._modules: dict[str, _Module | None] = {}
        #: ``{attribute: [(module, unit)]}`` over every class of the MRO.
        self.methods: dict[str, list[tuple[str, str]]] = {}
        for klass in task_cls.__mro__:
            for attr, value in vars(klass).items():
                for func in _methods_of(value):
                    self._index(klass, attr, func)
            self._index_source(klass)
        #: Modules holding a class of the MRO: their shells hold class
        #: attributes any stage may read, so every stage reaches them.
        self.class_modules = {m for m in starts if _followed(m, self.roots)}
        #: Modules read a function at a time.
        self.split = self.class_modules | {
            m for units in self.methods.values() for m, _ in units
        }

    def _index(self, klass: type, attr: str, func: object) -> None:
        if isinstance(func, type) or not callable(func):
            return
        module = getattr(func, "__module__", None)
        if not isinstance(module, str) or not _followed(module, self.roots):
            return
        qualname = getattr(func, "__qualname__", None)
        unit = qualname.partition(".<locals>")[0] if isinstance(qualname, str) else ""
        parsed = self.module(module)
        if parsed is None or unit not in parsed.units:
            raise _Unattributable(
                f"{klass.__qualname__}.{attr} is not a function in {module}'s source"
            )
        self.methods.setdefault(attr, []).append((module, unit))

    def unreached(self) -> dict[str, dict[str, str]]:
        """Files of each package outside ``sieval`` that no stage reaches.

        They count in every stage that reaches any of the package: a stage
        may still run one (read by path, say) and nothing tells which.
        """
        reached: set[str] = set()
        tops: set[str] = set()
        for entries in STAGE_ENTRY_METHODS.values():
            try:
                walk = self._walk(entries)
            except _Unattributable:
                # That stage counts everything (its fallback): it reaches no
                # file in particular, so leaves none out of the others.
                continue
            for module in (*walk.shells, *walk.whole):
                reached.add(module)
                tops.add(module.partition(".")[0])
        tops.discard("sieval")
        out: dict[str, dict[str, str]] = {}
        for top in sorted(tops):
            files: dict[str, str] = {}
            for module in (top, *_package_files(top)):
                if module in reached or module in self.split:
                    continue
                located = _locate(module)
                if located is None or located[0] is None:
                    continue
                source = self._hashed(module, _read_source(located[0]))
                files[self._key(module, located[0])] = (
                    "sha256:" + hashlib.sha256(source).hexdigest()
                )
            if top == self.own_package:
                files = _absent_inits(top) | files
            out[top] = files
        return out

    def _index_source(self, klass: type) -> None:
        """Index the methods *klass*'s source defines, patched over or not:
        a patch may wrap the one it replaces."""
        module = klass.__module__
        if not _followed(module, self.roots):
            return
        parsed = self.module(module)
        if parsed is None:
            return
        prefix = klass.__qualname__ + "."
        for qualname in parsed.units:
            if qualname.startswith(prefix):
                attr = qualname[len(prefix) :]
                units = self.methods.setdefault(attr, [])
                if (module, qualname) not in units:
                    units.append((module, qualname))

    def module(self, name: str) -> _Module | None:
        if name not in self._modules:
            located = _locate(name)
            path, is_package = located if located is not None else (None, False)
            if path is None:
                self._modules[name] = None
            else:
                own = name.partition(".")[0] == self.own_package
                self._modules[name] = _Module(
                    name, path, _read_source(path), is_package, own
                )
        return self._modules[name]

    def _hashed(self, name: str, source: bytes) -> bytes:
        top = name.partition(".")[0]
        return _without_own_name(source, top) if top == self.own_package else source

    def _key(self, name: str, path: str) -> str:
        return _file_key(name, path, self.own_package)

    def stage(
        self, entries: Iterable[str]
    ) -> tuple[dict[str, str], "_StageWalk", set[str]]:
        """What *entries* reach: ``({key: digest}, the walk, top-level
        packages outside sieval reached)``.

        A function's key is ``<file key>:<qualname>``; a module shell's or a
        whole module's is its file key. Raises :class:`_Unattributable` for
        code that reaches other code by a name only known at run time.
        """
        walk = self._walk(entries)
        tops = {m.partition(".")[0] for m in (*walk.shells, *walk.whole)}
        tops.discard("sieval")
        return walk.digests(), walk, tops

    def _walk(self, entries: Iterable[str]) -> "_StageWalk":
        walk = _StageWalk(self)
        # Dunder methods are reached by protocol, not by name.
        dunders = [a for a in self.methods if a.startswith("__") and a.endswith("__")]
        for entry in (*entries, *SHARED_ENTRY_METHODS, *dunders):
            walk.reach_attr(entry)
        for module in sorted(self.class_modules):
            walk.reach_shell(module)
        walk.run()
        return walk


class _StageWalk:
    """One stage's walk over a :class:`_StageWalker`'s modules."""

    def __init__(self, walker: _StageWalker):
        self.w = walker
        self.units: set[tuple[str, str]] = set()
        self.shells: set[str] = set()
        self.whole: set[str] = set()
        self.attrs: set[str] = set()
        self.loaded: set[str] = set()
        self.stores: list[tuple[str, frozenset[str], frozenset[str]]] = []
        self.strings: set[str] = set()
        self._pending: list[tuple[str, str]] = []
        self._pending_shells: list[str] = []
        self._pending_whole: list[str] = []
        self._bound: set[tuple[str, str | None]] = set()

    def reach_attr(self, attr: str) -> None:
        if attr not in self.attrs:
            self.attrs.add(attr)
            self._pending.extend(self.w.methods.get(attr, ()))

    def reach_shell(self, module: str) -> None:
        if module not in self.shells:
            self.shells.add(module)
            self._pending_shells.append(module)

    def reach_module(self, module: str, attr: str | None) -> None:
        """Reach what an import binds: a split module's unit, or a whole module.

        A name a split module imports rather than defines (a re-export) is
        followed to where it is defined.
        """
        if not _followed(module, self.w.roots) or (module, attr) in self._bound:
            return
        self._bound.add((module, attr))
        if attr is not None and _locate(f"{module}.{attr}") is not None:
            module, attr = f"{module}.{attr}", None
        if module not in self.w.split:
            if module not in self.whole:
                self.whole.add(module)
                self._pending_whole.append(module)
            return
        self.reach_shell(module)
        parsed = self.w.module(module)
        if parsed is None:
            return
        if attr is None:
            # The module object itself: any of its names may be read off it.
            for name in parsed.top:
                self.reach_name(parsed, name)
            self.reach_bound(parsed.bindings, parsed.bindings)
        elif attr in parsed.top:
            self.reach_name(parsed, attr)
        else:
            self.reach_bound(parsed.bindings, [attr])

    def reach_bound(self, bindings: _Bindings, names: Iterable[str]) -> None:
        for name in names:
            for module, attr in bindings.get(name, ()):
                self.reach_module(module, attr)

    def reach_name(self, parsed: _Module, name: str) -> None:
        """Reach a top-level name of *parsed*: a function, or a helper class."""
        node = parsed.top.get(name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._pending.append((parsed.name, name))
        elif isinstance(node, ast.ClassDef):
            if (parsed.name, name) in self.w.mro_classes:
                return  # the task's own class: its methods are reached by name
            prefix = name + "."
            self._pending.extend(
                (parsed.name, q) for q in parsed.units if q.startswith(prefix)
            )

    def read(self, parsed: _Module, refs: _Refs, where: str, local: _Bindings) -> None:
        """Follow what *refs* read in *parsed*. *local* holds the reading
        unit's own imports: they run whenever it does, so all are reached, and
        a name they bind resolves to them as well as to the module's."""
        dynamic = list(refs.dynamic)
        for target in refs.computed:
            bound = [
                *(local.get(target, ()) if target is not None else ()),
                *(parsed.bindings.get(target, ()) if target is not None else ()),
            ]
            # Only a module outside the walk is safe to look a name up on.
            if not bound or any(_followed(m, self.w.roots) for m, _ in bound):
                dynamic.append(f"getattr({target or 'self'}, <expr>)")
        if dynamic:
            raise _Unattributable(
                f"{where} reaches code by a name only known at run time "
                f"({', '.join(sorted(set(dynamic)))})"
            )
        for attr in refs.attrs:
            self.reach_attr(attr)
        self.loaded |= refs.loaded
        self.stores.extend(refs.stores)
        self.strings |= refs.strings
        self.reach_bound(local, local)
        for name in refs.names:
            if name in parsed.top:
                self.reach_name(parsed, name)
            self.reach_bound(parsed.bindings, [name])

    def run(self) -> None:
        while self._pending or self._pending_shells or self._pending_whole:
            if self._pending_shells:
                module = self._pending_shells.pop()
                parsed = self.w.module(module)
                if parsed is None:
                    continue
                self.read(parsed, _Refs(parsed.shell_nodes), f"{module} at import", {})
                # A star import, or one whose names the module never reads,
                # runs for its side effects: it counts wherever the module does.
                read = _Refs([parsed.tree]).names
                self.reach_bound(
                    parsed.bindings,
                    [b for b in parsed.bindings if b.startswith("*") or b not in read],
                )
            elif self._pending:
                module, qualname = self._pending.pop()
                if (module, qualname) in self.units:
                    continue
                self.units.add((module, qualname))
                self.reach_shell(module)
                parsed = self.w.module(module)
                if parsed is not None:
                    self.read(
                        parsed,
                        _Refs([parsed.units[qualname]]),
                        f"{module}.{qualname}",
                        parsed.unit_bindings(qualname),
                    )
            else:
                self._whole_step(self._pending_whole.pop())

    def _whole_step(self, name: str) -> None:
        """Follow a whole module's imports; a split module it imports is reached
        by the names imported from it."""
        path, is_package = _locate(name) or (None, False)
        if path is None:
            return
        tree_source = _read_source(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(tree_source)
        loads = sorted({d for d in _Refs([tree]).dynamic if d in _LOADERS})
        if loads and name.partition(".")[0] != "sieval":
            # It may load any file of its package, which no import shows and
            # another stage may reach. Under `sieval` the walk follows imports
            # only, as `grader_modules` does.
            raise _Unattributable(f"{name} loads modules by name ({', '.join(loads)})")
        for module in _imports_of(name, tree_source, is_package, self.w.roots):
            if module in self.w.split:
                parsed = self.w.module(module)
                if parsed is None:
                    continue
                bound = [
                    t for ts in _bindings([tree], name, is_package).values() for t in ts
                ]
                attrs = [a for m, a in bound if m == module and a]
                if not attrs or len(attrs) != sum(1 for m, _ in bound if m == module):
                    self.reach_module(module, None)
                for attr in attrs:
                    self.reach_module(module, attr)
            else:
                self.reach_module(module, None)

    def reads(self, attrs: Iterable[str], strings: Iterable[str] = ()) -> bool:
        """Whether this stage's code reads one of *attrs*, a string of
        *strings*, or an attribute it stored from either (``self._client =
        self._grader.client``), followed to a fixpoint."""
        derived = set(attrs)
        texts = set(strings)
        grew = True
        while grew:
            grew = False
            for target, reads, consts in self.stores:
                if target not in derived and (reads & derived or consts & texts):
                    derived.add(target)
                    grew = True
        return bool(self.loaded & derived) or bool(self.strings & texts)

    def digests(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for module in self.shells:
            parsed = self.w.module(module)
            if parsed is not None:
                key = self.w._key(module, parsed.path)
                out[key] = "sha256:" + hashlib.sha256(parsed.shell).hexdigest()
        for module, qualname in self.units:
            parsed = self.w.module(module)
            if parsed is not None:
                key = f"{self.w._key(module, parsed.path)}:{qualname}"
                source = parsed.unit_source(qualname)
                out[key] = "sha256:" + hashlib.sha256(source).hexdigest()
        for module in self.whole:
            located = _locate(module)
            if located is None or located[0] is None:
                continue
            source = self.w._hashed(module, _read_source(located[0]))
            out[self.w._key(module, located[0])] = (
                "sha256:" + hashlib.sha256(source).hexdigest()
            )
        return dict(sorted(out.items()))


class StageCode(TypedDict):
    """The code one stage reaches (:func:`stage_code`).

    ``modules`` is ``{key: digest}``: a function or method as
    ``<file key>:<qualname>``, a module's shell or a whole module as its file
    key. ``judges`` are the judge roles the stage's code reads; ``services``
    whether it reads a declared service's endpoint. ``fallback`` says why the
    stage counts all of the task's grading code, every judge and service
    instead.
    """

    modules: dict[str, str]
    judges: list[str]
    services: bool
    fallback: NotRequired[str]


def stage_code(
    task_cls: type,
    judge_roles: Iterable[str] = (),
    services: Iterable[str] = (),
) -> dict[str, StageCode]:
    """The code each stage of *task_cls* reaches, by stage.

    Each stage starts from its :data:`STAGE_ENTRY_METHODS` and the
    :data:`SHARED_ENTRY_METHODS` and follows names statically: a method by
    attribute name across the MRO, a module function or helper class by bare
    name, an import to what it binds (through re-exports; a method's own
    imports resolve within it). Code two stages reach counts in both.

    A stage reads a judge when its code reads ``self._<role>``/``self.<role>``
    or an attribute stored from one (every judge, if the task overrides
    ``judge_models``), and a service when it reads its endpoint variable name.

    A construct the walk cannot follow (``getattr`` on the task with a
    computed name, ``eval``, ``importlib``, ``vars()``, a method with no
    source) makes that grading stage and every later one count all of
    :func:`grader_modules`, every judge and service; ``fallback`` names it.
    ``preprocess`` falls back on its own: its prompts are digested per sample.
    """
    roles = sorted(set(judge_roles))
    markers = {SERVICE_ENDPOINT_ENV[s] for s in services if s in SERVICE_ENDPOINT_ENV}
    unknown_services = any(s not in SERVICE_ENDPOINT_ENV for s in services)
    has_services = bool(markers) or unknown_services
    out: dict[str, StageCode] = {}
    try:
        walker: _StageWalker | None = _StageWalker(task_cls)
        reason: str | None = None
    except _Unattributable as exc:
        walker, reason = None, str(exc)
    own_judges = walker is not None and "judge_models" in walker.methods
    unreached = walker.unreached() if walker is not None else {}
    whole: dict[str, str] | None = None
    grading_reason = reason
    for stage in ("preprocess", *GRADING_STAGES):
        why = reason if stage == "preprocess" else grading_reason
        if walker is not None and why is None:
            try:
                modules, walk, tops = walker.stage(STAGE_ENTRY_METHODS[stage])
            except _Unattributable as exc:
                why = str(exc)
            else:
                for top in sorted(tops):
                    modules = unreached.get(top, {}) | modules
                judges = [r for r in roles if own_judges or walk.reads((r, f"_{r}"))]
                out[stage] = {
                    "modules": dict(sorted(modules.items())),
                    "judges": judges,
                    "services": has_services
                    and (unknown_services or walk.reads((), markers)),
                }
                continue
        if stage != "preprocess":
            grading_reason = why
        if whole is None:
            whole = grader_modules(task_cls)
        out[stage] = {
            "modules": dict(whole),
            "judges": roles,
            "services": has_services,
            "fallback": cast(str, why),
        }
    return out


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


def warn_unmatched_judge_refs(
    roles: Iterable[str], environ: Mapping[str, str] | None = None
) -> list[str]:
    """Warn about :data:`JUDGE_MODEL_REF_ENV` roles none of *roles* match.

    Such a reference is never recorded, most likely a typo. Returns them.
    """
    parsed = _parse_judge_model_ref(
        (os.environ if environ is None else environ).get(JUDGE_MODEL_REF_ENV, "")
    )
    if not isinstance(parsed, dict):
        return []
    known = set(roles)
    unmatched = sorted(set(parsed) - known)
    if unmatched:
        logger.warning(
            "{} names judge role(s) {} that no task in this run has (roles: {}); "
            "those references are not recorded",
            JUDGE_MODEL_REF_ENV,
            ", ".join(unmatched),
            ", ".join(sorted(known)) or "none",
        )
    return unmatched


#: Google's resource prefixes for a model id: a base model and a tuned one.
_GOOGLE_MODEL_PREFIXES = ("models/", "tunedModels/")


def judge_model_id(model: str) -> str:
    """A judge's model id without Google's resource prefix.

    ``models/gemini-x`` and ``gemini-x`` name one model, as do
    ``tunedModels/my-judge`` and ``my-judge``.
    """
    for prefix in _GOOGLE_MODEL_PREFIXES:
        if model.startswith(prefix):
            return model.removeprefix(prefix)
    return model


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
        # Capability settings (``reasoning.budget_tokens``) reach the request
        # through the runtime plan, not ``default_params``.
        plan = judges[role].runtime_plan
        if plan is not None and plan.request_defaults.values:
            sampling["request_defaults"] = plan.request_defaults.to_json_value()
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
    modules: Mapping[str, str],
    judges: Sequence[JudgeIdentity],
    services: Sequence[ServiceIdentity] = (),
    args: Mapping[str, JSONValue] | None = None,
    task: str | None = None,
) -> GraderIdentity:
    """Build the grader identity.

    *task* is the task class's import path: two task classes in one module
    share every module digest and may still grade differently.
    """
    identity: GraderIdentity = {
        "digest": "",
        "modules": dict(modules),
        "args": dict(args or {}),
        "judges": list(judges),
    }
    if task is not None:
        identity["task"] = task
    if services:
        identity["services"] = list(services)
    identity["digest"] = digest({k: v for k, v in identity.items() if k != "digest"})
    return identity


def grader_key(grader: Mapping[str, Any] | None) -> str | None:
    """What a resume compares of a grader identity: all but what a judge reported.

    Every field except ``digest`` counts, so a field added to the grader
    identity later counts without a change here. A judge's ``version`` is what
    its API answered once called, so a run that has not called it yet cannot
    know it, and ``ref`` names where the judge was served; both are recorded,
    neither decides a regrade. A block written before ``args`` existed reads
    as empty ``args``. ``None`` when *grader* is unknown or malformed.
    """
    if not isinstance(grader, Mapping):
        return None
    if not isinstance(grader.get("modules"), Mapping) or not isinstance(
        grader.get("judges"), list
    ):
        return None
    payload = {k: v for k, v in grader.items() if k != "digest"}
    payload.setdefault("args", {})
    payload["judges"] = [
        {k: v for k, v in j.items() if k not in ("version", "ref")}
        for j in grader["judges"]
        if isinstance(j, Mapping)
    ]
    try:
        return digest(payload)
    except (TypeError, ValueError):
        return None


def grading_identities(
    code: Mapping[str, StageCode],
    judges: Sequence[JudgeIdentity],
    services: Sequence[ServiceIdentity],
    args: Mapping[str, JSONValue],
    task: str,
) -> dict[str, GradingIdentity]:
    """Build the ``postprocess``, ``feedback`` and ``report`` identities.

    Each holds its stage's code (*code*, from :func:`stage_code`), the task
    class and :meth:`~sieval.core.tasks.task.Task.grader_args`. ``feedback``
    holds every judge and service; another stage holds those its code reads
    (agieval's extractor, were it called in ``postprocess``).
    """
    out: dict[str, GradingIdentity] = {}
    for stage in GRADING_STAGES:
        stage_code_ = code[stage]
        identity: GradingIdentity = {
            "digest": "",
            "task": task,
            "modules": dict(stage_code_["modules"]),
            "args": dict(args),
        }
        feedback = stage == "feedback"
        read = set(stage_code_["judges"])
        mine = [j for j in judges if feedback or j["role"] in read]
        if mine or feedback:
            identity["judges"] = mine
        if services and (feedback or stage_code_["services"]):
            identity["services"] = list(services)
        identity["digest"] = digest(
            {k: v for k, v in identity.items() if k != "digest"}
        )
        out[stage] = identity
    return out


def stage_key(stage: Mapping[str, Any] | None) -> str | None:
    """What a resume compares of a grading identity: all but what a judge reported.

    Every field except ``digest`` counts, so a field added later counts
    without a change here. A judge's ``version`` is what its API answered once
    called, so a run that has not called it yet cannot know it, and ``ref``
    names where the judge was served; both are recorded, neither decides a
    regrade. ``None`` when *stage* is unknown or malformed.
    """
    if not isinstance(stage, Mapping) or not isinstance(stage.get("modules"), Mapping):
        return None
    payload = {k: v for k, v in stage.items() if k != "digest"}
    if "judges" in stage:
        if not isinstance(stage["judges"], list):
            return None
        payload["judges"] = [
            {k: v for k, v in j.items() if k not in ("version", "ref")}
            for j in stage["judges"]
            if isinstance(j, Mapping)
        ]
    try:
        return digest(payload)
    except (TypeError, ValueError):
        return None


def legacy_grader_keys(
    task_cls: type,
    judges: Sequence[JudgeIdentity],
    services: Sequence[ServiceIdentity],
    args: Mapping[str, JSONValue],
) -> set[str]:
    """:func:`grader_key` of the schema-1 ``grader`` each release would write.

    0.10.0 keyed modules and the task by import name; 0.10.1 to 0.11.x by file
    path (:func:`grader_modules`). A recorded ``grader`` whose key is one of
    these was written for this task's grading code as it is now. A walk that
    fails is left out, so it matches nothing.
    """
    keys: set[str] = set()
    for walk, task in (
        (grader_modules, grader_task),
        (grader_modules_v0_10_0, grader_task_v0_10_0),
    ):
        try:
            identity = grader_identity(
                walk(task_cls), judges, services, args=args, task=task(task_cls)
            )
        except IDENTITY_ERRORS:
            continue
        key = grader_key(identity)
        if key is not None:
            keys.add(key)
    return keys


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def empty_identity() -> RunIdentity:
    return {
        "schema": IDENTITY_SCHEMA,
        "data": None,
        "preprocess": None,
        "inference": None,
        "postprocess": None,
        "feedback": None,
        "report": None,
    }


def read_identity(
    block: object,
) -> tuple[RunIdentity, Mapping[str, Any] | None] | None:
    """A recorded ``identity`` block as schema 2, with a schema-1 ``grader``.

    Schema 2 is read as written. Schema 1 maps ``data`` and ``inference``
    across and ``prompt`` to ``preprocess`` (:data:`SCHEMA_1_STAGES`); its
    ``grader`` cannot be split, so the grading stages read as unknown and the
    ``grader`` is returned beside them, for the caller to compare whole
    (:func:`legacy_grader_keys`). ``None`` for any other schema.
    """
    if not isinstance(block, Mapping):
        return None
    block = cast(Mapping[str, Any], block)
    out = empty_identity()
    schema = block.get("schema")
    if schema == IDENTITY_SCHEMA:
        for stage in STAGES:
            if isinstance(block.get(stage), dict):
                out[stage] = block[stage]  # type: ignore[literal-required]
        return out, None
    if schema == 1:
        for old, new in SCHEMA_1_STAGES.items():
            if isinstance(block.get(old), dict):
                out[new] = block[old]  # type: ignore[literal-required]
        grader = block.get("grader")
        return out, grader if isinstance(grader, Mapping) else None
    return None


def carried_identity(existing: Mapping[str, Any] | None) -> RunIdentity:
    """The previous run's identity as schema 2 (:func:`read_identity`), else empty.

    A schema-1 block's grading stages are unknown until the run writes them.
    """
    read = read_identity((existing or {}).get("identity"))
    return read[0] if read is not None else empty_identity()


def carried_runs(existing: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The previous ``runs[]``; a ``meta.json`` older than the field has none."""
    runs = (existing or {}).get("runs")
    return [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []


def new_run_id(now: datetime | None = None) -> str:
    """A run's ID: its UTC start time, then random hex, so IDs sort by start."""
    moment = now if now is not None else datetime.now(UTC)
    return f"{moment.astimezone(UTC):%Y%m%dT%H%M%S.%fZ}-{secrets.token_hex(4)}"


def new_run_record(
    version: str,
    start: Literal["fresh", "resume", "from"],
    config: Sequence[str] = (),
    now: datetime | None = None,
) -> RunRecord:
    moment = now if now is not None else datetime.now(UTC)
    record: RunRecord = {
        "id": new_run_id(moment),
        "sieval": version,
        "at": moment.isoformat(),
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
