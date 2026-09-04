"""Shared doubles for the MultiPL-E task family.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import build_judgement_record, build_rollout_judgement
from sieval.datasets.multipl_e_humaneval import MultiPLEHumanEvalDataset
from sieval.datasets.multipl_e_mbpp import MultiPLEMbppDataset
from tests.conftest import HandlerTransport


class CapturingGenModel(GenModel):
    def __init__(self, text: str = "  return 1;"):
        self.last_req: Request | None = None
        self._text = text
        super().__init__(model="mock-gen", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_completions")

    async def _stub_arun(self, req: Request) -> Response:
        self.last_req = req
        return Response(texts=(self._text,) * req.sampling.n)


class CapturingChatModel(ChatModel):
    def __init__(self, text: str = "  return 1;"):
        self.last_req: Request | None = None
        self._text = text
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self.last_req = req
        return Response(texts=(self._text,) * req.sampling.n)


class EvaluatorDouble:
    """Stands in for the code-eval service, recording what was asked of it.

    ``verdicts`` maps a `lang` to the boolean the service should answer with;
    anything unlisted passes. ``languages`` is what ``GET /languages``
    advertises, and ``languages=None`` makes the endpoint answer 404 — the
    shape of an evaluator that predates the capability probe.
    """

    def __init__(
        self,
        *,
        languages=("cpp", "javascript", "bash", "perl"),
        verdicts=None,
        msgs=None,
    ):
        self.bodies: list[dict] = []
        self.deadlines: list[float | None] = []
        self.language_calls: list[str] = []
        self._languages = languages
        self._verdicts = verdicts or {}
        self._msgs = msgs or {}

    async def get(self, url, *, timeout):
        del timeout
        self.language_calls.append(url)
        if self._languages is None:
            return _Response(404, {})
        payload = {"status": True, "msg": "", "data": list(self._languages)}
        return _Response(200, payload)

    async def post(self, url, *, json, timeout):
        del url
        self.bodies.append(json)
        self.deadlines.append(timeout)
        lang = json["lang"]
        ok = self._verdicts.get(lang, True)
        return _Response(
            200,
            {
                "status": ok,
                "msg": self._msgs.get(lang, "" if ok else "failed [exit 1]: boom"),
                "data": {
                    "avg_cpu_percent": 0.0,
                    "peak_cpu_percent": 0.0,
                    "avg_memory_mb": 0.0,
                    "peak_memory_mb": 0.0,
                    "n_cases": 1,
                    "n_passed": int(ok),
                },
            },
        )

    async def aclose(self) -> None:
        return None


class UnreachableEvaluator(EvaluatorDouble):
    async def get(self, url, *, timeout):
        del url, timeout
        raise OSError("connection refused")


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def row(
    language: str = "cpp",
    *,
    name: str = "HumanEval_23_strlen",
    prompt: str = "long f(std::string s) {\n",
    tests: str = '}\nint main() { assert(f("x") == 1); }',
    stop_tokens: tuple[str, ...] = ("\n}",),
) -> dict:
    return {
        "name": name,
        "language": language,
        "prompt": prompt,
        "doctests": "transform",
        "original": "/cluster/path/HumanEval_23_strlen.py",
        "prompt_terminology": "reworded",
        "tests": tests,
        "stop_tokens": list(stop_tokens),
    }


def dataset_for(rows, *, suite: str = "humaneval"):
    cls = MultiPLEHumanEvalDataset if suite == "humaneval" else MultiPLEMbppDataset
    return cls(_hf_dict=HFDatasetDict({"test": HFDataset.from_list(rows)}))


def judgement(language: str, *verdicts: tuple[bool, str]):
    """A JudgementRecord in the shape report() reads."""
    return build_judgement_record(
        None,
        [
            build_rollout_judgement(
                i, correct, extra={"msg": msg, "language": language}
            )
            for i, (correct, msg) in enumerate(verdicts)
        ],
        extra={"name": "n", "language": language},
    )
