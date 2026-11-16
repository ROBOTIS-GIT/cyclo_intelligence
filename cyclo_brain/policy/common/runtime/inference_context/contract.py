"""Resolved execution behavior, independent of policy names and tensor libraries."""

from dataclasses import dataclass
import json
import math

from .execution import ExecutionContext


@dataclass(frozen=True)
class ExecutionContract:
    mode: str = "chunk"
    initial_action_timeout_s: float | None = None
    observation_warmup_timeout_s: float | None = None
    pending_command_count: int | None = None
    feedback_schema: int = 1
    request_after: str = "prediction_success"
    request_after_count: int = 1

    def __post_init__(self):
        if self.feedback_schema not in {1, 2} or type(self.feedback_schema) is not int:
            raise ValueError("unsupported feedback schema")
        if self.request_after not in {"prediction_success", "plan_accepted", "first_publication", "published_count", "plan_terminal"}:
            raise ValueError("unsupported request prerequisite")
        if type(self.request_after_count) is not int or not 1 <= self.request_after_count <= 4096:
            raise ValueError("request prerequisite count must be 1..4096")
        if self.request_after != "published_count" and self.request_after_count != 1:
            raise ValueError("request count is only meaningful for published_count")
        if self.request_after != "prediction_success" and self.feedback_schema != 2:
            raise ValueError("execution prerequisites require feedback schema 2")
        if self.pending_command_count is not None and (
            type(self.pending_command_count) is not int or not 0 <= self.pending_command_count <= 4096
        ):
            raise ValueError("pending command count must be 0..4096 or None for the full plan")
        if self.mode not in {"chunk", "step"}:
            raise ValueError(f"unsupported execution mode: {self.mode}")
        if self.initial_action_timeout_s is not None and (
            (self.mode != "step" and self.observation_warmup_timeout_s is None)
            or type(self.initial_action_timeout_s) not in (int, float)
            or not math.isfinite(self.initial_action_timeout_s)
            or not 1 <= self.initial_action_timeout_s <= 120
        ):
            raise ValueError("initial action timeout requires step or temporal preparation and 1..120 seconds")
        if self.observation_warmup_timeout_s is not None and (
            type(self.observation_warmup_timeout_s) not in (int, float)
            or not math.isfinite(self.observation_warmup_timeout_s)
            or not 1 <= self.observation_warmup_timeout_s <= 120
        ):
            raise ValueError("observation warmup timeout must be 1..120 seconds")

    @property
    def is_step(self):
        return self.mode == "step"

    @property
    def requires_context(self):
        return (self.is_step or self.observation_warmup_timeout_s is not None
                or self.pending_command_count is not None or self.feedback_schema == 2)


@dataclass(frozen=True)
class LoadedExecution:
    """LOAD-only metadata; DESCRIBE capabilities keep their existing meaning.

    The Worker allocates a session identity after weights load. The Runtime owns
    every subsequent execution fact. Empty metadata preserves legacy chunk LOAD.
    """

    contract: ExecutionContract = ExecutionContract()
    context: ExecutionContext | None = None

    def __post_init__(self):
        if not isinstance(self.contract, ExecutionContract):
            raise ValueError("LOAD execution contract is invalid")
        if self.contract.requires_context and self.context is None:
            raise ValueError("contextual LOAD requires an execution session")
        if self.context is not None and (
            not isinstance(self.context, ExecutionContext)
            or self.context.phase != "ready"
            or self.context.actions or self.context.planning or self.context.resets
            or self.context.after_event_id or self.context.latest_event_id
        ):
            raise ValueError("LOAD execution session must be empty and ready")
        if self.context is not None and self.context.feedback_schema != self.contract.feedback_schema:
            raise ValueError("LOAD feedback schema mismatch")

    def to_json(self):
        if self.context is None and not self.contract.is_step:
            return ""
        contract = {"mode": self.contract.mode}
        if self.contract.feedback_schema == 2:
            contract.update(feedback_schema=2, request_after=self.contract.request_after,
                            request_after_count=self.contract.request_after_count)
        if self.contract.initial_action_timeout_s is not None:
            contract["initial_action_timeout_s"] = self.contract.initial_action_timeout_s
        if self.contract.observation_warmup_timeout_s is not None:
            contract["observation_warmup_timeout_s"] = self.contract.observation_warmup_timeout_s
        if self.contract.pending_command_count is not None:
            contract["pending_command_count"] = self.contract.pending_command_count
        return json.dumps({
            "execution_contract": contract,
            "execution_context": None if self.context is None else json.loads(self.context.to_json()),
        }, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw):
        if raw == "":
            return cls()
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 4096:
            raise ValueError("LOAD execution metadata exceeds 4096 bytes")
        try:
            data = json.loads(raw)
            if data == {}:
                return cls()
            if not isinstance(data, dict) or set(data) != {"execution_contract", "execution_context"}:
                raise ValueError("invalid LOAD execution metadata fields")
            contract = data["execution_contract"]
            if (not isinstance(contract, dict) or "mode" not in contract
                    or set(contract) - {"mode", "initial_action_timeout_s", "observation_warmup_timeout_s",
                                        "pending_command_count", "feedback_schema", "request_after", "request_after_count"}):
                raise ValueError("invalid LOAD execution contract fields")
            context = data["execution_context"]
            return cls(ExecutionContract(**contract),
                       None if context is None else ExecutionContext.from_json(json.dumps(context)))
        except (TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("invalid LOAD execution metadata") from exc
