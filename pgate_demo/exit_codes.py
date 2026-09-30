"""Exit-code contract for Placement Gate.

Command execution status and domain verdict are separate. A governed refusal is
a successful evaluation:

    pgate plan <big-model>   -> EXCEEDS_RESOURCE_BUDGET  -> exit 0
    pgate plan --no-server   -> server unreachable       -> exit 3
    pgate place (no server)  -> cannot execute a live op -> exit 3
    pgate hardware (broken)  -> capability unavailable    -> exit 2
    pgate plan --bogus       -> malformed invocation     -> exit 64

The distinction matters here more than in a typical tool: the whole point of the
demo is that refusing is the system working. A refusal reported as a crash would
invert the lesson.
"""

from __future__ import annotations

from dataclasses import dataclass

EXIT_OK = 0
EXIT_CAPABILITY_UNAVAILABLE = 2
EXIT_SERVER_UNAVAILABLE = 3
EXIT_ASSERTION_FAILED = 4
EXIT_BAD_INVOCATION = 64


@dataclass(frozen=True)
class ExitContract:
    command: str
    required: tuple[str, ...]
    success: str
    failure: str
    needs_server: str = "no"
    notes: str = ""

    def render(self) -> str:
        lines = [
            f"{self.command}",
            f"    upstream     : {', '.join(self.required) or '(none)'}",
            f"    needs server : {self.needs_server}",
            f"    exit 0 when  : {self.success}",
            f"    exit != 0    : {self.failure}",
        ]
        if self.notes:
            lines.append(f"    note         : {self.notes}")
        return "\n".join(lines)


CONTRACTS: dict[str, ExitContract] = {
    "hardware": ExitContract(
        command="hardware",
        required=("hardware_observer", "hardware_memory", "hardware_nvidia"),
        success="observed hardware and reported it",
        failure="a hardware capability was unavailable",
        needs_server="no",
        notes="works with no Ollama server: the facts come from nvidia-smi and ctypes",
    ),
    "census": ExitContract(
        command="census",
        required=("chat_adapter",),
        success="listed the local service's models",
        failure="the service was unreachable or the adapter unavailable",
        needs_server="yes",
        notes="reports only the metadata placement needs: tag, size, context, digest",
    ),
    "plan": ExitContract(
        command="plan",
        required=("placement_planner", "model_profile", "hardware_observer"),
        success="a plan was computed, including a refused plan",
        failure="a capability was unavailable, or model facts were needed but unreachable",
        needs_server="yes for model facts, no with --size/--parameters",
        notes="EXCEEDS_RESOURCE_BUDGET and HARDWARE_UNKNOWN are domain verdicts and exit 0",
    ),
    "place": ExitContract(
        command="place",
        required=("controller", "placement_planner", "chat_adapter", "ledger"),
        success=(
            "place_and_load_model() ran, including a legitimate verification failure"
        ),
        failure="a capability was unavailable or the server was unreachable",
        needs_server="yes",
        notes=(
            "mutates Ollama residency. REFUSED at verification is a successful "
            "evaluation and exits 0; it does not unload on your behalf"
        ),
    ),
    "selftest": ExitContract(
        command="selftest",
        required=("placement_planner", "model_profile"),
        success="every expected assertion held",
        failure="an assertion failed, or a capability was unavailable",
        needs_server="only with --live",
        notes=(
            "the one command whose domain result IS its exit condition: it "
            "asserts expected behavior rather than reporting it"
        ),
    ),
}

EXIT_MEANINGS: dict[int, str] = {
    EXIT_OK: "the requested evaluation completed",
    EXIT_CAPABILITY_UNAVAILABLE: "a required upstream capability was unavailable",
    EXIT_SERVER_UNAVAILABLE: "the Ollama service was unreachable",
    EXIT_ASSERTION_FAILED: "an expected behavioral assertion did not hold",
    EXIT_BAD_INVOCATION: "the invocation was malformed",
}

# Domain outcomes that a reader will meet and that deserve an explicit note.
# They are all exit 0: the question was asked and answered.
DOMAIN_STATUSES: dict[str, str] = {
    "ADMITTED": "the model was admitted and loaded, and the placement verified",
    "REFUSED": "upstream refused the placement at the planning stage",
    "EXCEEDS_RESOURCE_BUDGET": "the model needs more than the observed budget",
    "HARDWARE_UNKNOWN": "hardware evidence was absent or contradictory",
    "VERIFICATION_FAILED": (
        "the model loaded but the observed evidence does not match the "
        "requested placement. Fail-closed, and the model is left loaded"
    ),
    "NOT_FOUND": (
        "no model with that exact tag on this service, or the name is "
        "ambiguous across several. A name is not an identity, so this is "
        "reported rather than resolved"
    ),
    "UNAVAILABLE": "a required capability was missing; no result was substituted",
    "SERVER_UNREACHABLE": "the model service could not be asked",
}


def contract_for(command: str) -> ExitContract | None:
    return CONTRACTS.get(command)


def render_contracts() -> str:
    blocks = ["exit-code contract:"]
    for name in sorted(CONTRACTS):
        blocks.append("")
        blocks.append(CONTRACTS[name].render())
    blocks.append("")
    blocks.append("exit codes:")
    for code in sorted(EXIT_MEANINGS):
        blocks.append(f"  {code:>3d}  {EXIT_MEANINGS[code]}")
    blocks.append("")
    blocks.append("domain outcomes (all of these are exit 0 -- the question was answered):")
    for name, meaning in DOMAIN_STATUSES.items():
        first, _, rest = meaning.partition(". ")
        blocks.append(f"  {name:<26s}{first}.")
        if rest:
            blocks.append(f"  {'':<26s}{rest}")
    blocks.append("")
    blocks.append(
        "execution status is separate from domain verdict: a governed refusal is "
        "a successful evaluation and exits 0"
    )
    return "\n".join(blocks)
