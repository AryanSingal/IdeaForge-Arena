"""IdeaForge Arena - standalone script version of the Colab notebook.

Usage:
    pip install "groq>=0.20.0" "google-genai>=1.16.0" "gradio>=5.20.0"
    export GROQ_API_KEY=...   GEMINI_API_KEY=...
    python ideaforge_arena.py            # launches the Gradio research room
    python ideaforge_arena.py "problem"  # headless run, prints debate + saves blueprint
"""

#@title 3. Core engine: retries, JSON repair, safe calculator, Groq + Gemini clients
import ast, json, math, os, random, re, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Retry helper
# ----------------------------------------------------------------------------
FATAL_HINTS = (
    "invalid api key", "invalid_api_key", "unauthorized", "api key not valid",
    "permission denied", "api_key_invalid", "401",
)

def retry_call(fn, *args, attempts: int = 4, base_delay: float = 1.5,
               label: str = "llm call", **kwargs):
    """Call fn with exponential backoff. Aborts early on auth errors."""
    last_err = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            low = str(e).lower()
            if any(h in low for h in FATAL_HINTS):
                break
            if i == attempts - 1:
                break
            time.sleep(base_delay * (2 ** i) + random.random() * 0.6)
    raise RuntimeError(f"{label} failed: {last_err}")

# ----------------------------------------------------------------------------
# Text hygiene: never surface raw chain-of-thought
# ----------------------------------------------------------------------------
_THINK_RE = re.compile(r"<(think|thinking|reasoning|scratchpad)>.*?</\1>", re.S | re.I)
_OPEN_THINK_RE = re.compile(r"<(think|thinking|reasoning|scratchpad)>.*$", re.S | re.I)

def sanitize(text: Optional[str]) -> str:
    if not text:
        return ""
    t = _THINK_RE.sub("", text)
    t = _OPEN_THINK_RE.sub("", t)
    return t.strip()

def clip(text: Any, n: int = 1200) -> str:
    s = "" if text is None else str(text)
    s = s.strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def bullets(items, n: int = 8, prefix: str = "- ") -> str:
    out = []
    for it in (items or [])[:n]:
        if isinstance(it, dict):
            it = "; ".join(f"{k}: {clip(v, 200)}" for k, v in it.items() if v not in (None, "", [], {}))
        out.append(prefix + clip(it, 400))
    return "\n".join(out) if out else prefix + "(none)"

# ----------------------------------------------------------------------------
# Robust JSON extraction (models sometimes wrap JSON in prose or fences)
# ----------------------------------------------------------------------------
def extract_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    t = sanitize(text)
    t = re.sub(r"^\s*```(?:json|JSON)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    t = t.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = t[start:i + 1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            try:
                                obj = json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
                                if isinstance(obj, dict):
                                    return obj
                            except Exception:
                                pass
                        break
        start = t.find("{", start + 1)
    return None

def as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, (str, int, float)):
        return [v]
    if isinstance(v, dict):
        return [v]
    return list(v)

def as_num(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        try:
            m = re.search(r"-?\d+(?:\.\d+)?", str(v))
            return float(m.group()) if m else default
        except Exception:
            return default

# ----------------------------------------------------------------------------
# Safe numeric evaluator - real arithmetic ground truth for agent claims
# ----------------------------------------------------------------------------
_SAFE_FUNCS = {n: getattr(math, n) for n in (
    "sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh "
    "floor ceil fabs pow hypot degrees radians erf gamma".split()
)}
_SAFE_FUNCS.update({"abs": abs, "min": min, "max": max, "round": round, "sum": sum})
_SAFE_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.Tuple, ast.List, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.IfExp, ast.Compare,
    ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq,
)

def safe_eval(expression: str, variables: Optional[Dict[str, Any]] = None) -> float:
    """Evaluate an arithmetic expression with math functions only. No builtins."""
    expr = str(expression).strip()
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"[$€£₹]", "", expr)
    expr = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", expr)          # 1,250,000 -> 1250000
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\w.(])", r"(\1/100)", expr)  # 12% -> (12/100)
    if "=" in expr and "==" not in expr and "<=" not in expr and ">=" not in expr:
        expr = expr.split("=", 1)[1].strip()
    if len(expr) > 600:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")
    env: Dict[str, Any] = dict(_SAFE_CONSTS)
    for k, v in (variables or {}).items():
        env[str(k)] = as_num(v)
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
                raise ValueError("disallowed function call")
            if node.keywords:
                raise ValueError("keyword args not allowed")
        if isinstance(node, ast.Name) and node.id not in env and node.id not in _SAFE_FUNCS:
            raise ValueError(f"unknown symbol '{node.id}'")
    result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}, **_SAFE_FUNCS}, env)  # noqa: S307
    return float(result)

def execute_computations(comps: list, max_items: int = 6) -> list:
    """Run agent-proposed calculations and return verified numbers or errors."""
    out = []
    for c in as_list(comps)[:max_items]:
        if not isinstance(c, dict):
            c = {"name": "calc", "expression": str(c)}
        name = clip(c.get("name") or "calculation", 80)
        expr = str(c.get("expression") or "").strip()
        variables = c.get("variables") if isinstance(c.get("variables"), dict) else {}
        rec = {"name": name, "expression": expr, "variables": variables,
               "unit": clip(c.get("unit") or c.get("units") or "", 30),
               "why": clip(c.get("why") or c.get("purpose") or "", 200)}
        if not expr:
            rec["error"] = "empty expression"
        else:
            try:
                val = safe_eval(expr, variables)
                rec["value"] = round(val, 6) if abs(val) < 1e12 else val
            except Exception as e:  # noqa: BLE001
                rec["error"] = clip(str(e), 160)
        out.append(rec)
    return out

def computations_report(results: list) -> str:
    if not results:
        return "- (no calculations submitted this round)"
    lines = []
    for r in results:
        head = f"- **{r['name']}**: `{r['expression']}`"
        if r.get("variables"):
            head += f" with {json.dumps(r['variables'])[:220]}"
        if "value" in r:
            head += f" = **{r['value']}** {r.get('unit','')}".rstrip()
        else:
            head += f" -> EXECUTION ERROR: {r.get('error')}"
        lines.append(head)
    return "\n".join(lines)

# ----------------------------------------------------------------------------
# Groq client (fast debate / reasoning)
# ----------------------------------------------------------------------------
GROQ_PREFERRED = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
_GROQ_BLOCK = ("whisper", "tts", "guard", "embed", "vision-preview", "prompt-guard")

class GroqLLM:
    def __init__(self, api_key: str, model: Optional[str] = None):
        from groq import Groq
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY")
        self.client = Groq(api_key=api_key, timeout=120.0, max_retries=0)
        self.available = self._list_models()
        self.model = model if model in self.available else self._pick_default()

    def _list_models(self) -> List[str]:
        try:
            data = self.client.models.list()
            ids = [m.id for m in getattr(data, "data", []) if getattr(m, "id", None)]
            ids = [i for i in ids if not any(b in i.lower() for b in _GROQ_BLOCK)]
            return sorted(set(ids)) or list(GROQ_PREFERRED)
        except Exception:
            return list(GROQ_PREFERRED)

    def _pick_default(self) -> str:
        for m in GROQ_PREFERRED:
            if m in self.available:
                return m
        return self.available[0]

    def _raw(self, messages, temperature, max_tokens, force_json):
        kw = dict(model=self.model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens, top_p=0.95)
        if force_json:
            kw["response_format"] = {"type": "json_object"}
        try:
            r = self.client.chat.completions.create(**kw)
        except Exception as e:
            if force_json:  # model may not support JSON mode -> retry plain
                kw.pop("response_format", None)
                r = self.client.chat.completions.create(**kw)
            else:
                raise e
        return sanitize(r.choices[0].message.content or "")

    def complete(self, system: str, user: str, temperature: float = 0.6,
                 max_tokens: int = 2400, force_json: bool = False,
                 label: str = "groq") -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return retry_call(self._raw, msgs, temperature, max_tokens, force_json,
                          attempts=4, label=f"{label}/{self.model}")

    def complete_json(self, system: str, user: str, temperature: float = 0.55,
                      max_tokens: int = 2400, label: str = "groq-json") -> Tuple[Optional[dict], str]:
        txt = self.complete(system, user, temperature, max_tokens, force_json=True, label=label)
        data = extract_json(txt)
        if data is None:
            txt2 = self.complete(
                system,
                user + "\n\nYour previous reply was not valid JSON. Reply again with ONE valid "
                       "JSON object only, no prose, no markdown fences.",
                temperature=0.2, max_tokens=max_tokens, force_json=True, label=label + "-repair")
            data = extract_json(txt2)
            txt = txt2 if data else txt
        return data, txt

# ----------------------------------------------------------------------------
# Gemini client (independent verification / grounded research)
# ----------------------------------------------------------------------------
GEMINI_PREFERRED = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-001",
]

class GeminiLLM:
    def __init__(self, api_key: str, model: Optional[str] = None):
        from google import genai
        from google.genai import types
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.available = self._list_models()
        self.model = model if model in self.available else self._pick_default()
        self._no_thinking_ok = True

    def _list_models(self) -> List[str]:
        try:
            names = []
            for m in self.client.models.list():
                name = (getattr(m, "name", "") or "").replace("models/", "")
                acts = getattr(m, "supported_actions", None) or []
                if not name or "embedding" in name or "imagen" in name or "veo" in name:
                    continue
                if acts and "generateContent" not in acts:
                    continue
                if name.startswith("gemini"):
                    names.append(name)
            return sorted(set(names)) or list(GEMINI_PREFERRED)
        except Exception:
            return list(GEMINI_PREFERRED)

    def _pick_default(self) -> str:
        for m in GEMINI_PREFERRED:
            if m in self.available:
                return m
        return self.available[0]

    @staticmethod
    def _text_of(resp) -> str:
        parts_text = []
        try:
            for cand in (resp.candidates or []):
                content = getattr(cand, "content", None)
                for p in (getattr(content, "parts", None) or []):
                    if getattr(p, "text", None):
                        parts_text.append(p.text)
        except Exception:
            pass
        if not parts_text:
            try:
                parts_text = [resp.text or ""]
            except Exception:
                parts_text = [""]
        return sanitize("\n".join(parts_text))

    @staticmethod
    def _citations_of(resp) -> List[Dict[str, str]]:
        out, seen = [], set()
        try:
            for cand in (resp.candidates or []):
                gm = getattr(cand, "grounding_metadata", None)
                for ch in (getattr(gm, "grounding_chunks", None) or []):
                    web = getattr(ch, "web", None)
                    uri = getattr(web, "uri", None)
                    if uri and uri not in seen:
                        seen.add(uri)
                        out.append({"title": getattr(web, "title", None) or uri, "uri": uri})
        except Exception:
            pass
        return out

    def _cfg(self, system, temperature, max_tokens, search, no_thinking):
        t = self.types
        kw: Dict[str, Any] = dict(temperature=temperature, max_output_tokens=max_tokens)
        if system:
            kw["system_instruction"] = system
        if search:
            kw["tools"] = [t.Tool(google_search=t.GoogleSearch())]
        if no_thinking and "2.5" in self.model and "pro" not in self.model:
            try:
                kw["thinking_config"] = t.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        return t.GenerateContentConfig(**kw)

    def _raw(self, prompt, system, temperature, max_tokens, search, no_thinking):
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=self._cfg(system, temperature, max_tokens, search, no_thinking))
        return self._text_of(resp), self._citations_of(resp)

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.35,
                 max_tokens: int = 4096, search: bool = False,
                 label: str = "gemini") -> Tuple[str, List[Dict[str, str]]]:
        try:
            return retry_call(self._raw, prompt, system, temperature, max_tokens, search,
                              self._no_thinking_ok, attempts=3, label=f"{label}/{self.model}")
        except Exception as e:
            if self._no_thinking_ok:
                self._no_thinking_ok = False
                return retry_call(self._raw, prompt, system, temperature, max_tokens, search,
                                  False, attempts=2, label=f"{label}/{self.model}")
            raise e

    def generate_json(self, prompt: str, system: Optional[str] = None,
                      temperature: float = 0.25, max_tokens: int = 3072,
                      label: str = "gemini-json") -> Optional[dict]:
        txt, _ = self.generate(prompt, system, temperature, max_tokens, search=False, label=label)
        data = extract_json(txt)
        if data is None:
            txt, _ = self.generate(
                prompt + "\n\nReply with ONE valid JSON object only. No prose. No fences.",
                system, 0.1, max_tokens, search=False, label=label + "-repair")
            data = extract_json(txt)
        return data

print("Core engine loaded: retry_call, safe_eval, extract_json, GroqLLM, GeminiLLM")

#@title 4. The two NPCs: Inventor (Groq) and Scientist (Groq + Gemini research) + Referee
# NOTE: agents must output concise reasoning SUMMARIES only - never raw chain-of-thought.

NO_COT_RULE = (
    "Never reveal internal step-by-step deliberation or hidden reasoning. "
    "Publish only the finished, concise, decision-grade summary: claims, numbers, "
    "critiques, evidence, and conclusions."
)

INVENTOR_SYSTEM = """You are ARIA-7 "The Inventor", a senior R&D engineer-inventor in a live research room.
Personality: creative, ambitious, fast, concrete, slightly impatient - but intellectually honest.
You invent novel solutions, build quantitative models, and you DEFEND or REVISE them under fire.

Hard rules:
- You are speaking to a skeptical Scientist who just critiqued you. Answer their points DIRECTLY,
  one by one, by name of the claim. Concede explicitly when they are right ("You're right about X, I'm dropping it").
- Never repeat your previous message. Every round must show a concrete REVISION or a defended position with new evidence.
- Always produce at least one equation (LaTeX) and at least one executable numeric calculation.
- Calculations must be plain arithmetic expressions evaluable in Python with math functions only
  (e.g. "0.85 * P * h * 365 / 1000"), with every symbol supplied in "variables" as numbers.
- Use real-world orders of magnitude. State units. No hand-waving like "very cheap" or "massively scalable".
- """ + NO_COT_RULE + """

Reply with ONE valid JSON object only, matching exactly this schema:
{
  "message": "markdown, 130-260 words, spoken aloud in the research room, addressed to the Scientist",
  "solution_title": "short name of the current version of the idea",
  "solution_summary": "3-5 sentences describing the CURRENT design",
  "revision": "what changed this round vs the previous version, or 'initial proposal'",
  "answers_to_critiques": ["critique -> my direct answer"],
  "concessions": ["points where the Scientist was right and I changed the design"],
  "key_claims": ["falsifiable claim with a number and unit"],
  "equations": [{"name": "", "latex": "", "explanation": "", "symbols": ""}],
  "computations": [{"name": "", "expression": "", "variables": {"symbol": 0}, "unit": "", "why": ""}],
  "quantitative_estimates": [{"metric": "", "value": "", "unit": "", "basis": ""}],
  "architecture": ["component -> role"],
  "experiments": [{"name": "", "protocol": "", "metric": "", "success_criterion": ""}],
  "assumptions": ["assumption -> how it could be tested"],
  "open_questions": ["question for the Scientist"]
}"""

SCIENTIST_SYSTEM = """You are DR. KOVACS "The Scientist", a rigorous research scientist and peer reviewer in a live research room.
Personality: skeptical, analytical, precise, unimpressed by hype - but constructive: you fix ideas, you don't just kill them.

Hard rules:
- Attack the Inventor's LATEST message specifically: quote the claim, then state the defect.
- Re-derive their math. You are given VERIFIED CALCULATOR OUTPUT for their expressions - trust that arithmetic
  over the Inventor's prose, and call out any mismatch between their stated number and the executed number.
- Use the supplied RESEARCH BRIEF (independent, web-grounded) as evidence. Cite sources as [S1], [S2] matching the brief.
- Every critique must come with a requested_fix that is concrete and testable.
- Score your endorsement honestly: it should rise only when defects are actually fixed, and it must not
  exceed 60 while any high-severity defect is unresolved.
- Also propose at least one improvement or cheaper/simpler alternative of your own.
- """ + NO_COT_RULE + """

Reply with ONE valid JSON object only, matching exactly this schema:
{
  "message": "markdown, 130-260 words, spoken aloud, addressed to the Inventor, with [S#] citations where used",
  "verdict": "reject | major-revision | minor-revision | accept",
  "endorsement": 0,
  "critiques": [{"target_claim": "", "issue": "", "severity": "high|medium|low", "requested_fix": ""}],
  "math_check": [{"item": "", "status": "ok|wrong|unverified", "note": "", "corrected_latex": ""}],
  "evidence_used": [{"claim": "", "verdict": "supported|mixed|contradicted|no-evidence", "source_tag": "[S1]"}],
  "risks": [{"risk": "", "likelihood": "high|medium|low", "impact": "high|medium|low", "mitigation": ""}],
  "assumptions_challenged": ["assumption -> why it may fail -> test"],
  "improvements": ["concrete improvement I propose"],
  "cost_check": ["cost/feasibility reality check with numbers"],
  "questions_to_inventor": ["hard question"],
  "resolved_previous_contradictions": ["contradiction -> how it is now resolved"]
}"""

RESEARCH_SYSTEM = """You are an independent research analyst with web search. You are NOT part of the debate.
Verify the listed claims against current external sources. Be blunt about weak evidence.
Output compact markdown, max 320 words, in this format:

**[S1] <claim, shortened>** - VERDICT: supported / mixed / contradicted / no-evidence
Key numbers found: ...
Why: 1-2 sentences.

...repeat for each claim...

**Independent alternative:** one different approach to the same problem that the debaters have not considered, with a number.
**Falsifier:** the single cheapest test that would prove the main idea wrong.
Never invent statistics. If you cannot find data, write 'no reliable public data found'."""

REFEREE_SYSTEM = """You are the impartial Referee of a two-agent research debate. You do not take sides.
Score the CURRENT state of the idea strictly. A generic idea with no numbers scores below 4 on most axes.
Detect genuine logical/numeric contradictions between the agents or within one agent across rounds.
Agreement = how close the two agents actually are to a shared, defensible position (not politeness).
""" + NO_COT_RULE + """

Reply with ONE valid JSON object only:
{
  "scores": {"novelty": 0, "feasibility": 0, "cost_efficiency": 0, "scalability": 0,
             "math_consistency": 0, "evidence": 0, "expected_impact": 0,
             "risk_management": 0, "assumption_quality": 0},
  "score_notes": {"novelty": "one line", "feasibility": "one line", "math_consistency": "one line"},
  "agreement": 0,
  "unresolved_high_severity": 0,
  "contradictions": [{"between": "Inventor vs Scientist | Inventor vs Inventor",
                      "statement_a": "", "statement_b": "", "why_contradictory": "",
                      "resolution_required": ""}],
  "must_resolve_next": ["specific item the next round MUST settle"],
  "timeline_entry": "ONE sentence: exactly how the idea changed this round",
  "stop_recommended": false,
  "rationale": "max 40 words"
}
All scores are 0-10 integers. agreement is 0-100."""

BLUEPRINT_SYSTEM = """You are the rapporteur of the research room. Both agents have converged.
Write the FINAL CONSENSUS BLUEPRINT as detailed, publication-grade markdown.
Use ONLY content that survived the debate: keep conceded points out, keep corrected math in.
Include real numbers, units, LaTeX equations in $$...$$ blocks, and cite evidence as [S#] where the research brief supports it.
""" + NO_COT_RULE + """

Use exactly these level-2 headings, in this order:
## 1. Problem
## 2. Improved Solution
## 3. Why It Works
## 4. Mathematical / Quantitative Model
## 5. Evidence & Research
## 6. Architecture
## 7. Experiments
## 8. Risks
## 9. Assumptions
## 10. Expected Results
## 11. Implementation Plan

Rules per section: #4 must contain >=2 LaTeX equations plus a worked numeric example.
#6 must contain a component table (markdown table: Component | Role | Tech | Notes).
#7 must contain >=3 experiments with metric + success criterion + rough cost.
#8 must be a markdown table: Risk | Likelihood | Impact | Mitigation | Early warning signal.
#11 must have phased milestones with durations, owners (roles), budget estimate and a go/no-go gate per phase.
No preamble, no closing chatter. Start directly with '## 1. Problem'."""

# ----------------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------------
def transcript_block(turns: list, max_turns: int = 6, chars: int = 900) -> str:
    if not turns:
        return "(empty - this is the opening round)"
    rows = []
    for t in turns[-max_turns:]:
        rows.append(f"[Round {t['round']}] {t['who'].upper()}: {clip(t['message'], chars)}")
    return "\n\n".join(rows)

def inventor_prompt(state) -> str:
    inv = state.last_inventor or {}
    sci = state.last_scientist or {}
    p = [
        f"PROBLEM FROM THE USER:\n{state.problem}",
        f"\nROUND: {state.round} of {state.max_rounds}",
        f"\nRECENT TRANSCRIPT:\n{transcript_block(state.turns)}",
    ]
    if state.round == 1:
        p.append(
            "\nTHIS IS YOUR OPENING INDEPENDENT ANALYSIS.\n"
            "Give your own reading of the problem, then propose a concrete, novel solution with a "
            "quantitative model. Be specific enough to be attacked."
        )
    else:
        p.append("\nYOUR PREVIOUS DESIGN:\n" + clip(inv.get("solution_summary"), 900))
        p.append("\nSCIENTIST'S LATEST CRITIQUE (answer each item):\n" + clip(sci.get("message"), 1400))
        p.append("\nOPEN CRITIQUES:\n" + bullets(sci.get("critiques"), 6))
        p.append("\nMATH CHECK RESULTS ON YOUR WORK:\n" + bullets(sci.get("math_check"), 5))
        p.append("\nSCIENTIST'S QUESTIONS:\n" + bullets(sci.get("questions_to_inventor"), 5))
        p.append("\nSCIENTIST'S PROPOSED IMPROVEMENTS (adopt or refute with reasons):\n"
                 + bullets(sci.get("improvements"), 5))
        if state.computations:
            p.append("\nVERIFIED CALCULATOR OUTPUT FROM YOUR LAST ROUND (ground truth):\n"
                     + computations_report(state.computations[-4:]))
    if state.must_resolve:
        p.append("\nREFEREE SAYS THESE MUST BE RESOLVED NOW:\n" + bullets(state.must_resolve, 5))
    if state.contradictions:
        p.append("\nDETECTED CONTRADICTIONS YOU MUST FIX EXPLICITLY:\n"
                 + bullets(state.contradictions[-4:], 4))
    if state.evidence:
        p.append("\nLATEST INDEPENDENT RESEARCH BRIEF (may contradict you):\n"
                 + clip(state.evidence[-1]["text"], 1600))
    p.append("\nNow speak. JSON object only.")
    return "\n".join(p)

def scientist_prompt(state, research_md: str, calc_report: str) -> str:
    inv = state.last_inventor or {}
    p = [
        f"PROBLEM FROM THE USER:\n{state.problem}",
        f"\nROUND: {state.round} of {state.max_rounds}",
        f"\nRECENT TRANSCRIPT:\n{transcript_block(state.turns)}",
        "\nINVENTOR'S LATEST MESSAGE (attack this):\n" + clip(inv.get("message"), 1600),
        "\nINVENTOR'S CURRENT DESIGN:\n" + clip(inv.get("solution_summary"), 900),
        "\nINVENTOR'S KEY CLAIMS:\n" + bullets(inv.get("key_claims"), 8),
        "\nINVENTOR'S EQUATIONS:\n" + bullets(inv.get("equations"), 5),
        "\nINVENTOR'S QUANTITATIVE ESTIMATES:\n" + bullets(inv.get("quantitative_estimates"), 6),
        "\nVERIFIED CALCULATOR OUTPUT (executed by a real Python evaluator - this arithmetic is ground truth):\n"
        + calc_report,
        "\nINDEPENDENT WEB-GROUNDED RESEARCH BRIEF (source tags [S#]):\n" + clip(research_md, 2600),
    ]
    if state.round == 1:
        p.append("\nAlso open with your OWN independent reading of the problem in one or two sentences "
                 "before you attack the proposal.")
    if state.must_resolve:
        p.append("\nREFEREE REQUIRES THESE TO BE SETTLED NOW:\n" + bullets(state.must_resolve, 5))
    if state.contradictions:
        p.append("\nOPEN CONTRADICTIONS:\n" + bullets(state.contradictions[-4:], 4))
    p.append("\nNow speak. JSON object only.")
    return "\n".join(p)

def research_prompt(state, claims: list) -> str:
    return (
        f"Problem under debate: {clip(state.problem, 600)}\n\n"
        f"Proposed solution: {clip((state.last_inventor or {}).get('solution_summary'), 700)}\n\n"
        "Verify these claims with web search, in order (label them [S1], [S2], ... in order):\n"
        + bullets(claims, 5, prefix="* ")
    )

def referee_prompt(state) -> str:
    inv, sci = state.last_inventor or {}, state.last_scientist or {}
    return "\n".join([
        f"PROBLEM: {clip(state.problem, 500)}",
        f"ROUND {state.round} of {state.max_rounds}",
        "\nINVENTOR THIS ROUND:\n" + clip(inv.get("message"), 1300),
        "Design: " + clip(inv.get("solution_summary"), 700),
        "Claims:\n" + bullets(inv.get("key_claims"), 6),
        "Concessions:\n" + bullets(inv.get("concessions"), 4),
        "\nSCIENTIST THIS ROUND:\n" + clip(sci.get("message"), 1300),
        f"Verdict: {sci.get('verdict')} | Endorsement: {sci.get('endorsement')}",
        "Critiques:\n" + bullets(sci.get("critiques"), 6),
        "Math check:\n" + bullets(sci.get("math_check"), 4),
        "\nEXECUTED CALCULATIONS (ground truth):\n" + computations_report(state.computations[-4:]),
        "\nPREVIOUS TIMELINE:\n" + bullets(state.timeline, 5),
        "\nPREVIOUSLY UNRESOLVED:\n" + bullets(state.must_resolve, 5),
        "\nScore now. JSON only.",
    ])

def blueprint_prompt(state) -> str:
    inv, sci = state.last_inventor or {}, state.last_scientist or {}
    ev = "\n\n".join(clip(e["text"], 1200) for e in state.evidence[-3:]) or "(no external research retrieved)"
    return "\n".join([
        f"ORIGINAL PROBLEM:\n{state.problem}",
        f"\nROUNDS COMPLETED: {state.round} | FINAL AGREEMENT: {state.agreement}/100 "
        f"| QUALITY SCORE: {state.quality}/100",
        "\nFINAL DESIGN (Inventor):\n" + clip(inv.get("solution_summary"), 1400),
        "\nFINAL ARCHITECTURE NOTES:\n" + bullets(inv.get("architecture"), 8),
        "\nSURVIVING EQUATIONS:\n" + bullets(state.equations[-8:], 8),
        "\nEXECUTED CALCULATIONS (use these exact numbers):\n" + computations_report(state.computations[-8:]),
        "\nEXPERIMENTS PROPOSED:\n" + bullets(inv.get("experiments"), 6),
        "\nSCIENTIST'S FINAL POSITION:\n" + clip(sci.get("message"), 1200),
        f"Verdict: {sci.get('verdict')} | Endorsement: {sci.get('endorsement')}",
        "\nREMAINING CRITIQUES (address them in Risks/Assumptions):\n" + bullets(sci.get("critiques"), 6),
        "\nRISKS RAISED:\n" + bullets(sci.get("risks"), 6),
        "\nASSUMPTIONS CHALLENGED:\n" + bullets(sci.get("assumptions_challenged"), 6),
        "\nIMPROVEMENTS ACCEPTED:\n" + bullets(sci.get("improvements"), 6),
        "\nIDEA EVOLUTION TIMELINE:\n" + bullets(state.timeline, 10),
        "\nRESEARCH EVIDENCE:\n" + ev,
        "\nWrite the blueprint now.",
    ])

VERIFY_SYSTEM = (
    "You are an external auditor with web search. Audit the blueprint for: factual errors, "
    "arithmetic errors, unsupported numbers, missing regulatory/safety issues, and over-optimism. "
    "Output max 220 words of markdown: a short bullet list of findings, each marked "
    "[CORRECT] / [QUESTIONABLE] / [WRONG], then one line 'Overall confidence: X/10'. "
    "Do not rewrite the blueprint."
)

print("Agents loaded: Inventor (Groq), Scientist (Groq+Gemini research), Referee (Gemini), Rapporteur.")

#@title 5. Orchestrator: alternating rounds, contradiction handling, consensus, blueprint
import traceback
from datetime import datetime

SCORE_WEIGHTS = {
    "novelty": 0.10, "feasibility": 0.16, "cost_efficiency": 0.10, "scalability": 0.10,
    "math_consistency": 0.14, "evidence": 0.14, "expected_impact": 0.12,
    "risk_management": 0.08, "assumption_quality": 0.06,
}
SCORE_LABELS = {
    "novelty": "Novelty", "feasibility": "Feasibility", "cost_efficiency": "Cost efficiency",
    "scalability": "Scalability", "math_consistency": "Math consistency", "evidence": "Evidence",
    "expected_impact": "Expected impact", "risk_management": "Risk management",
    "assumption_quality": "Assumption quality",
}
# Hard ceiling on the agreement meter implied by the Scientist's formal verdict.
VERDICT_CAP = {"reject": 40, "major-revision": 65, "minor-revision": 90, "accept": 100}

@dataclass
class ArenaState:
    problem: str = ""
    round: int = 0
    max_rounds: int = 4
    threshold: int = 85
    turns: List[Dict[str, Any]] = field(default_factory=list)
    chat: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)
    equations: List[Dict[str, Any]] = field(default_factory=list)
    computations: List[Dict[str, Any]] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    score_notes: Dict[str, str] = field(default_factory=dict)
    agreement: int = 0
    quality: float = 0.0
    endorsement: int = 0
    verdict: str = "-"
    timeline: List[str] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    must_resolve: List[str] = field(default_factory=list)
    last_inventor: Dict[str, Any] = field(default_factory=dict)
    last_scientist: Dict[str, Any] = field(default_factory=dict)
    inv_status: str = "idle"
    sci_status: str = "idle"
    ref_status: str = "idle"
    banner: str = "Waiting for a problem."
    blueprint: str = ""
    verification: str = ""
    blueprint_path: str = ""
    finished: bool = False
    error: str = ""
    stop_reason: str = ""

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def push_chat(state: ArenaState, who: str, text: str, meta: str = "") -> None:
    role = "user" if who == "inventor" else "assistant"
    head = ("### 🛠️ Inventor · ARIA-7" if who == "inventor" else "### 🔬 Scientist · Dr. Kovacs")
    sub = f"Round {state.round} · {_now()}" + (f" · {meta}" if meta else "")
    state.chat.append({"role": role, "content": f"{head}\n*{sub}*\n\n{text}"})

def compute_quality(scores: Dict[str, Any]) -> float:
    if not scores:
        return 0.0
    total, wsum = 0.0, 0.0
    for k, w in SCORE_WEIGHTS.items():
        if k in scores:
            v = max(0.0, min(10.0, as_num(scores[k])))
            total += v * w
            wsum += w
    return round((total / wsum) * 10, 1) if wsum else 0.0

def blend_agreement(referee_agreement: float, endorsement: float, verdict: str,
                    unresolved_high: float) -> int:
    base = 0.55 * as_num(referee_agreement) + 0.45 * as_num(endorsement)
    base = min(base, VERDICT_CAP.get(str(verdict).strip().lower(), 100))
    base -= 8.0 * max(0.0, as_num(unresolved_high))
    return int(max(0, min(100, round(base))))

def normalize_inventor(data: Optional[dict], raw: str) -> dict:
    d = dict(data or {})
    if not d.get("message"):
        d["message"] = clip(raw, 1800) or "_(no output produced this round)_"
    d["message"] = sanitize(str(d["message"]))
    for k in ("key_claims", "equations", "computations", "quantitative_estimates",
              "experiments", "architecture", "assumptions", "open_questions",
              "answers_to_critiques", "concessions"):
        d[k] = as_list(d.get(k))
    d["solution_title"] = clip(d.get("solution_title") or "Untitled concept", 120)
    d["solution_summary"] = clip(d.get("solution_summary") or d["message"], 1500)
    d["revision"] = clip(d.get("revision") or "", 400)
    return d

def normalize_scientist(data: Optional[dict], raw: str) -> dict:
    d = dict(data or {})
    if not d.get("message"):
        d["message"] = clip(raw, 1800) or "_(no output produced this round)_"
    d["message"] = sanitize(str(d["message"]))
    for k in ("critiques", "math_check", "evidence_used", "risks", "assumptions_challenged",
              "improvements", "cost_check", "questions_to_inventor",
              "resolved_previous_contradictions"):
        d[k] = as_list(d.get(k))
    v = str(d.get("verdict") or "major-revision").strip().lower()
    d["verdict"] = v if v in VERDICT_CAP else "major-revision"
    d["endorsement"] = int(max(0, min(100, as_num(d.get("endorsement"), 40))))
    return d

def high_severity_open(sci: dict) -> int:
    n = 0
    for c in sci.get("critiques", []):
        if isinstance(c, dict) and str(c.get("severity", "")).lower().startswith("high"):
            n += 1
    return n

# ----------------------------------------------------------------------------
# Main streaming orchestrator
# ----------------------------------------------------------------------------
def run_arena(problem: str, max_rounds: int, threshold: int,
              groq: "GroqLLM", gemini: "GeminiLLM", do_research: bool = True):
    """Generator that yields an ArenaState snapshot after every micro-step."""
    st = ArenaState(problem=problem.strip(), max_rounds=int(max_rounds), threshold=int(threshold))
    st.banner = "Briefing both agents..."
    yield st

    # ---- Phase 0: independent problem brief (Gemini, grounded) --------------
    if do_research:
        st.sci_status = "reading the literature"
        st.banner = "Independent problem brief (Gemini + web)..."
        yield st
        try:
            brief, cits = gemini.generate(
                "Give a compact, source-grounded brief on this problem before any solution is proposed: "
                f"\"{clip(problem, 700)}\"\n"
                "Cover in max 250 words: (1) what is already done today and its measured limits with numbers, "
                "(2) the 3 hardest technical/economic constraints, (3) the key metric any solution must beat, "
                "with its current best value and units. Label sources [S1], [S2], ...",
                system=RESEARCH_SYSTEM, search=True, temperature=0.3, label="brief")
            if brief:
                st.evidence.append({"round": 0, "title": "Independent problem brief",
                                    "text": brief, "citations": cits})
                st.citations.extend(cits)
        except Exception as e:  # noqa: BLE001
            st.evidence.append({"round": 0, "title": "Independent problem brief",
                                "text": f"_Research unavailable: {clip(e, 300)}_", "citations": []})
        st.sci_status = "idle"
        yield st

    # ---- Debate rounds ------------------------------------------------------
    for rnd in range(1, st.max_rounds + 1):
        st.round = rnd

        # --- Inventor turn (Groq) -------------------------------------------
        st.inv_status = "drafting a design"
        st.banner = f"Round {rnd}: Inventor is designing..."
        yield st
        try:
            data, raw = groq.complete_json(INVENTOR_SYSTEM, inventor_prompt(st),
                                           temperature=0.75, max_tokens=2600,
                                           label=f"inventor-r{rnd}")
            inv = normalize_inventor(data, raw)
        except Exception as e:  # noqa: BLE001
            st.error = f"Inventor failed in round {rnd}: {clip(e, 400)}"
            st.inv_status = "error"
            st.banner = st.error
            yield st
            break
        st.last_inventor = inv
        st.turns.append({"round": rnd, "who": "inventor", "message": inv["message"]})
        for eq in inv["equations"]:
            if isinstance(eq, dict) and eq.get("latex"):
                st.equations.append({"round": rnd, "owner": "Inventor", **eq})
        push_chat(st, "inventor", inv["message"],
                  meta=f"v{rnd} · {clip(inv['solution_title'], 60)}")
        st.inv_status = "idle"
        yield st

        # --- Execute the Inventor's arithmetic (real ground truth) -----------
        st.ref_status = "running calculations"
        st.banner = f"Round {rnd}: executing {len(inv['computations'])} calculation(s)..."
        yield st
        results = execute_computations(inv["computations"])
        for r in results:
            r["round"] = rnd
        st.computations.extend(results)
        calc_report = computations_report(results)
        st.ref_status = "idle"
        yield st

        # --- Independent research (Gemini, grounded) -------------------------
        research_md = "(research disabled)"
        if do_research:
            st.sci_status = "researching claims"
            st.banner = f"Round {rnd}: Scientist is verifying claims against sources..."
            yield st
            claims = [c for c in inv["key_claims"] if c][:5] or [inv["solution_summary"]]
            try:
                research_md, cits = gemini.generate(research_prompt(st, claims),
                                                    system=RESEARCH_SYSTEM, search=True,
                                                    temperature=0.3, label=f"research-r{rnd}")
                research_md = research_md or "(no research text returned)"
                st.evidence.append({"round": rnd, "title": f"Round {rnd} claim verification",
                                    "text": research_md, "citations": cits})
                st.citations.extend(cits)
            except Exception as e:  # noqa: BLE001
                research_md = f"_Research unavailable this round: {clip(e, 260)}_"
                st.evidence.append({"round": rnd, "title": f"Round {rnd} claim verification",
                                    "text": research_md, "citations": []})
            yield st

        # --- Scientist turn (Groq, armed with research + verified math) ------
        st.sci_status = "writing the critique"
        st.banner = f"Round {rnd}: Scientist is responding..."
        yield st
        try:
            data, raw = groq.complete_json(SCIENTIST_SYSTEM,
                                           scientist_prompt(st, research_md, calc_report),
                                           temperature=0.45, max_tokens=2600,
                                           label=f"scientist-r{rnd}")
            sci = normalize_scientist(data, raw)
        except Exception as e:  # noqa: BLE001
            st.error = f"Scientist failed in round {rnd}: {clip(e, 400)}"
            st.sci_status = "error"
            st.banner = st.error
            yield st
            break
        st.last_scientist = sci
        st.endorsement = sci["endorsement"]
        st.verdict = sci["verdict"]
        st.turns.append({"round": rnd, "who": "scientist", "message": sci["message"]})
        for mc in sci["math_check"]:
            if isinstance(mc, dict) and mc.get("corrected_latex"):
                st.equations.append({"round": rnd, "owner": "Scientist (correction)",
                                     "name": clip(mc.get("item"), 80),
                                     "latex": mc["corrected_latex"],
                                     "explanation": clip(mc.get("note"), 300)})
        push_chat(st, "scientist", sci["message"],
                  meta=f"verdict: {sci['verdict']} · endorsement {sci['endorsement']}/100")
        st.sci_status = "idle"
        yield st

        # --- Referee scoring + contradiction detection (Gemini) --------------
        st.ref_status = "scoring the round"
        st.banner = f"Round {rnd}: Referee is scoring and checking for contradictions..."
        yield st
        ref = None
        try:
            ref = gemini.generate_json(referee_prompt(st), system=REFEREE_SYSTEM,
                                       temperature=0.2, label=f"referee-r{rnd}")
        except Exception:
            ref = None
        if ref is None:  # fallback to Groq so the arena never stalls
            try:
                ref, _ = groq.complete_json(REFEREE_SYSTEM, referee_prompt(st),
                                            temperature=0.2, max_tokens=1600,
                                            label=f"referee-fb-r{rnd}")
            except Exception:
                ref = None
        ref = ref or {}

        raw_scores = ref.get("scores") if isinstance(ref.get("scores"), dict) else {}
        st.scores = {k: max(0.0, min(10.0, as_num(raw_scores.get(k), 0))) for k in SCORE_WEIGHTS}
        st.score_notes = {k: clip(v, 160) for k, v in
                          (ref.get("score_notes") or {}).items() if isinstance(v, str)}
        st.quality = compute_quality(st.scores)

        open_high = as_num(ref.get("unresolved_high_severity"), high_severity_open(sci))
        st.agreement = blend_agreement(ref.get("agreement", sci["endorsement"]),
                                       sci["endorsement"], sci["verdict"], open_high)

        new_contra = [c for c in as_list(ref.get("contradictions")) if isinstance(c, dict)]
        st.contradictions = new_contra
        st.must_resolve = [clip(x, 300) for x in as_list(ref.get("must_resolve_next"))[:5] if x]
        entry = clip(ref.get("timeline_entry") or inv.get("revision") or "Idea refined.", 320)
        st.timeline.append(
            f"**Round {rnd}** · agreement {st.agreement}/100 · quality {st.quality}/100 · "
            f"verdict `{st.verdict}`\n  - {entry}"
            + (f"\n  - Conceded: {clip(inv['concessions'][0], 200)}" if inv["concessions"] else "")
            + (f"\n  - ⚠ {len(new_contra)} contradiction(s) flagged for next round" if new_contra else "")
        )
        st.ref_status = "idle"
        yield st

        # --- Consensus check -------------------------------------------------
        converged = (st.agreement >= st.threshold and st.verdict in ("accept", "minor-revision")
                     and open_high == 0)
        if converged:
            st.stop_reason = (f"Consensus reached in round {rnd}: agreement {st.agreement} "
                              f">= threshold {st.threshold}, verdict `{st.verdict}`, "
                              f"no unresolved high-severity defects.")
            st.banner = st.stop_reason
            yield st
            break
        if rnd == st.max_rounds:
            st.stop_reason = (f"Round limit reached ({st.max_rounds}). Final agreement "
                              f"{st.agreement}/100 (threshold {st.threshold}). "
                              "Blueprint will carry the open items as risks.")
            st.banner = st.stop_reason
            yield st
        else:
            st.banner = (f"Round {rnd} closed · agreement {st.agreement}/100 · "
                         f"{len(st.must_resolve)} item(s) pushed to round {rnd + 1}")
            yield st

    # ---- Final blueprint ----------------------------------------------------
    if st.turns:
        st.inv_status = st.sci_status = "drafting consensus"
        st.banner = "Both agents are writing the Consensus Blueprint..."
        yield st
        try:
            bp = groq.complete(BLUEPRINT_SYSTEM, blueprint_prompt(st),
                               temperature=0.4, max_tokens=5200, label="blueprint")
            st.blueprint = sanitize(bp)
        except Exception as e:  # noqa: BLE001
            st.blueprint = f"**Blueprint generation failed:** {clip(e, 400)}"

        if do_research and st.blueprint and not st.blueprint.startswith("**Blueprint generation failed"):
            st.ref_status = "auditing blueprint"
            st.banner = "Independent audit of the blueprint (Gemini + web)..."
            yield st
            try:
                ver, cits = gemini.generate(
                    "Audit this consensus blueprint:\n\n" + clip(st.blueprint, 9000),
                    system=VERIFY_SYSTEM, search=True, temperature=0.25, label="audit")
                st.verification = ver
                st.citations.extend(cits)
            except Exception as e:  # noqa: BLE001
                st.verification = f"_Audit unavailable: {clip(e, 260)}_"

        st.blueprint_path = save_blueprint(st)
        st.inv_status = st.sci_status = st.ref_status = "idle"

    st.finished = True
    if not st.error:
        st.banner = st.stop_reason or "Session complete."
    yield st

OUT_DIR = "/content" if os.path.isdir("/content") else os.getcwd()

def save_blueprint(st: ArenaState) -> str:
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUT_DIR, f"IdeaForge_Blueprint_{stamp}.md")
        seen, cit_lines = set(), []
        for i, c in enumerate(st.citations, 1):
            if c.get("uri") and c["uri"] not in seen:
                seen.add(c["uri"])
                cit_lines.append(f"{len(cit_lines)+1}. [{clip(c.get('title'), 120)}]({c['uri']})")
        doc = [
            "# IdeaForge Arena - Consensus Blueprint",
            f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · rounds: {st.round} · "
            f"agreement: {st.agreement}/100 · idea quality: {st.quality}/100 · "
            f"final verdict: {st.verdict}_",
            f"\n**Original problem:** {st.problem}\n",
            "---\n",
            st.blueprint or "_(no blueprint)_",
            "\n---\n\n## Independent Audit (Gemini, web-grounded)\n",
            st.verification or "_(not run)_",
            "\n---\n\n## Idea Evolution Timeline\n",
            "\n".join(st.timeline) or "_(empty)_",
            "\n---\n\n## Sources\n",
            "\n".join(cit_lines) or "_(no web sources retrieved)_",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(doc))
        return path
    except Exception:
        traceback.print_exc()
        return ""

print("Orchestrator loaded: run_arena() streams an ArenaState after every step.")

#@title 6. UI: the live AI research room (Gradio)
import html as _html
import gradio as gr

# Gradio 5 and 6 differ in a few constructor arguments - adapt instead of pinning.
GR_MAJOR = int(str(getattr(gr, "__version__", "5.0")).split(".")[0] or 5)

LATEX = [{"left": "$$", "right": "$$", "display": True},
         {"left": "$", "right": "$", "display": False},
         {"left": "\\(", "right": "\\)", "display": False},
         {"left": "\\[", "right": "\\]", "display": True}]

CSS = """
.gradio-container {max-width: 1500px !important;}
#ifa-title {text-align:center; padding:14px 8px 4px 8px;}
#ifa-title h1 {margin:0; font-size:30px; letter-spacing:.5px;
  background:linear-gradient(90deg,#7c3aed,#06b6d4,#22c55e);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
#ifa-title p {margin:4px 0 0 0; opacity:.72; font-size:13px;}
.ifa-card {border-radius:14px; padding:12px 14px; border:1px solid rgba(128,128,128,.28);
  background:linear-gradient(180deg, rgba(120,120,180,.10), rgba(120,120,180,.03)); min-height:96px;}
.ifa-card.inv {border-left:5px solid #f59e0b;}
.ifa-card.sci {border-left:5px solid #06b6d4;}
.ifa-card.active {box-shadow:0 0 0 2px rgba(124,58,237,.35); animation: ifapulse 1.4s ease-in-out infinite;}
@keyframes ifapulse {0%{box-shadow:0 0 0 0 rgba(124,58,237,.40);}
  70%{box-shadow:0 0 0 12px rgba(124,58,237,0);} 100%{box-shadow:0 0 0 0 rgba(124,58,237,0);}}
.ifa-name {font-weight:700; font-size:15px; display:flex; justify-content:space-between; align-items:center;}
.ifa-role {font-size:12px; opacity:.72; margin-top:2px;}
.ifa-status {margin-top:8px; font-size:12.5px; font-family:ui-monospace,Menlo,monospace;}
.ifa-dot {display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px;
  background:#9ca3af; vertical-align:middle;}
.ifa-dot.on {background:#22c55e; animation: ifablink .9s infinite alternate;}
.ifa-dot.err {background:#ef4444;}
@keyframes ifablink {from{opacity:.25;} to{opacity:1;}}
.ifa-bar {height:9px; border-radius:6px; background:rgba(128,128,128,.22); overflow:hidden; margin:3px 0 9px 0;}
.ifa-bar > span {display:block; height:100%; border-radius:6px;}
.ifa-lbl {display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px;}
.ifa-kpi {display:flex; gap:8px; margin-bottom:10px;}
.ifa-kpi div {flex:1; text-align:center; border-radius:10px; padding:7px 4px;
  background:rgba(128,128,128,.13); border:1px solid rgba(128,128,128,.2);}
.ifa-kpi b {display:block; font-size:19px;}
.ifa-kpi small {opacity:.7; font-size:11px;}
#ifa-banner {font-size:13px; padding:9px 13px; border-radius:10px;
  background:rgba(124,58,237,.10); border:1px solid rgba(124,58,237,.30);}
"""

def _bar(label, value, maxv=100, color="#7c3aed", suffix=""):
    pct = max(0.0, min(100.0, (as_num(value) / maxv) * 100.0))
    return (f'<div class="ifa-lbl"><span>{_html.escape(str(label))}</span>'
            f'<span>{value}{suffix}</span></div>'
            f'<div class="ifa-bar"><span style="width:{pct:.1f}%;background:{color};"></span></div>')

def _color_for(v, maxv=100):
    p = as_num(v) / maxv
    return "#ef4444" if p < 0.4 else ("#f59e0b" if p < 0.7 else "#22c55e")

def render_card(who, st: ArenaState) -> str:
    if who == "inventor":
        name, role, cls = "ARIA-7 · The Inventor", "Creative synthesis · models · ambition (Groq)", "inv"
        status, extra = st.inv_status, clip((st.last_inventor or {}).get("solution_title"), 70)
    else:
        name, role, cls = "Dr. Kovacs · The Scientist", "Skeptical review · evidence · math (Groq + Gemini)", "sci"
        status = st.sci_status
        extra = (f"verdict: {st.verdict} · endorsement {st.endorsement}/100"
                 if st.last_scientist else "")
    busy = status not in ("idle", "error")
    dot = "err" if status == "error" else ("on" if busy else "")
    return (f'<div class="ifa-card {cls} {"active" if busy else ""}">'
            f'<div class="ifa-name"><span>{_html.escape(name)}</span>'
            f'<span style="font-size:11px;opacity:.6;">Round {st.round}/{st.max_rounds}</span></div>'
            f'<div class="ifa-role">{_html.escape(role)}</div>'
            f'<div class="ifa-status"><span class="ifa-dot {dot}"></span>'
            f'{_html.escape(status if busy or status == "error" else ("done" if st.finished else "standing by"))}'
            f'{" · thinking…" if busy else ""}</div>'
            f'<div class="ifa-status" style="opacity:.75;">{_html.escape(extra or "—")}</div></div>')

def render_meters(st: ArenaState) -> str:
    h = ['<div class="ifa-kpi">'
         f'<div><b>{st.round}/{st.max_rounds}</b><small>round</small></div>'
         f'<div><b>{st.agreement}</b><small>agreement</small></div>'
         f'<div><b>{st.quality:g}</b><small>idea score</small></div>'
         f'<div><b>{len(st.citations)}</b><small>sources</small></div></div>']
    h.append(_bar("Agreement meter", st.agreement, 100, _color_for(st.agreement)))
    h.append(_bar(f"Consensus threshold ({st.threshold})", st.threshold, 100, "#64748b"))
    h.append(_bar("Idea quality score", st.quality, 100, _color_for(st.quality)))
    h.append('<div style="font-size:12px;opacity:.75;margin:6px 0 4px;">Referee criteria (0–10)</div>')
    for k in SCORE_WEIGHTS:
        v = st.scores.get(k, 0)
        h.append(_bar(f"{SCORE_LABELS[k]} · w={SCORE_WEIGHTS[k]:.2f}", round(as_num(v), 1), 10,
                      _color_for(v, 10)))
    return "".join(h)

def render_banner(st: ArenaState) -> str:
    icon = "⛔" if st.error else ("✅" if st.finished else "⚙️")
    return f'<div id="ifa-banner">{icon} {_html.escape(st.error or st.banner)}</div>'

def render_evidence(st: ArenaState) -> str:
    if not st.evidence:
        return "_No research performed yet._"
    out = []
    for e in reversed(st.evidence):
        out.append(f"### {e['title']}\n{e['text']}")
        if e.get("citations"):
            out.append("**Sources:** " + " · ".join(
                f"[{clip(c.get('title'), 60)}]({c['uri']})" for c in e["citations"][:8]))
        out.append("\n---")
    seen, links = set(), []
    for c in st.citations:
        if c.get("uri") and c["uri"] not in seen:
            seen.add(c["uri"])
            links.append(f"- [{clip(c.get('title'), 90)}]({c['uri']})")
    if links:
        out.append("### All retrieved sources\n" + "\n".join(links[:40]))
    return "\n\n".join(out)

def render_equations(st: ArenaState) -> str:
    out = []
    if st.equations:
        out.append("## Equations on the board")
        for eq in st.equations[-12:]:
            tex = str(eq.get("latex", "")).strip().strip("$")
            out.append(f"**R{eq.get('round')} · {eq.get('owner')} · "
                       f"{clip(eq.get('name') or 'equation', 90)}**\n\n$$ {tex} $$\n\n"
                       f"{clip(eq.get('explanation'), 400)}")
            if eq.get("symbols"):
                out.append(f"*Symbols:* {clip(eq.get('symbols'), 300)}")
    if st.computations:
        out.append("## Executed calculations (Python-verified)")
        out.append(computations_report(st.computations[-14:]))
        bad = [c for c in st.computations if "error" in c]
        if bad:
            out.append(f"> ⚠ {len(bad)} expression(s) failed execution and were flagged to the Scientist.")
    return "\n\n".join(out) if out else "_No equations or calculations yet._"

def render_timeline(st: ArenaState) -> str:
    if not st.timeline:
        return "_The idea has not evolved yet._"
    head = f"**Original problem:** {clip(st.problem, 400)}\n\n"
    return head + "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(st.timeline))

def render_contradictions(st: ArenaState) -> str:
    out = []
    if st.contradictions:
        out.append("## ⚠ Contradictions the agents must resolve")
        for c in st.contradictions:
            out.append(
                f"- **{clip(c.get('between'), 60)}**\n"
                f"  - A: {clip(c.get('statement_a'), 300)}\n"
                f"  - B: {clip(c.get('statement_b'), 300)}\n"
                f"  - Why: {clip(c.get('why_contradictory'), 300)}\n"
                f"  - Required resolution: {clip(c.get('resolution_required'), 300)}")
    else:
        out.append("## Contradictions\n_None outstanding._")
    if st.must_resolve:
        out.append("## Referee's must-resolve queue\n" + bullets(st.must_resolve, 6))
    sci = st.last_scientist or {}
    if sci.get("critiques"):
        out.append("## Open critiques (latest round)\n" + bullets(sci["critiques"], 8))
    if sci.get("risks"):
        out.append("## Risk register\n" + bullets(sci["risks"], 8))
    if sci.get("assumptions_challenged"):
        out.append("## Assumptions under challenge\n" + bullets(sci["assumptions_challenged"], 8))
    if st.score_notes:
        out.append("## Referee notes\n" + bullets(
            [f"{SCORE_LABELS.get(k, k)}: {v}" for k, v in st.score_notes.items()], 9))
    return "\n\n".join(out)

def render_blueprint(st: ArenaState) -> str:
    if not st.blueprint:
        return ("_The Consensus Blueprint appears here once the agents converge "
                "or the round limit is reached._")
    head = (f"# 📐 Consensus Blueprint\n"
            f"*Rounds: {st.round} · Agreement: {st.agreement}/100 · Idea quality: {st.quality}/100 · "
            f"Final verdict: `{st.verdict}` · Stop reason: {st.stop_reason or 'n/a'}*\n\n---\n\n")
    return head + st.blueprint

def render_audit(st: ArenaState) -> str:
    if not st.verification:
        return "_Independent audit runs after the blueprint is written._"
    return "## 🔎 Independent audit of the blueprint (Gemini + web)\n\n" + st.verification

def snapshot(st: ArenaState):
    return (
        st.chat,
        render_card("inventor", st),
        render_card("scientist", st),
        render_banner(st),
        render_meters(st),
        render_evidence(st),
        render_equations(st),
        render_timeline(st),
        render_contradictions(st),
        render_blueprint(st),
        render_audit(st),
        gr.update(value=st.blueprint_path or None, visible=bool(st.blueprint_path)),
        gr.update(interactive=st.finished or bool(st.error)),
    )

# ----------------------------------------------------------------------------
# Client management
# ----------------------------------------------------------------------------
ENGINE: Dict[str, Any] = {"groq": None, "gemini": None, "gk": "", "mk": "", "gm": "", "mm": ""}

def ensure_clients(groq_key, gemini_key, groq_model, gemini_model):
    gk = (groq_key or os.environ.get("GROQ_API_KEY") or "").strip()
    mk = (gemini_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not gk:
        raise ValueError("Groq API key missing. Paste it in the Keys & models panel.")
    if not mk:
        raise ValueError("Gemini API key missing. Paste it in the Keys & models panel.")
    if ENGINE["groq"] is None or gk != ENGINE["gk"]:
        ENGINE["groq"], ENGINE["gk"] = GroqLLM(gk, groq_model or None), gk
    if groq_model and groq_model in ENGINE["groq"].available:
        ENGINE["groq"].model = groq_model
    if ENGINE["gemini"] is None or mk != ENGINE["mk"]:
        ENGINE["gemini"], ENGINE["mk"] = GeminiLLM(mk, gemini_model or None), mk
    if gemini_model and gemini_model in ENGINE["gemini"].available:
        ENGINE["gemini"].model = gemini_model
    return ENGINE["groq"], ENGINE["gemini"]

def connect(groq_key, gemini_key):
    try:
        g, m = ensure_clients(groq_key, gemini_key, None, None)
        msg = (f"✅ Connected · Groq: **{g.model}** ({len(g.available)} models) · "
               f"Gemini: **{m.model}** ({len(m.available)} models)")
        return (gr.update(choices=g.available, value=g.model),
                gr.update(choices=m.available, value=m.model), msg)
    except Exception as e:  # noqa: BLE001
        return (gr.update(), gr.update(), f"❌ {clip(e, 400)}")

# ----------------------------------------------------------------------------
# Main streaming callback
# ----------------------------------------------------------------------------
def start(problem, max_rounds, threshold, do_research, groq_key, gemini_key, groq_model, gemini_model):
    st = ArenaState(problem=problem or "", max_rounds=int(max_rounds), threshold=int(threshold))
    if not (problem or "").strip():
        st.error = "Enter a problem or a vague idea first."
        st.finished = True
        yield snapshot(st)
        return
    st.banner = "Connecting to Groq and Gemini..."
    yield snapshot(st)
    try:
        groq_c, gem_c = ensure_clients(groq_key, gemini_key, groq_model, gemini_model)
    except Exception as e:  # noqa: BLE001
        st.error = f"Connection failed: {clip(e, 400)}"
        st.finished = True
        yield snapshot(st)
        return
    try:
        for state in run_arena(problem, int(max_rounds), int(threshold), groq_c, gem_c,
                               do_research=bool(do_research)):
            yield snapshot(state)
    except Exception as e:  # noqa: BLE001
        st.error = f"Arena crashed: {clip(e, 500)}"
        st.finished = True
        traceback.print_exc()
        yield snapshot(st)

def reset():
    st = ArenaState()
    st.banner = "Cleared. Enter a new problem."
    return snapshot(st)

EXAMPLES = [
    "Cut peak electricity bills for a 40-flat apartment building in Hisar without a grid upgrade.",
    "A cheap way to detect early-stage wheat rust in smallholder fields using phones.",
    "Make lithium-ion battery recycling profitable at a 500 tonnes/year scale.",
    "Something that fixes last-mile cold chain for vaccines in rural areas. Vague, make it real.",
    "Reduce hallucination in retrieval-augmented LLM systems without a bigger model.",
]

THEME = gr.themes.Soft(primary_hue="violet", secondary_hue="cyan")
CHAT_KW = dict(label="Research room · live transcript", height=600, latex_delimiters=LATEX)
BLOCKS_KW: Dict[str, Any] = dict(title="IdeaForge Arena")
LAUNCH_KW: Dict[str, Any] = dict(share=True, inline=False, debug=False, show_error=True)
if GR_MAJOR >= 6:                      # css/theme moved to launch(); messages is the default format
    LAUNCH_KW.update(css=CSS, theme=THEME)
else:
    BLOCKS_KW.update(css=CSS, theme=THEME)
    CHAT_KW.update(type="messages", show_copy_button=True)

with gr.Blocks(**BLOCKS_KW) as demo:
    gr.HTML('<div id="ifa-title"><h1>⚗️ IdeaForge Arena</h1>'
            '<p>Two autonomous NPC experts debate your problem live · '
            'Groq for reasoning &amp; debate · Gemini for grounded verification &amp; refereeing</p></div>')

    with gr.Accordion("🔑 Keys & models", open=False):
        with gr.Row():
            groq_key = gr.Textbox(label="GROQ_API_KEY", type="password",
                                  value=os.environ.get("GROQ_API_KEY", ""), scale=2)
            gemini_key = gr.Textbox(label="GEMINI_API_KEY", type="password",
                                    value=os.environ.get("GEMINI_API_KEY", ""), scale=2)
            connect_btn = gr.Button("Connect / refresh models", variant="secondary", scale=1)
        with gr.Row():
            groq_model = gr.Dropdown(label="Groq model (debate)", choices=GROQ_PREFERRED,
                                     value=GROQ_PREFERRED[0], allow_custom_value=True)
            gemini_model = gr.Dropdown(label="Gemini model (research/referee)", choices=GEMINI_PREFERRED,
                                       value=GEMINI_PREFERRED[0], allow_custom_value=True)
        conn_status = gr.Markdown("_Not connected yet. Keys from the setup cell are pre-filled._")

    with gr.Row():
        problem = gr.Textbox(label="Your problem or vague idea", lines=3, scale=4,
                             placeholder="e.g. Cut peak electricity bills for a 40-flat building "
                                         "without a grid upgrade...")
        with gr.Column(scale=1):
            max_rounds = gr.Slider(1, 8, value=3, step=1, label="Max rounds")
            threshold = gr.Slider(50, 98, value=85, step=1, label="Consensus threshold")
            do_research = gr.Checkbox(value=True, label="Web research + citations (Gemini)")
    with gr.Row():
        start_btn = gr.Button("▶️  Start the debate", variant="primary", scale=3)
        clear_btn = gr.Button("🧹 Clear", scale=1)
    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[problem], label="Try a problem")

    banner = gr.HTML(render_banner(ArenaState()))
    with gr.Row():
        card_inv = gr.HTML(render_card("inventor", ArenaState()))
        card_sci = gr.HTML(render_card("scientist", ArenaState()))

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(**CHAT_KW)
        with gr.Column(scale=2):
            meters = gr.HTML(render_meters(ArenaState()))
            with gr.Tabs():
                with gr.Tab("🔬 Evidence"):
                    evidence_md = gr.Markdown("_No research performed yet._")
                with gr.Tab("∑ Equations"):
                    equations_md = gr.Markdown("_No equations yet._", latex_delimiters=LATEX)
                with gr.Tab("🕓 Timeline"):
                    timeline_md = gr.Markdown("_The idea has not evolved yet._")
                with gr.Tab("⚠ Conflicts"):
                    contra_md = gr.Markdown("_None outstanding._")

    with gr.Tabs():
        with gr.Tab("📐 Consensus Blueprint"):
            blueprint_md = gr.Markdown(render_blueprint(ArenaState()), latex_delimiters=LATEX)
            file_out = gr.File(label="Download blueprint (.md)", visible=False)
        with gr.Tab("🔎 Audit"):
            audit_md = gr.Markdown("_Independent audit runs after the blueprint is written._")

    OUTPUTS = [chatbot, card_inv, card_sci, banner, meters, evidence_md, equations_md,
               timeline_md, contra_md, blueprint_md, audit_md, file_out, start_btn]

    connect_btn.click(connect, [groq_key, gemini_key], [groq_model, gemini_model, conn_status])
    start_btn.click(lambda: gr.update(interactive=False), None, [start_btn]).then(
        start,
        [problem, max_rounds, threshold, do_research, groq_key, gemini_key, groq_model, gemini_model],
        OUTPUTS)
    clear_btn.click(reset, None, OUTPUTS)

print("UI built. Launch it in the next cell.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        rounds = int(os.environ.get("IFA_ROUNDS", "3"))
        g_, m_ = ensure_clients(os.environ.get("GROQ_API_KEY"), os.environ.get("GEMINI_API_KEY"), None, None)
        final = None
        seen = 0
        for _s in run_arena(" ".join(sys.argv[1:]), rounds, 85, g_, m_, do_research=True):
            final = _s
            while seen < len(_s.turns):
                _t = _s.turns[seen]; seen += 1
                who = "INVENTOR" if _t["who"] == "inventor" else "SCIENTIST"
                print(f"\n[Round {_t['round']}] {who}\n{'-' * 70}\n{_t['message']}")
        if final:
            print("\n" + "=" * 70)
            print(f"Agreement {final.agreement}/100 | Quality {final.quality}/100 | {final.stop_reason}")
            print("\n" + (final.blueprint or ""))
            print("\nSaved:", final.blueprint_path)
    else:
        demo.queue(default_concurrency_limit=4, max_size=16).launch(**LAUNCH_KW)
