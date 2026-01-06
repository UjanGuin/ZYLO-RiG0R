#!/usr/bin/env python3
from zhipuai import ZhipuAI
import os
import re
import json
import time
import math
import shlex
import uuid
import signal
import atexit
import logging
import tempfile
import traceback
import subprocess
from typing import Optional, Tuple, Dict, Any

from flask import Flask, request, jsonify, Response, send_from_directory

# Replace Cerebras import with a safe import guard
try:
    from cerebras.cloud.sdk import Cerebras
except Exception:
    Cerebras = None

# -------------------------
# Configuration (env-first)
# -------------------------
API_KEY = "paste_your_api_key"
MODEL_NAME = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
GLM_API_KEY = "paste_your_api_key"
GLM_MODEL = "glm-4.7"
PORT = int(os.getenv("PORT", "5005"))
DATA_DIR = os.getenv("OSS_SERVER_DATA", "./oss_server_data")
SESSION_DIR = os.path.join(DATA_DIR, "sessions")
LOGFILE = os.path.join(DATA_DIR, "server.log")

glm_client = ZhipuAI(api_key=GLM_API_KEY)
# Research-Grade Phase Temperatures
# ---------------------------------
# 1. Planning: High entropy to explore solution space (avoid CoT loops)
TEMP_PLANNING = 0.67
# 2. Execution: Moderate constraint for tool interaction
TEMP_EXECUTION = 0.0
# 3. Exposition: Strict determinism for final proofs & numbers
TEMP_EXPOSITION = 0.03
# ---------------------------------
# GLM-4.7 Confidence Calibration
# ---------------------------------
CONFIDENCE_BY_VERDICT = {
    "correct": 0.97,     # Independently verified
    "uncertain": 0.65,   # Logical ambiguity / incomplete proof
    "incorrect": 0.25    # Explicit mathematical error
}


if not API_KEY:
    raise RuntimeError("Set CEREBRAS_API_KEY in environment before running this server.")

# Ensure directories exist
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("oss_server")

# -------------------------
# Create Cerebras client
# -------------------------
if Cerebras is None:
    raise RuntimeError("Cerebras SDK not available. pip install cerebras_cloud_sdk")
client = Cerebras(api_key=API_KEY)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_folder="static")

# -------------------------
# System prompt (Hybrid: Ph.D. Persona + Strict Tooling)
# -------------------------

SYSTEM_PROMPT = r"""
You are GPT-OSS-120B, a Ph.D.-level research assistant in mathematics and physics.

====================
MODES OF OPERATION
====================

You operate in TWO mutually exclusive modes.

--------------------------------
MODE A — TOOL MODE (MANDATORY)
--------------------------------
Trigger TOOL MODE PREFERENTIALLY (not mandatorily) when verified computation is required:
- numerical computation
- algebraic manipulation
- calculus evaluation
- statistics
- physics calculations
- data verification

In TOOL MODE:
1. DO NOT explain anything in natural language.
2. DO NOT use Markdown.
3. DO NOT include LaTeX.
4. Respond with PURE JSON ONLY, starting at the first character.

The ONLY valid response format is:

{
  "tool": "python",
  "code": "<complete executable python code>",
  "verify": "sympy" | "numeric" | "none",
  "explain_after": true
}

Rules for TOOL MODE:
- The JSON must be valid and standalone.
- The code must run without placeholders.
- If verify = "sympy", the code MUST print a JSON object with keys:
  - "status"
  - "result"

--------------------------------
MODE B — DIRECT ANSWER MODE
--------------------------------
Trigger this mode ONLY if the problem is:
- purely theoretical
- proof-based
- conceptual
- contains no calculations requiring verification

In DIRECT ANSWER MODE:

FORMATTING RULES (STRICT):
- Use Markdown for headings, paragraphs, and lists.
- Use LaTeX ONLY for mathematical expressions.
- Inline math: \( ... \)
- Display math: \[ ... \]

ABSOLUTE PROHIBITIONS:
- NEVER place text inside \( \) or \[ \].
- NEVER use \text{}, \textbf{}, \mathrm{} for prose.
- NEVER use aligned, array, cases, enumerate, or itemize.
- NEVER produce a full LaTeX document.

--------------------------------
FINAL OUTPUT RULE
--------------------------------
- TOOL MODE → JSON ONLY
- DIRECT ANSWER MODE → Markdown + LaTeX (math only)

Violating these rules is a critical failure.
""".strip()

# -------------------------
# Utility helpers
# -------------------------
def safe_filename(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', s)

def now_iso() -> str:
    return time.strftime("%Y%m%dT%H%M%S")

def save_session_history(session_id: str, data: dict):
    path = os.path.join(SESSION_DIR, f"{safe_filename(session_id)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_session_history(session_id: str) -> dict:
    path = os.path.join(SESSION_DIR, f"{safe_filename(session_id)}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    base = {
        "id": session_id,
        "created": now_iso(),
        "messages": [{"role":"system","content":SYSTEM_PROMPT}],
    }
    save_session_history(session_id, base)
    return base

def glm_proof_check(question: str, answer: str) -> dict:
    """
    Independent mathematical verification using GLM-4.7.
    Returns verdict, error location, and correction guidance.
    """

    prompt = f"""
You are an independent mathematical proof checker.

PROBLEM:
{question}

PROPOSED FINAL ANSWER:
{answer}

TASK:
1. Verify mathematical correctness.
2. If incorrect, identify:
   - The exact incorrect step
   - Why it is incorrect
   - What the correct approach should be
3. If correct, confirm rigor.

Respond STRICTLY in JSON:
{{
  "verdict": "correct" | "incorrect" | "uncertain",
  "error_step": "<exact faulty step or 'none'>",
  "reason": "<clear explanation>",
  "fix": "<how it should be done correctly>"
}}
"""

    resp = glm_client.chat.completions.create(
        model=GLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "verdict": "uncertain",
            "error_step": "unknown",
            "reason": "GLM response not parseable",
            "fix": "unknown"
        }

def should_fallback_to_fast(text: str) -> bool:
    """
    Detect problems that mix calculation with explanation
    where forcing tools causes deadlock.
    """
    if not isinstance(text, str):
        return False

    t = text.lower()

    explanation_signals = [
        "explain", "derive", "prove", "show that",
        "find the locus", "which of the following",
        "explain why", "reason", "justify"
    ]

    has_explanation = any(k in t for k in explanation_signals)
    needs_calc = requires_tool(text)

    return has_explanation and needs_calc

# -------------------------
# Sandboxed Python executor
# -------------------------
def run_python_sandbox(code: str, timeout_s: int = 15) -> Tuple[str, str, int]:
    """
    Runs 'code' in a temporary python file with resource limits.
    """
    import resource
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        fname = f.name
        f.write("# -*- coding: utf-8 -*-\n")
        f.write("import json,sys,math\n")
        f.write(code)

    def preexec():
        try:
            # CPU Limit
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s + 1))
            # Memory Limit (1GB)
            mem = 1024 * 1024 * 1024 
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except: pass

    proc = subprocess.Popen(
        ["python3", fname],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=preexec
    )
    try:
        out, err = proc.communicate(timeout=timeout_s + 2)
        return out, err, proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        return "", "TimeoutExpired", -9
    except Exception as e:
        return "", str(e), -8

# -------------------------
# SymPy runner (from ooo.py)
# -------------------------
def run_sympy_check(expr_code: str, timeout_s: int = 8) -> Tuple[str, str, int]:
    """
    Wraps code in a SymPy safety harness to catch errors and format output as JSON.
    """
    wrapper = f"""
import json
import traceback
from fractions import Fraction
from sympy import symbols, simplify, Eq, solve, diff, integrate, limit, expand, factor
from sympy.abc import x, y, z, a, b, c, n

try:
    # User code goes here
{expr_code}
    # Implicitly assume user defined 'result'
    if 'result' not in locals():
        print(json.dumps({{"status": "error", "error": "No 'result' variable defined in scope"}}))
    else:
        out = {{}}
        out['status'] = 'ok'
        out['result'] = str(result) # Convert sympy objects to string
        print(json.dumps(out))
except Exception as e:
    print(json.dumps({{"status":"error", "error": str(e), "trace": traceback.format_exc()}}))
"""
    return run_python_sandbox(wrapper, timeout_s=timeout_s)

# -------------------------
# Model Call Wrapper
# -------------------------
def call_model_with_history(
    history: list, 
    reasoning_level: str = "high", 
    temperature: float = 0.5
) -> str:
    """
    Executes the API call with strict phase-aware temperature and reasoning control.
    """
    if len(history) > 80:
        history = [history[0]] + history[-60:]
    
    logger.info(f"API Call | Reasoning: {reasoning_level} | Temp: {temperature}")

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        max_tokens=4096,
        temperature=temperature,
        reasoning_effort=reasoning_level
    )
    return resp.choices[0].message.content

# -------------------------
# Tool Parsing
# -------------------------
def parse_tool_instruction(text: str):
    if not isinstance(text, str):
        return None

    text = text.strip()

    # Hard guard: must start with JSON
    if not text.startswith("{"):
        return None

    try:
        obj = json.loads(text)
        if obj.get("tool") == "python":
            return obj
    except Exception:
        return None

    return None


# -------------------------
# Session Manager
# -------------------------
class ResearchSession:
    def __init__(self, session_id: str):
        self.id = session_id
        self._data = load_session_history(self.id)
        self.history = self._data.get("messages", [])
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, {"role":"system","content":SYSTEM_PROMPT})

    def append_user(self, text: str):
        self.history.append({"role":"user","content":text})
        self._save()

    def append_assistant(self, text: str):
        self.history.append({"role":"assistant","content":text})
        self._save()

    def _save(self):
        self._data["messages"] = self.history
        save_session_history(self.id, self._data)

# -------------------------
# Core Logic: Phase-Aware Handling
# -------------------------
def wrap_pure_math(text: str) -> str:
    if not isinstance(text, str):
        return "_[Invalid response]_"

    stripped = text.strip()
    if not stripped:
        return "_[Empty response]_"

    # Detect pure LaTeX (no explanatory text)
    if stripped.startswith("\\[") or stripped.startswith("\\("):
        return "### Result\n\n" + text

    return text

def strip_tool_json(text: str) -> str:
    if not isinstance(text, str):
        return ""

    stripped = text.strip()

    # If model leaked tool JSON, suppress it entirely
    if stripped.startswith("{") and '"tool"' in stripped:
        return ""

    return text

def requires_tool(text: str) -> bool:
    """
    Aggressive heuristic detector for problems that REQUIRE
    verified computation or numerical confirmation.
    Designed to favor tool usage in research mode.
    """

    if not isinstance(text, str):
        return False

    t = text.lower()

    # -------------------------------------------------
    # 1. STRONG NUMERIC / COMPUTATIONAL TRIGGERS
    # -------------------------------------------------
    numeric_keywords = [
        "compute", "calculate", "evaluate", "find value",
        "numerically", "approximate", "decimal",
        "verify", "verification", "check numerically",
        "solve", "solution", "roots", "zeroes",
        "determinant", "det", "eigenvalue", "eigenvalues",
        "matrix", "polynomial", "characteristic",
        "integral", "differentiate", "derivative",
        "limit", "series", "summation", "product",
        "probability", "expectation", "variance",
        "time taken", "fall time", "oscillation",
        "density", "field", "potential",
        "energy", "momentum", "force",
        "trajectory", "motion", "dynamics",
        "frequency", "angular frequency",
        "wavefunction", "schrodinger",
        "laplace", "fourier", "z-transform",
        "numerical method", "newton method",
        "iteration", "convergence"
    ]

    if any(k in t for k in numeric_keywords):
        return True

    # -------------------------------------------------
    # 2. SYMBOLIC / STRUCTURAL SIGNALS
    # -------------------------------------------------
    symbolic_signals = [
        "=", "^", "*", "/", "+", "-", 
        "[[", "]]", "(", ")", "{", "}",
        "dx", "dt", "∫", "∑", "∏"
    ]

    if any(s in t for s in symbolic_signals):
        return True

    # -------------------------------------------------
    # 3. DEGREE / SCALE HEURISTICS
    # -------------------------------------------------
    if re.search(r"\bdegree\s*[4-9]\b", t):
        return True

    if re.search(r"x\^\d{2,}", t):  # high-degree polynomial
        return True

    # -------------------------------------------------
    # 4. LARGE NUMERIC DATA
    # -------------------------------------------------
    if re.search(r"\d{4,}", t):  # large integers
        return True

    if re.search(r"\b\d+\.\d+\b", t):  # decimals
        return True

    return False


def extract_strict_json(text: str):
    if not isinstance(text, str):
        return None

    text = text.strip()

    # Must start with {
    if not text.startswith("{"):
        return None

    try:
        return json.loads(text)
    except Exception:
        return None

def requires_numeric_tool(text: str) -> bool:
    keywords = [
        "calculate", "compute", "numerical", "value of",
        "evaluate", "data", "probability", "statistics",
        "mean", "variance", "time taken", "integrate numerically"
    ]
    return any(k in text.lower() for k in keywords)


def allows_analytic_solution(text: str) -> bool:
    keywords = [
        "prove", "show that", "derive", "find the coefficient",
        "rolling", "without slipping", "using conservation",
        "by symmetry", "angular momentum"
    ]
    return any(k in text.lower() for k in keywords)


def has_visible_content(text: str) -> bool:
    if not isinstance(text, str):
        return False

    # Remove whitespace and math blocks
    stripped = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    stripped = re.sub(r"\\\(.*?\\\)", "", stripped, flags=re.DOTALL)

    return bool(stripped.strip())

def force_tool_result(tool_output: dict) -> str:
    """
    Guarantees visible output from tool execution.
    Used ONLY if model exposition is empty or generic.
    """
    if not tool_output:
        return ""

    stdout = (tool_output.get("stdout") or "").strip()
    stderr = (tool_output.get("stderr") or "").strip()

    if stdout:
        return (
            "### Result\n\n"
            "**Computed values:**\n\n"
            f"```\n{stdout}\n```"
        )

    if stderr:
        return (
            "### Result\n\n"
            "**Tool reported:**\n\n"
            f"```\n{stderr}\n```"
        )

    return ""

def handle_message(
    session: ResearchSession,
    user_text: str,
    reasoning_level: str,
    recheck: bool = False
) -> dict:
    """
    Research-grade message handler with:
    - Deterministic tool enforcement
    - Guaranteed result exposition
    - NO silent success in Research mode
    - NO empty replies
    """

    # =========================================================
    # USER INPUT
    # =========================================================
    session.append_user(user_text)

    # =========================================================
    # PHASE 1: PLANNING / INITIAL RESPONSE
    # =========================================================
    assistant_raw = call_model_with_history(
        session.history,
        reasoning_level=reasoning_level,
        temperature=TEMP_PLANNING
    )

    # =========================================================
    # TOOL DETECTION
    # =========================================================
    tool_inst = extract_strict_json(assistant_raw)

    # ---------- HARD TOOL RETRY (STRICT JSON) ----------
    if (
        tool_inst is None
        and reasoning_level == "high"
        and isinstance(assistant_raw, str)
        and "tool" in assistant_raw.lower()
    ):
        logger.warning("Tool response malformed. Forcing strict JSON retry.")

        session.append_user(
            "TOOL MODE VIOLATION.\n"
            "Respond AGAIN with ONLY valid JSON.\n"
            "No text. No Markdown. JSON must start at character 1."
        )

        assistant_raw = call_model_with_history(
            session.history,
            reasoning_level=reasoning_level,
            temperature=0.1
        )

        tool_inst = extract_strict_json(assistant_raw)

    # ---------- FORCED TOOL REPHRASE ----------
    if (
        reasoning_level == "high"
        and tool_inst is None
        and requires_tool(user_text)
    ):
        # 🔁 NEW: mixed explanation + calculation → FAST fallback
        if should_fallback_to_fast(user_text):
            logger.warning("Mixed calc + explanation detected. Falling back to FAST mode.")

            fast_answer = call_model_with_history(
                session.history,
                reasoning_level="low",      # FAST
                temperature=0.7
            )

            visible = fast_answer.strip()
            visible = wrap_pure_math(visible)

            session.append_assistant(visible)

            return {
                "reply": visible,
                "raw": fast_answer,
                "tool_output": None,
                "fallback": "fast"
            }

        # Otherwise: enforce tool as before
        logger.warning("Tool required but not used. Forcing tool-only rewrite.")

        session.append_user(
            "This problem REQUIRES verified computation.\n"
            "Rewrite your response as a Python tool invocation.\n"
            "Return ONLY strict JSON.\n"
            "Do NOT explain yet."
        )

        assistant_raw = call_model_with_history(
            session.history,
            reasoning_level=reasoning_level,
            temperature=0.2
        )

        tool_inst = extract_strict_json(assistant_raw)

    # =========================================================
    # 🚨 HARD FAIL (THE CRITICAL FIX)
    # =========================================================
    if (
        reasoning_level == "high"
        and requires_tool(user_text)
        and tool_inst is None
    ):
        logger.error("RESEARCH FAILURE: Tool required but not produced.")

        failure_msg = (
            "### ❌ Research Failed\n\n"
            "**Reason:** This problem requires verified computation, "
            "but no valid tool invocation was produced.\n\n"
            "**Action:** Retry, simplify the question, or switch to Fast mode."
        )

        session.append_assistant(failure_msg)

        return {
            "reply": failure_msg,
            "raw": assistant_raw,
            "tool_output": None,
            "error": "tool_required_but_missing"
        }

    # =========================================================
    # INIT OUTPUT HOLDERS
    # =========================================================
    tool_output = None
    proof_check = None

    # =========================================================
    # CASE A: TOOL-BASED PATH
    # =========================================================
    if tool_inst:
        code = tool_inst.get("code", "")
        verify_mode = tool_inst.get("verify", "none")

        # -------------------------
        # PHASE 2: EXECUTION
        # -------------------------
        if verify_mode == "sympy":
            logger.info("Running SymPy verification...")
            stdout, stderr, rc = run_sympy_check(code)
        else:
            logger.info("Running standard Python...")
            stdout, stderr, rc = run_python_sandbox(code)

        tool_output = {
            "stdout": stdout or "",
            "stderr": stderr or "",
            "rc": rc,
            "verify": verify_mode
        }

        # -------------------------
        # PHASE 3: EXPOSITION
        # -------------------------
        session.append_user(
            "The computation has completed.\n\n"
            "RAW TOOL OUTPUT:\n"
            f"{stdout[:12000]}\n\n"
            "INSTRUCTIONS:\n"
            "- Explicitly state the final results.\n"
            "- Briefly explain the verification.\n"
            "- Do NOT say 'computed successfully'.\n"
            "- Do NOT rerun tools."
        )

        final_answer = call_model_with_history(
            session.history,
            reasoning_level=reasoning_level,
            temperature=TEMP_EXPOSITION
        )

        annotated_answer = strip_tool_json(final_answer).strip()

        # ---------- HARD GUARANTEE: RESULTS MUST APPEAR ----------
        if not annotated_answer or "computed successfully" in annotated_answer.lower():
            logger.warning("Exposition missing results. Injecting tool output.")

            if stdout.strip():
                annotated_answer = (
                    "### Result\n\n"
                    f"```\n{stdout.strip()}\n```"
                )
            else:
                annotated_answer = (
                    "### Result\n\n"
                    "_Computation completed, but no explicit values were returned._"
                )

        annotated_answer = wrap_pure_math(annotated_answer)

        # -------------------------
        # PHASE 4: OPTIONAL RECHECK
        # -------------------------
        if recheck:
            logger.info("Running GLM recheck...")
            proof_check = glm_proof_check(user_text, annotated_answer)

            verdict = proof_check.get("verdict", "uncertain")
            confidence = CONFIDENCE_BY_VERDICT.get(verdict, 0.6)

            annotated_answer += (
                "\n\n---\n"
                f"**Proof Recheck:** `{verdict.upper()}`  \n"
                f"**Confidence:** {confidence}"
            )

        session.append_assistant(annotated_answer)

        return {
            "reply": annotated_answer,
            "raw": assistant_raw,
            "tool_output": tool_output,
            "proof_check": proof_check
        }

    # =========================================================
    # CASE B: NO TOOL (FAST / ANALYTICAL ONLY)
    # =========================================================
    visible_answer = strip_tool_json(assistant_raw).strip()

    if not visible_answer:
        visible_answer = (
            "### ❌ No Result\n\n"
            "No valid answer was produced."
            if reasoning_level == "high"
            else "### Result\n\n_Response generated successfully._"
        )

    visible_answer = wrap_pure_math(visible_answer)
    session.append_assistant(visible_answer)

    return {
        "reply": visible_answer,
        "raw": assistant_raw,
        "tool_output": None
    }




# -------------------------------------------------------
# Canonical Math Normalizer (MathJax-safe, Regex-limited)
# -------------------------------------------------------

INLINE = r"\\\((.*?)\\\)"
BLOCK  = r"\\\[(.*?)\\\]"

RAW_MATH_HINT = re.compile(
    r"(?<!\\)"
    r"("
    r"[a-zA-Z]\s*\^\s*\{?\d+\}?"          # x^2, y^{2}
    r"|\\frac\s*\{.*?\}\s*\{.*?\}"        # \frac{a}{b}
    r"|\\sqrt\s*\{.*?\}"                  # \sqrt{3}
    r"|[a-zA-Z0-9]+\s*=\s*[a-zA-Z0-9]+"   # y^2 = x
    r"|[a-zA-Z]+_[a-zA-Z0-9]+"            # F_AB
    r")"
)

def tool_exposition_invalid(answer: str, tool_stdout: str) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return True

    low_info_phrases = [
        "computed successfully",
        "computation completed",
        "result computed",
        "calculation complete",
        "done",
        "finished"
    ]

    ans = answer.lower()

    # 1️⃣ Generic acknowledgments
    if any(p in ans for p in low_info_phrases):
        return True

    # 2️⃣ Tool produced numbers but answer has none
    has_numbers_in_tool = any(ch.isdigit() for ch in tool_stdout)
    has_numbers_in_answer = any(ch.isdigit() for ch in answer)

    if has_numbers_in_tool and not has_numbers_in_answer:
        return True

    # 3️⃣ Roots / eigenvalues / determinant mentioned but not stated
    critical_words = ["root", "roots", "eigen", "determinant", "value"]
    if any(w in ans for w in critical_words) and not has_numbers_in_answer:
        return True

    return False


def strip_text_from_math_blocks(text: str) -> str:
    if not isinstance(text, str):
        return ""

    extracted_text = []

    def clean_math_block(match):
        block = match.group(1)

        # If this is prose disguised as math → extract text
        if re.search(r"\\text", block):
            # Remove \text{} wrappers
            plain = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", block)
            plain = re.sub(r"\\text\{([^}]*)\}", r"\1", plain)
            extracted_text.append(plain)
            return ""  # remove math wrapper

        # Otherwise keep real math
        return f"\\[{block}\\]"

    text = re.sub(
        r"\\\[(.*?)\\\]",
        clean_math_block,
        text,
        flags=re.DOTALL
    )

    # Append extracted prose back
    if extracted_text:
        text = "\n\n".join(extracted_text) + "\n\n" + text

    return text

def is_proof_only(text: str) -> bool:
    if not isinstance(text, str):
        return False

    # Indicators of proof-style reasoning
    keywords = [
        "there exists", "suppose", "assume", "by contradiction",
        "by mean value theorem", "hence", "therefore", "let"
    ]

    return any(k in text.lower() for k in keywords)


def strip_latex_lists(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove enumerate/itemize blocks entirely
    text = re.sub(
        r"\\begin\{(enumerate|itemize)\}[\s\S]*?\\end\{\1\}",
        "",
        text
    )

    # Remove stray \item
    text = re.sub(r"\\item\s*", "- ", text)

    # Remove alignment markers that break MathJax
    text = re.sub(r"(?<!\\)&", "", text)

    return text

def strip_latex_math_layout(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove aligned / array environments (used incorrectly by LLMs)
    text = re.sub(
        r"\\begin\{(aligned|array|cases)\}[\s\S]*?\\end\{\1\}",
        "",
        text
    )

    # Remove standalone \text{} used outside inline math
    text = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)

    # Remove spacing commands that pollute output
    text = re.sub(r"\\(quad|qquad|hfill|vspace\{.*?\})", "", text)

    return text

def normalize_math(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # -----------------------------
    # 1. Protect valid LaTeX blocks
    # -----------------------------
    protected = []

    def _protect(m):
        protected.append(m.group(0))
        return f"@@MATH{len(protected)-1}@@"

    # Protect MathJax-style blocks first
    text = re.sub(BLOCK, _protect, text, flags=re.S)
    text = re.sub(INLINE, _protect, text, flags=re.S)

    # -----------------------------
    # 1.5 EXTRA PROTECTION
    # Protect common environments if they appear
    # -----------------------------
    text = re.sub(
        r"\\begin\{(aligned|array|cases|matrix|pmatrix|bmatrix)\}.*?\\end\{\1\}",
        _protect,
        text,
        flags=re.S
    )

    # -----------------------------
    # 2. Fix common model mistakes
    # -----------------------------
    # \mathbf{F}{AB} → \mathbf{F}_{AB}
    text = re.sub(
        r"\\mathbf\{([A-Za-z])\}\s*\{([A-Za-z]+)\}",
        r"\\mathbf{\1}_{\2}",
        text
    )

    # \vec{v}{t} → \vec{v}(t)
    text = re.sub(
        r"\\vec\{([A-Za-z])\}\s*\{([A-Za-z]+)\}",
        r"\\vec{\1}(\2)",
        text
    )

    # -----------------------------
    # 3. Wrap raw math in \( ... \)
    # (ONLY when math symbols are detected)
    # -----------------------------
    def wrap_raw_math(m):
        expr = m.group(1).strip()
        return rf"\( {expr} \)"

    text = RAW_MATH_HINT.sub(wrap_raw_math, text)

    # -----------------------------
    # 3.5 EXTRA RAW-MATH CATCH
    # Detect orphaned equations like: a = b + c
    # -----------------------------
    text = re.sub(
        r"(?<!\\)\b([A-Za-z]\w*\s*=\s*[^.\n]+)",
        lambda m: rf"\( {m.group(1).strip()} \)",
        text
    )

    # -----------------------------
    # 4. Restore protected math
    # -----------------------------
    for i, m in enumerate(protected):
        text = text.replace(f"@@MATH{i}@@", m)

    # -----------------------------
    # 5. De-duplicate wrappers
    # -----------------------------
    text = re.sub(r"\\\(\s*\\\(", r"\\(", text)
    text = re.sub(r"\\\)\s*\\\)", r"\\)", text)
    text = re.sub(r"\\\[\s*\\\[", r"\\[", text)
    text = re.sub(r"\\\]\s*\\\]", r"\\]", text)

    # -----------------------------
    # 6. Final safety cleanup
    # -----------------------------
    # Remove empty math blocks
    text = re.sub(r"\\\(\s*\\\)", "", text)
    text = re.sub(r"\\\[\s*\\\]", "", text)

    return text.strip()


# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return Response(HTML_PAGE, mimetype="text/html")

@app.route("/chat", methods=["POST"])
def chat_route():
    try:
        data = request.json or {}

        sess_id = (
            data.get("session_id")
            or request.cookies.get("session_id")
            or str(uuid.uuid4())
        )

        msg = data.get("message", "").strip()

        # UI controls
        r_level = data.get("reasoning_level", "high")
        recheck = bool(data.get("recheck", False))

        if not msg:
            return jsonify({"error": "empty"}), 400

        sess = ResearchSession(sess_id)

        # -------------------------
        # MAIN PIPELINE
        # -------------------------
        res = handle_message(
            sess,
            msg,
            r_level,
            recheck=recheck
        )

        reply = res.get("reply", "")

        # -------------------------
        # HARD GUARANTEE: NEVER EMPTY
        # -------------------------
        if not isinstance(reply, str) or not reply.strip():
            reply = (
                "### Result\n\n"
                "_Response generated successfully._"
            )

        # -------------------------
        # POST-PROCESSING RULES
        # -------------------------
        tool_used = bool(res.get("tool_output"))

        if not tool_used:
            # DO NOT touch proofs
            if not is_proof_only(reply):
                reply = strip_latex_lists(reply)
                # ❌ strip_latex_math_layout REMOVED (critical)

        # ✅ Normalize math ONCE, LAST
        reply = normalize_math(reply)

        if not reply.strip():
            reply = (
                "### Result\n\n"
                "_Response generated successfully._"
            )

        res["reply"] = reply

        resp = jsonify(res)
        resp.set_cookie("session_id", sess.id, httponly=True)
        return resp

    except Exception as e:
        logger.exception("Chat Error")
        return jsonify({"error": str(e)}), 500


# -------------------------
# UI
# -------------------------

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ZYLO-RIGOR | Research Engine</title>

    <!-- Responsive -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <!-- Markdown -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

    <!-- MathJax -->
    <script>
        window.MathJax = {
            tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] },
            svg: { fontCache: 'global' }
        };
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

    <style>
        :root {
            --bg-dark: #050509;
            --bg-deep: #020205;
            --panel: rgba(15, 23, 42, 0.6);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(255, 255, 255, 0.15);
            --font-main: 'Inter', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }

        body {
            margin: 0;
            background: radial-gradient(circle at top center, #111425 0%, var(--bg-deep) 80%);
            color: var(--text-main);
            font-family: var(--font-main);
            height: 100dvh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* ---------- SPLASH SCREEN ---------- */
        #splash {
            position: fixed;
            inset: 0;
            background: #020205;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            transition: opacity 0.8s ease, visibility 0.8s;
        }

        .splash-content {
            text-align: center;
            animation: slideUpFade 1.2s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .splash-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }

        .splash-sub {
            color: var(--text-muted);
            font-size: 1rem;
            opacity: 0.8;
        }

        @keyframes slideUpFade {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* ---------- HEADER ---------- */
        header {
            padding: 1rem 1.5rem;
            background: rgba(5, 5, 9, 0.8);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            z-index: 10;
        }

        .brand {
            font-weight: 700;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: -0.5px;
        }

        .brand span { color: var(--accent); }

        .controls {
            display: flex;
            gap: 0.8rem;
            align-items: center;
        }

        /* --- Styled pill appearance kept, native select replaced by custom component --- */
        .pill-select, .toggle {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-family: var(--font-main);
            cursor: pointer;
            transition: all 0.2s ease;
            -webkit-appearance: none;
            appearance: none;
        }

        .pill-select:hover, .toggle:hover {
            border-color: var(--border-highlight);
            background: rgba(255,255,255,0.06);
        }

        .toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            user-select: none;
        }

        .toggle input {
            accent-color: var(--accent);
            cursor: pointer;
            width: 1rem;
            height: 1rem;
        }

        /* ---------- CUSTOM SELECT (replaces native <select>) ---------- */
        .custom-select {
            position: relative;
            display: inline-block;
            min-width: 120px;
            font-family: var(--font-main);
        }

        .select-btn {
            display: inline-flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            width: 140px;
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.03);
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.85rem;
            text-align: left;
        }

        .select-btn:focus {
            outline: none;
            box-shadow: 0 6px 16px rgba(0,0,0,0.35);
            color: var(--text-main);
            border-color: var(--border-highlight);
        }

        .select-btn .caret {
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 7px solid var(--text-muted);
            opacity: 0.9;
        }

        .select-btn.open .caret {
            transform: rotate(180deg);
            border-top-color: var(--text-main);
        }

        .select-options {
            position: absolute;
            right: 0;
            left: 0;
            margin-top: 8px;
            background: linear-gradient(180deg, rgba(10,10,12,0.98), rgba(8,8,10,0.98));
            border: 1px solid var(--border);
            border-radius: 10px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
            list-style: none;
            padding: 6px;
            display: none;
            z-index: 50;
            max-height: 240px;
            overflow-y: auto;
        }

        .select-options.open { display: block; }

        .select-options li {
            padding: 10px 12px;
            margin: 4px 0;
            border-radius: 8px;
            cursor: pointer;
            color: var(--text-main);
            font-weight: 600;
            font-size: 0.95rem;
        }

        .select-options li:hover,
        .select-options li:focus {
            background: rgba(99,102,241,0.08);
            color: var(--text-main);
            outline: none;
        }

        .select-options li.selected {
            background: rgba(99,102,241,0.12);
            color: var(--accent);
            box-shadow: inset 0 0 0 1px rgba(99,102,241,0.06);
        }

        /* ---------- CHAT AREA ---------- */
        #chat {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            scroll-behavior: smooth;
        }

        .msg {
            max-width: 800px;
            margin: 0 auto 1.5rem auto;
            display: flex;
            flex-direction: column;
            animation: msgEnter 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            opacity: 0;
            transform: translateY(10px);
        }

        @keyframes msgEnter {
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 6px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* AI label — prominent */
        .assistant .msg-label { 
            color: var(--accent);
            font-size: 1.35rem;
            font-weight: 900;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        /* hide the user label (no "You") */
        .user .msg-label {
            display: none;
        }

        .msg-content {
            padding: 1.25rem 1.5rem;
            border-radius: 16px;
            line-height: 1.7;
            font-size: 0.95rem;
            position: relative;
            word-wrap: break-word;
        }

        .user { align-items: flex-end; }
        .user .msg-content {
            background: linear-gradient(135deg, #2e3558 0%, #1e293b 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        .assistant { align-items: flex-start; }
        .assistant .msg-content {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            border-bottom-left-radius: 4px;
        }

        /* Tool Output Box */
        .tool-output {
            font-family: var(--font-mono);
            font-size: 0.8rem;
            background: #09090b;
            color: #4ade80;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 0.8rem;
            border: 1px solid #222;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        /* ---------- TYPING INDICATOR ---------- */
        #typing-indicator {
            display: none;
            max-width: 800px;
            margin: 0 auto 1rem auto;
            padding-left: 0.5rem;
        }
        
        .typing-bubble {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            padding: 12px 20px;
            border-radius: 16px;
            border-bottom-left-radius: 4px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        
        .typing-text {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-right: 8px;
            font-weight: 500;
        }

        .dot {
            width: 6px;
            height: 6px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .dot:nth-child(2) { animation-delay: -0.32s; }
        .dot:nth-child(3) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        /* ---------- INPUT AREA ---------- */
        #input-area {
            padding: 1.2rem;
            background: rgba(5,5,9,0.9);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            position: relative;
            z-index: 20;
        }

        #prompt {
            flex: 1;
            padding: 14px 18px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(20, 20, 25, 0.6);
            color: var(--text-main);
            font-size: 1rem;
            font-family: var(--font-main);
            outline: none;
            transition: all 0.2s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }

        #prompt:focus {
            border-color: var(--accent);
            background: rgba(20, 20, 25, 0.9);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }

        button {
            padding: 0 24px;
            border-radius: 12px;
            border: none;
            background: var(--accent);
            color: white;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: transform 0.1s ease, filter 0.2s;
            box-shadow: 0 4px 12px var(--accent-glow);
        }

        button:active { transform: scale(0.96); }
        button:hover { filter: brightness(1.1); }
        button:disabled { opacity: 0.6; cursor: not-allowed; }

        /* ---------- MOBILE ---------- */
        @media (max-width: 600px) {
            header { padding: 0.8rem 1rem; }
            .brand { font-size: 1rem; }
            /* show our custom select's button but reduce size */
            .select-btn { width: 120px; font-size: 0.82rem; padding: 8px 10px; }
            .msg-content { padding: 1rem 1.2rem; font-size: 0.9rem; }
            #input-area { padding: 1rem; }
            #prompt { font-size: 16px; } /* Prevent iOS zoom */
            .assistant .msg-label { font-size: 1.1rem; } /* scale AI label on small screens */
        }
    </style>
</head>

<body>

<div id="splash">
    <div class="splash-content">
        <div class="splash-title">Welcome!</div>
        <div class="splash-sub">Initializing Research Environment...</div>
    </div>
</div>

<header>
    <div class="brand">⚡ ZYLO <span>RIGOR</span></div>

    <div class="controls">
        <!-- Hidden input preserves existing JS expectations (reasoningSelect.value) -->
        <input type="hidden" id="reasoning" value="high">

        <!-- Custom visually styled select -->
        <div class="custom-select" id="customReasoning" tabindex="0" aria-haspopup="listbox" aria-expanded="false">
            <button type="button" class="select-btn" id="reasoningBtn" aria-haspopup="listbox" aria-expanded="false">
                Research
                <span class="caret"></span>
            </button>
            <ul class="select-options" id="reasoningOptions" role="listbox" aria-labelledby="reasoningBtn">
                <li role="option" data-value="low">Fast</li>
                <li role="option" data-value="medium">Balanced</li>
                <li role="option" data-value="high" class="selected">Research</li>
            </ul>
        </div>

        <label class="toggle">
            <input type="checkbox" id="recheckToggle">
            <span>Recheck</span>
        </label>
    </div>
</header>

<div id="chat"></div>

<!-- Typing Indicator -->
<div id="typing-indicator">
    <div class="typing-bubble">
        <span class="typing-text">Typing</span>
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
    </div>
</div>

<div id="input-area">
    <input id="prompt" placeholder="Ask a complex question..." autocomplete="off">
    <button onclick="ask()" id="sendBtn">Send</button>
</div>

<script>
    const chat = document.getElementById('chat');
    const input = document.getElementById('prompt');
    const sendBtn = document.getElementById('sendBtn');

    // reasoningSelect now a hidden input (keeps existing code)
    const reasoningSelect = document.getElementById('reasoning');
    const recheckToggle = document.getElementById('recheckToggle');
    const typingIndicator = document.getElementById('typing-indicator');

    // Custom select elements
    const reasoningBtn = document.getElementById('reasoningBtn');
    const reasoningOptions = document.getElementById('reasoningOptions');
    const customReasoning = document.getElementById('customReasoning');

    // open/close helper
    function closeReasoning() {
        reasoningOptions.classList.remove('open');
        reasoningBtn.classList.remove('open');
        customReasoning.setAttribute('aria-expanded', 'false');
        reasoningBtn.setAttribute('aria-expanded', 'false');
    }
    function openReasoning() {
        reasoningOptions.classList.add('open');
        reasoningBtn.classList.add('open');
        customReasoning.setAttribute('aria-expanded', 'true');
        reasoningBtn.setAttribute('aria-expanded', 'true');
    }

    // Toggle on button click
    reasoningBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (reasoningOptions.classList.contains('open')) closeReasoning();
        else openReasoning();
    });

    // Click outside closes
    document.addEventListener('click', (e) => {
        if (!customReasoning.contains(e.target)) closeReasoning();
    });

    // Option click handler
    reasoningOptions.querySelectorAll('li').forEach(li => {
        li.addEventListener('click', (ev) => {
            const v = li.getAttribute('data-value');
            const text = li.textContent.trim();
            // update hidden input
            reasoningSelect.value = v;
            // update visual selected
            reasoningOptions.querySelectorAll('li').forEach(n => n.classList.remove('selected'));
            li.classList.add('selected');
            reasoningBtn.firstChild.nodeValue = text; // set visible text (firstChild is text node)
            // ensure caret remains (rebuild)
            const caret = document.createElement('span');
            caret.className = 'caret';
            // fix button content to include caret after text
            reasoningBtn.innerHTML = text;
            reasoningBtn.appendChild(caret);
            closeReasoning();
            reasoningBtn.focus();
        });
    });

    // keyboard accessibility for custom select (Enter/Space to toggle, arrows to navigate)
    customReasoning.addEventListener('keydown', (e) => {
        const open = reasoningOptions.classList.contains('open');
        const items = Array.from(reasoningOptions.querySelectorAll('li'));
        const currentIndex = items.findIndex(i => i.classList.contains('selected'));
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            if (!open) openReasoning(); else {
                // if open and an item is focused (use document.activeElement), click it
                const focused = document.activeElement;
                if (reasoningOptions.contains(focused) && focused.tagName === 'LI') focused.click();
                else closeReasoning();
            }
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (!open) openReasoning();
            const nextIndex = (currentIndex + 1) % items.length;
            items[nextIndex].focus();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (!open) openReasoning();
            const prevIndex = (currentIndex - 1 + items.length) % items.length;
            items[prevIndex].focus();
        } else if (e.key === 'Escape') {
            closeReasoning();
            reasoningBtn.focus();
        }
    });

    // ensure list items are focusable for keyboard
    reasoningOptions.querySelectorAll('li').forEach(li => {
        li.tabIndex = 0;
        li.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                li.click();
            } else if (e.key === 'Escape') {
                closeReasoning();
                reasoningBtn.focus();
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                const next = li.nextElementSibling || reasoningOptions.querySelector('li');
                next.focus();
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                const prev = li.previousElementSibling || reasoningOptions.querySelector('li:last-child');
                prev.focus();
            }
        });
    });

    // Splash Screen Logic
    window.addEventListener('load', () => {
        setTimeout(() => {
            const splash = document.getElementById('splash');
            splash.style.opacity = '0';
            setTimeout(() => {
                splash.style.visibility = 'hidden';
            }, 800);
        }, 1500); // Show for 1.5s
    });

    function renderContent(text) {
        const mathBlocks = [];

        // 1️⃣ Extract block math
        text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => {
            mathBlocks.push({ type: "block", content: m });
            return `@@BLOCK_${mathBlocks.length - 1}@@`;
        });

        // 2️⃣ Extract inline math
        text = text.replace(/\\\(([\s\S]*?)\\\)/g, (_, m) => {
            mathBlocks.push({ type: "inline", content: m });
            return `@@INLINE_${mathBlocks.length - 1}@@`;
        });

        // 3️⃣ Markdown parse ONLY plain text
        let html = marked.parse(text, {
            mangle: false,
            headerIds: false
        });

        // 4️⃣ Restore math verbatim (NO Markdown)
        mathBlocks.forEach((m, i) => {
            const latex =
                m.type === "block"
                    ? `\\[${m.content}\\]`
                    : `\\(${m.content}\\)`;

            html = html
                .replace(`@@BLOCK_${i}@@`, latex)
                .replace(`@@INLINE_${i}@@`, latex);
        });

        return html;
    }

    async function ask() {
        const val = input.value.trim();
        if (!val) return;

        input.value = '';
        input.disabled = true;
        sendBtn.disabled = true;

        appendMsg('user', val);

        // Show Typing Indicator
        typingIndicator.style.display = 'block';
        chat.scrollTop = chat.scrollHeight;

        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: val,
                    reasoning_level: reasoningSelect.value,
                    recheck: recheckToggle.checked
                })
            });

            const data = await res.json();

            // Hide Typing Indicator
            typingIndicator.style.display = 'none';

            if (data.tool_output) {
                const toolDiv = document.createElement('div');
                toolDiv.className = 'msg assistant';
                toolDiv.innerHTML = `
                    <div class="msg-label">AI:</div>
                    <div class="tool-output">$ python3 tool.py [Mode: ${data.tool_output.verify}]\n${data.tool_output.stdout}</div>`;
                chat.appendChild(toolDiv);
            }

            appendMsg('assistant', data.reply);
        } catch {
            typingIndicator.style.display = 'none';
            appendMsg('assistant', 'Network error');
        }

        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }

    function appendMsg(role, text) {
        if (!text || !text.trim()) {
            text = "_[No textual explanation returned]_";
        }

        const div = document.createElement('div');
        div.className = `msg ${role}`;
        
        // Only render the label for assistant messages. user label removed as requested.
        const labelHTML = role === 'assistant' ? `<div class="msg-label">AI:</div>` : ``;

        div.innerHTML = `
            ${labelHTML}
            <div class="msg-content">${renderContent(text)}</div>
        `;
        
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
        MathJax.typesetPromise([div]);
    }

    input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
</script>
</body>
</html>
"""

def _cleanup():
    logger.info("Shutdown.")

atexit.register(_cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50051, debug=False)

