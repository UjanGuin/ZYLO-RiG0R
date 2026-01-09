# ZYLO-RiG0R
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Robotics](https://img.shields.io/badge/Domain-Robotics-critical)
![AI](https://img.shields.io/badge/AI-Enabled-purple)
![Sensor Fusion](https://img.shields.io/badge/Sensor%20Fusion-Implemented-informational)
![PID](https://img.shields.io/badge/Control-PID-orange)
![Kalman](https://img.shields.io/badge/Filter-Kalman-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/UjanGuin/ZYLO-RiG0R)


ZYLO-RiG0R is an ultra-fast, research-grade math & physics reasoning engine built on GPT-OSS-120B with Cerebras-accelerated inference and strict tool-augmented computation.

Unlike generic chat systems, ZYLO-RiG0R forces verified computation, executes sandboxed Python and SymPy, and optionally rechecks proofs using GLM-4.7, ensuring results are correct, reproducible, and defensible.

It is designed to outperform top-tier AI models on calculation-heavy, exam-level, and research-grade problems by eliminating hallucinations and unverifiable reasoning.


<p align="center">
  <img src="assets/ui1.png" width="700">
</p>
<p align="center">
  <img src="assets/ui2.png" width="700">
</p>

---

## Why ZYLO-RiG0R Exists ?

Modern LLMs are optimized for fluency, not correctness.
ZYLO-RiG0R was built to solve that failure mode.

### ZYLO-RiG0R enforces:

- Tool-only execution when computation is required

- Strict JSON-based tool invocation

- Sandboxed Python execution with resource limits

- Symbolic verification using SymPy

- Independent proof rechecking via GLM-4.7

- MathJax-safe mathematical rendering

- Phase-aware inference (Planning → Execution → Exposition)

#### The result is a system that cannot “fake” answers.



## Core Capabilities

- 🚀 Ultra-low-latency inference via Cerebras

- 🧮 Guaranteed computation correctness (Python + SymPy)

- 🔒 Sandboxed execution with CPU & memory limits

- 📐 Research-grade math normalization

- 🔁 Optional independent proof verification

- 🌐 Minimal, distraction-free web UI

- ⚙️ Reasoning-level control (Fast / Balanced / Research)



## Philosophy

"If a result cannot be verified, it is not an answer."

### ZYLO-RiG0R prioritizes:

- Correctness over verbosity

- Verification over explanation

- Results over rhetoric



## Who This Is For

- JEE / Olympiad / university-level students

- Researchers and engineers

- Anyone who cares about correct answers, not confident lies




## Status

#### ⚠️ Experimental / Research System
This project pushes LLMs beyond conversational use into formal reasoning infrastructure.



---

## Architecture Overview

ZYLO-RiG0R is built as a **phase-aware reasoning pipeline** that strictly separates thinking, computation, and exposition.

At a high level, every request flows through the following stages:

1. **Planning Phase**
   - High-entropy reasoning to understand the problem
   - Decides whether verified computation is mandatory

2. **Execution Phase**
   - Forces tool-only execution when required
   - Runs sandboxed Python or SymPy with strict CPU and memory limits
   - Blocks natural-language output during computation

3. **Exposition Phase**
   - Converts verified results into human-readable explanations
   - Guarantees that computed values explicitly appear in the final answer

4. **Optional Proof Recheck**
   - Independently validates results using GLM-4.7
   - Assigns confidence based on verification verdict

This architecture prevents hallucinated results by **design**, not by prompt tuning.



---

## Installation

### Requirements

- Python 3.9+
- Linux or macOS (sandboxed execution required). Windows is not preferred.
- Cerebras API access
- GLM-4.7 API key (optional, for proof rechecking)

### Install Dependencies

```bash
pip install flask flask-cors requests cerebras-cloud-sdk sympy numpy regex jsonschema rich python-dotenv scipy
```
or
```bash
pip install flask flask-cors requests cerebras-cloud-sdk sympy numpy regex jsonschema rich python-dotenv scipy --break-system-packages
```

## Requirements
To run the code, one need to replace the placeholders of Cerebras and GLM 4.7 API key with his own api keys. Both Crebras and Zhipu AI(GLM 4.7) offer free API key with some limits. The limits are enough for general work. Just login and get started.
### The free-tier limits:
<p align="center">
  <img src="assets/Cerebraslimits.png" width="700">
</p>

<p align="center">
  <img src="assets/Zhipulimits.png" width="700">
</p>

##### Get Cerebras AI API ket at-
https://cloud.cerebras.ai/platform/get-started
##### Get Zhipu AI API key at-
https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys

<p align="center">
  <img src="assets/apikeys.png" width="700">
</p>

After inserting the API keys, one is ready to run the code.

Running the code will produce 2 links. Click any of them to start chatting. 
The links will only work under the WiFi which the laptop running the code is connected to.

#### Access from internet:
To access the running server from anywhere from the world, beyond the wifi, one may use ngrok.
##### Follow these Steps:
- Login: https://dashboard.ngrok.com/login
- Copy the Authtoken: https://dashboard.ngrok.com/get-started/your-authtoken
- To download ngrok paste in terminal:
```bash
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
```
```bash
tar -xvzf ngrok-v3-stable-linux-amd64.tgz
```
```bash
sudo mv ngrok /usr/local/bin
```
- To add the Authtoken, paste:
```bash
ngrok config add-authtoken $YOUR_AUTHTOKEN
```
(Replace $YOUR_AUTHTOKEN with the copied Authtoken.)
- Now paste:
```bash
ngrok http 50051
```
(50051 is the port. One can change it as wanted, but also ensure to change it in python code.)
<p align="center">
  <img src="assets/port.png" width="600">
</p>

- After it, one will see output similar to:
```bash
Forwarding  https://xxxx-xxxx.ngrok-free.app -> http://localhost:50051
```
- Copy
```https://xxxx-xxxx.ngrok-free.app```
and paste at any browser to get started.

**Note:** The URL is not static.


## Specials
ZYLO-RiG0R is specialized for mathematics and physics problem-solving. In practice, it often produces correct solutions to complex questions before mainstream models such as GPT-5.2 and Gemini 3 Pro complete their reasoning. In multiple evaluations, ZYLO-RiG0R has delivered correct results in cases where GPT-5.2 produced incorrect answers. Most responses are generated within 1–2 seconds.


<p align="center">
  <img src="assets/JEEADVq1.png" width="700">
</p>

<p align="center">
  <img src="assets/comparison11.png" width="700">
</p>



---

## Credits & Acknowledgements

This project was designed, implemented, and maintained by **Swapnanil Guin**.

Special thanks to:
- Open-source contributors whose libraries made this project possible
- ChatGPT & Gemini for constant help
- API and infrastructure providers used strictly as inference backends

All system design, architecture, verification logic, and optimizations are original unless explicitly stated otherwise.

### Third-Party Technologies
- **SymPy** — symbolic mathematics engine
- **Flask** — lightweight web framework
- **Cerebras Inference API** — high-throughput LLM backend
