# Contributing to ZYLO-RiG0R

Thank you for your interest in contributing to **ZYLO-RiG0R**.

ZYLO-RiG0R is a research-grade mathematics and physics reasoning engine with a strong emphasis on **correctness, verification, and transparency**. Contributions are welcome, provided they align with the project’s design philosophy.

---

## 📌 Contribution Scope

You may contribute in the following ways:

- Bug reports with clear reproduction steps
- Feature proposals with strong motivation and trade-off analysis
- Improvements to documentation or Wiki content
- Performance optimizations that do not reduce correctness guarantees
- Security improvements or hardening (see SECURITY.md)

Low-effort, speculative, or correctness-breaking contributions are unlikely to be accepted.

---

## 🐞 Reporting Bugs

Please use the **Bug Report** issue template.

A valid bug report must include:
- Clear input or configuration that triggers the issue
- Expected vs actual behavior
- Environment details (OS, Python version, commit/release)
- Logs or error output where applicable

Vague reports without reproduction steps may be closed.

---

## 💡 Proposing Features

Please use the **Feature Request** issue template.

Feature proposals should address:
- Motivation and real-world use cases
- Impact on correctness, performance, and security
- Potential trade-offs or limitations

Features that compromise verifiability or introduce silent failure modes will not be accepted.

---

## 🔐 Security Issues

**Do not report security vulnerabilities publicly.**

If you discover a security issue, follow the instructions in `SECURITY.md` and report it privately.

---

## 🧠 Design Principles (Read Before Contributing)

All contributions must respect the following principles:

- **Correctness over convenience**
- **Explicit computation over inferred answers**
- **Transparent failure over silent guessing**
- **Deterministic behavior wherever possible**

If a contribution weakens these principles, it will be rejected regardless of implementation quality.

---

## 🛠 Development Guidelines

- Follow existing code structure and style
- Avoid introducing unnecessary dependencies
- Document non-trivial logic clearly
- Prefer explicit logic over “clever” abstractions
- Do not hardcode secrets, tokens, or credentials

---

## 🔁 Pull Request Process

1. Fork the repository
2. Create a focused branch for your change
3. Ensure your changes are minimal and well-scoped
4. Update documentation if behavior changes
5. Submit a Pull Request with a clear description

Pull Requests may be rejected or requested for revision if:
- They reduce correctness or verification guarantees
- They lack sufficient explanation
- They introduce unnecessary complexity

---

## 🧾 Licensing

By contributing to this repository, you agree that your contributions will be licensed under the **MIT License**, the same license as the project.

---

## 📣 Final Note

ZYLO-RiG0R is intentionally conservative in accepting changes.  
This is by design.

Thoughtful, well-reasoned contributions are always welcome.

Thank you for contributing responsibly.
