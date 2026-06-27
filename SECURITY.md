# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in AtomicRAG, please **do not open a public issue**.

Instead, email directly: amareshhebbar@gmail.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 48 hours.

## Scope

- Model output hallucination leading to incorrect query decomposition
- Prompt injection via malicious input questions
- Unsafe deserialization in JSON output parsing (src/utils.py)
- Dependency vulnerabilities in requirements.txt
