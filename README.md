# ICE.OS

A modular, multi-provider workflow engine for orchestrating and chaining LLM (Large Language Model) calls across providers like OpenAI, Anthropic, Google Gemini, and DeepSeek.

---

## Overview
ICE.OS enables you to build, test, and deploy complex LLM-powered workflows using a node-based architecture. It abstracts away provider-specific quirks, supports flexible chaining, and exposes a FastAPI backend for easy integration.

---

## Features
- **Multi-Provider Support:** OpenAI, Anthropic, Google Gemini, DeepSeek (extensible to more)
- **Node-Based Workflow Engine:** Compose chains of LLM calls and operations
- **Provider-Agnostic API:** Unified interface for different LLMs
- **Extensible:** Add new providers, nodes, or utilities easily
- **Testable:** Comprehensive test suite and fixtures
- **Modern Python:** Async, Pydantic, FastAPI, type hints

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ice.os
   ```
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quickstart

See [QUICKSTART.md](QUICKSTART.md) for detailed setup and usage instructions.

To start the API server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at [http://localhost:8000](http://localhost:8000)

---

## Folder Structure

See [CODEBASE_INFO.md](CODEBASE_INFO.md) for a detailed breakdown of the codebase structure and file descriptions.

---

## Testing

Run the test suite:
```bash
python tests/openai_chain_test.py
```

---

## Contributing

1. Fork the repo and create your branch (`git checkout -b feature/fooBar`)
2. Commit your changes (`git commit -am 'Add some fooBar'`)
3. Push to the branch (`git push origin feature/fooBar`)
4. Create a new Pull Request

---

## License

MIT License. See `LICENSE` file for details.

---

## Authors & Credits

- [Your Name or Team]
- Inspired by the open-source LLM ecosystem

---

## Market Opportunity & Vision

See [SCRIPTCHAIN_ANALYSIS.md](SCRIPTCHAIN_ANALYSIS.md) for a strategic and technical analysis of ICE.OS's potential as a startup seed. 