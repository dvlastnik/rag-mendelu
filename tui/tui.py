import argparse
import datetime
import os
import subprocess
import sys

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from prompt_toolkit import prompt as _pt_prompt
from prompt_toolkit.key_binding import KeyBindings

BACK = "__back__"

class _GoBack(Exception):
    """Raised internally when Backspace is pressed on an empty text field."""


def _prompt(fn, *args, **kwargs):
    """Execute an InquirerPy prompt, exiting cleanly on Ctrl+C."""
    try:
        return fn(*args, **kwargs).execute()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)


def _text_with_back(message: str, default: str = "") -> str | object:
    """Plain text prompt where Backspace on an empty field returns BACK."""
    kb = KeyBindings()

    @kb.add("backspace")
    def _(event):
        if event.app.current_buffer.text == "":
            raise _GoBack()
        event.app.current_buffer.delete_before_cursor()

    try:
        return _pt_prompt(f"{message} ", key_bindings=kb, default=default)
    except _GoBack:
        return BACK
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)


def _back_choice() -> Choice:
    return Choice(BACK, "<< Back")


class TuiWizard:
    def __init__(
        self,
        default_model: str = "ministral-3:8b",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ) -> None:
        self._default_model = default_model
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port

    @staticmethod
    def _get_ollama_models() -> list[str]:
        """Return model names from `ollama list`, empty list on failure."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]
                return [line.split()[0] for line in lines if line.strip()]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return []

    @staticmethod
    def _get_qdrant_collections(host: str, port: int) -> list[str]:
        """Return existing Qdrant collection names, empty list on failure."""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=host, port=port, timeout=3)
            result = client.get_collections()
            client.close()
            return [c.name for c in result.collections]
        except Exception:
            return []

    @staticmethod
    def _is_qdrant_running(host: str, port: int) -> bool:
        """Return True if Qdrant is reachable."""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=host, port=port, timeout=3)
            client.get_collections()
            client.close()
            return True
        except Exception:
            return False

    def _step_log_file(self) -> str | None:
        """Ask whether to save logs to a dated file. Returns file path or None."""
        want_logs = _prompt(
            inquirer.confirm,
            message="Save logs to a file?",
            default=False,
        )
        if want_logs:
            date_str = datetime.date.today().strftime("%Y%m%d")
            return f"logs/{date_str}.log"
        return None

    def _step_mode(self) -> str:
        return _prompt(
            inquirer.select,
            message="What would you like to do?",
            choices=[
                _back_choice(),
                Choice("chat",      "Chat — interactive RAG conversation"),
                Choice("ask",       "Ask — single question, then exit"),
                Choice("run_etl",   "Run ETL — ingest documents into the database"),
                Choice("check_dbs", "Check database status"),
            ],
        )

    def _step_model(self, ollama_models: list[str]) -> str:
        if ollama_models:
            return _prompt(
                inquirer.select,
                message="LLM model:",
                choices=[
                    _back_choice(),
                    *[Choice(m, m) for m in ollama_models],
                ],
                default=self._default_model if self._default_model in ollama_models else None,
            )
        return _prompt(inquirer.text, message="LLM model (ollama):", default=self._default_model)

    def _step_chat_collection(self, existing: list[str]) -> str:
        if not existing:
            print("No collection found — please run ETL first.")
            sys.exit(1)
        return _prompt(
            inquirer.select,
            message="Collection:",
            choices=[
                _back_choice(),
                *[Choice(c, c) for c in existing],
            ],
        )

    def _step_embed_model(self) -> str | None:
        raw = _text_with_back("Embedding model (leave blank for default BAAI/bge-small-en-v1.5, Backspace to go back):")
        if raw is BACK:
            return BACK
        return raw.strip() or None

    def _step_json_output(self) -> bool:
        return _prompt(
            inquirer.confirm,
            message="Output responses as JSON?",
            default=False,
        )

    def _step_question(self) -> str:
        val = _text_with_back("Your question (Backspace to go back):")
        return val

    def _step_etl_target(self) -> str:
        return _prompt(
            inquirer.select,
            message="Ingest into:",
            choices=[
                _back_choice(),
                Choice("new",      "New collection — create a fresh one"),
                Choice("existing", "Existing collection — append to an existing one"),
            ],
        )

    def _step_etl_new_name(self) -> str | object:
        val = _text_with_back("New collection name (Backspace to go back):")
        if val is BACK or not str(val).strip():
            return BACK
        return str(val).strip()

    def _step_etl_existing_collection(self, existing: list[str]) -> str | object:
        if existing:
            return _prompt(
                inquirer.select,
                message="Select collection:",
                choices=[
                    _back_choice(),
                    *[Choice(c, c) for c in existing],
                ],
            )
        val = _text_with_back("No collections found. Enter collection name (Backspace to go back):")
        if val is BACK or not str(val).strip():
            return BACK
        return str(val).strip()

    def _step_etl_erase(self) -> bool:
        return _prompt(
            inquirer.confirm,
            message="Erase existing collection before ingesting?",
            default=False,
        )

    def _step_etl_path(self) -> str:
        result = _prompt(
            inquirer.filepath,
            message="Path to file or folder to ingest (if running from docker select /app/data/input):",
            default=os.getcwd(),
            only_directories=False,
        )

        if result == "":
            print("Path is required through ETL.")
            sys.exit(1)

        return result

    def _step_embed_model_etl(self) -> str | None:
        return self._step_embed_model()

    # state machine
    def _run_step(self, step: str) -> object:
        """Dispatch to the prompt for `step` and return the raw value (or BACK)."""
        existing = self._get_qdrant_collections(self._qdrant_host, self._qdrant_port)
        ollama_models = self._get_ollama_models()

        dispatch = {
            "log_file":               self._step_log_file,
            "mode":                   self._step_mode,
            "model":                  lambda: self._step_model(ollama_models),
            "chat_collection":        lambda: self._step_chat_collection(existing),
            "embed_model":            self._step_embed_model,
            "json_output":            self._step_json_output,
            "question":               self._step_question,
            "etl_target":             self._step_etl_target,
            "etl_new_name":           self._step_etl_new_name,
            "etl_existing_collection": lambda: self._step_etl_existing_collection(existing),
            "etl_erase":              self._step_etl_erase,
            "etl_path":               self._step_etl_path,
            "embed_model_etl":        self._step_embed_model_etl,
        }
        return dispatch[step]()

    @staticmethod
    def _next_step(step: str, results: dict) -> str:
        """Return the name of the step that follows `step` given collected results."""
        mode = results.get("mode")
        etl_target = results.get("etl_target")

        transitions = {
            "log_file":    "mode",
            "mode":        (
                "model"      if mode in ("chat", "ask") else
                "etl_target" if mode == "run_etl" else
                "DONE"
            ),
            "model":           "chat_collection",
            "chat_collection": "embed_model",
            "embed_model":     "json_output",
            "json_output":     "question" if mode == "ask" else "DONE",
            "question":        "DONE",
            "etl_target":      "etl_new_name" if etl_target == "new" else "etl_existing_collection",
            "etl_new_name":    "etl_path",
            "etl_existing_collection": "etl_erase",
            "etl_erase":       "etl_path",
            "etl_path":        "embed_model_etl",
            "embed_model_etl": "DONE",
        }
        return transitions[step]

    @staticmethod
    def _build_namespace(results: dict) -> argparse.Namespace:
        mode = results.get("mode", "")
        etl_target = results.get("etl_target", "")

        if etl_target == "new":
            collection_name = results.get("etl_new_name", "")
            erase = True
        else:
            collection_name = results.get("etl_existing_collection", "")
            erase = results.get("etl_erase", False)

        return argparse.Namespace(
            chat=(mode == "chat"),
            ask=results.get("question", ""),
            run_etl=(mode == "run_etl"),
            check_dbs=(mode == "check_dbs"),
            model=results.get("model", ""),
            collection_name=collection_name or results.get("chat_collection", ""),
            embed_model=results.get("embed_model") or results.get("embed_model_etl"),
            path=results.get("etl_path", ""),
            erase=erase,
            json_output=results.get("json_output", False),
            tui_mode=True,
            log_file=results.get("log_file"),
        )

    # entry point
    def run(self) -> argparse.Namespace:
        """Run the interactive wizard and return a populated argparse.Namespace."""
        if not self._is_qdrant_running(self._qdrant_host, self._qdrant_port):
            print(
                f"Qdrant is not running at {self._qdrant_host}:{self._qdrant_port}. "
                "Please start it with: docker compose up -d"
            )
            sys.exit(1)

        try:
            results: dict = {}
            history: list[str] = []
            step = "log_file"

            while step != "DONE":
                val = self._run_step(step)

                if val == BACK:
                    if history:
                        step = history.pop()
                        results.pop(step, None)
                else:
                    results[step] = val
                    history.append(step)
                    step = self._next_step(step, results)

        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

        return self._build_namespace(results)