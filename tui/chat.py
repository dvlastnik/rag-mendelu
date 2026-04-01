import sys
import threading
import time

from rag.agentic_rag import AgenticRAG

_DIVIDER = "─" * 62
_MAX_EXCERPT = 120


class TuiChat:
    def __init__(self, rag: AgenticRAG, model_name: str) -> None:
        self._rag = rag
        self._model_name = model_name

    def _print_header(self) -> None:
        print(f"\nRAG Assistant ({self._model_name}) — type 'exit' or press Ctrl+C to quit")
        print(_DIVIDER)

    def _animate_thinking(self, stop_event: threading.Event) -> None:
        frames = [".", "..", "..."]
        i = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r  {frames[i % 3]}   ")
            sys.stdout.flush()
            time.sleep(0.4)
            i += 1
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()

    def _ask(self, question: str) -> dict:
        stop_event = threading.Event()
        anim_thread = threading.Thread(
            target=self._animate_thinking, args=(stop_event,), daemon=True
        )
        anim_thread.start()
        try:
            result = self._rag.chat(question)
        finally:
            stop_event.set()
            anim_thread.join()
        return result

    def _print_result(self, result: dict) -> None:
        response = result.get("response", "")
        sources = result.get("sources") or []

        print(f"\nAssistant ({self._model_name}):")
        print(f"  {response}\n")

        if sources:
            print("  Sources:")
            for i, doc in enumerate(sources, start=1):
                source_name = (doc.metadata or {}).get("source", "unknown")
                excerpt = doc.text.replace("\n", " ").strip()
                if len(excerpt) > _MAX_EXCERPT:
                    excerpt = excerpt[:_MAX_EXCERPT] + "…"
                print(f'    [{i}] {source_name:<20} — "{excerpt}"')
            print()

        print(_DIVIDER)

    def run_chat(self) -> None:
        self._print_header()
        while True:
            try:
                print()
                question = input("You: ").strip()
                if not question:
                    continue
                if question.lower() in ("exit", "quit"):
                    break
                print()
                result = self._ask(question)
                self._print_result(result)
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break

    def run_ask(self, question: str) -> None:
        self._print_header()
        print(f"\nYou: {question}\n")
        result = self._ask(question)
        self._print_result(result)
