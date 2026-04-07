import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Send
from langgraph.graph import END

from rag.agents.nodes.rag_nodes import RagNodes
from rag.agents.enums import NodeName, Intent
from rag.agents.models import QueryPlan, QueryStrategy
from rag.agents.prompts import Prompts

# ==========================================
# 0d. SQL PROMPT RULE TESTS
# ==========================================
def test_sql_prompt_has_no_backslash_escape_rule():
    """SQL prompt must forbid backslash escaping of single quotes."""
    prompt = Prompts.get_sql_generator_prompt("games_2025(name:varchar) [5 rows]")
    assert "backslash" in prompt.lower(), \
        "SQL prompt must explicitly forbid backslash escaping"

def test_sql_prompt_has_union_all_vs_intersect_rule():
    """SQL prompt must explain UNION ALL vs INTERSECT distinction in the rules section."""
    prompt = Prompts.get_sql_generator_prompt("games_2025(name:varchar) [5 rows]")
    rules_section = prompt.split("EXAMPLES")[0]
    assert "INTERSECT" in rules_section, "INTERSECT rule must appear before EXAMPLES"

def test_sql_prompt_has_stealth_games_union_example():
    """SQL prompt must include a stealth/spy UNION ALL example."""
    prompt = Prompts.get_sql_generator_prompt("games_2025(name:varchar) [5 rows]")
    assert "stealth" in prompt.lower() or "spy" in prompt.lower()

# ==========================================
# 0b. ETL COLUMN SANITIZATION TESTS
# ==========================================
from etl.table_extractor import TableProcessor

def test_column_sanitizer_strips_dots_and_specials():
    """Column keys must be valid SQL identifiers — no dots, spaces, or leading digits."""
    processor = TableProcessor()
    md = (
        "| Key Messages . . . | . _ . _ ii | 2nd col |\n"
        "|---|---|---|\n"
        "| val1 | val2 | val3 |\n"
    )
    _, row_docs = processor.process_document(md, base_metadata={"source": "test", "file_type": "md"})
    keys = list(row_docs[0]['metadata'].keys())
    sql_keys = [k for k in keys if k not in {'is_table', 'table_index', 'source', 'file_type'}]
    for key in sql_keys:
        assert '.' not in key, f"Dot in key: {key}"
        assert not key[0].isdigit(), f"Digit-leading key: {key}"

def test_column_sanitizer_deduplicates_keys():
    """Two headers that sanitize to the same key must be differentiated."""
    processor = TableProcessor()
    # Both '...' and '_ _' sanitize to '' → should become col_0 and col_1
    md = (
        "| ... | _ _ |\n"
        "|---|---|\n"
        "| a | b |\n"
    )
    _, row_docs = processor.process_document(md, base_metadata={"source": "test", "file_type": "md"})
    keys = [k for k in row_docs[0]['metadata'] if k not in {'is_table', 'table_index', 'source', 'file_type'}]
    assert len(keys) == 2, f"Expected 2 keys (one per column), got {len(keys)}: {keys}"
    assert len(keys) == len(set(keys)), f"Duplicate keys: {keys}"

# ==========================================
# 0c. BAD-TABLE GUARD TESTS
# ==========================================
def test_bad_table_guard_logic_detects_mostly_empty_keys():
    """Tables with >60% empty/single-char headers must be identified as bad."""
    from etl.table_extractor import TableProcessor
    processor = TableProcessor()
    # 3 of 4 headers are effectively single-char after sanitization:
    # '.' → '' → 'col_0' (len=5, OK), '_ _' → '' → 'col_1' (len=5, OK),
    # Actually let's use headers that produce short sanitized keys
    # '.' sanitizes to '' → col_0 (len=5), '|' → '' → col_1 (len=5)
    # We need headers that produce ≤1 char keys. Use single non-alnum chars.
    # Actually after the Task 1 fix: '.' → '' → col_0. That's len=5.
    # The BAD-TABLE guard checks len(k) <= 1 on the KEYS in the metadata row dict.
    # Let's think: what produces a 1-char key?
    # 'a' → 'a' (len=1, BAD), 'ab' → 'ab' (len=2, OK), 'a.b' → 'a_b' (len=3, OK)
    # A single letter header produces a 1-char key.
    md = (
        "| a | b | c | Good Header |\n"
        "|---|---|---|---|\n"
        "| val1 | val2 | val3 | val4 |\n"
    )
    _, row_docs = processor.process_document(md, base_metadata={"source": "test", "file_type": "md"})
    _INTERNAL = {'is_table', 'source', 'file_type', 'table_index'}
    row = {k: v for k, v in row_docs[0]['metadata'].items() if k not in _INTERNAL}
    short_keys = [k for k in row if len(k) <= 1]
    fraction_bad = len(short_keys) / len(row) if row else 1.0
    assert fraction_bad > 0.6, f"Expected >60% bad keys, got {fraction_bad:.0%} for keys: {list(row.keys())}"

def test_bad_table_guard_skips_register_dataframe_for_mostly_short_keys():
    """GeneralEtl.transform() must NOT call register_dataframe for bad tables."""
    import tempfile, os
    from unittest.mock import MagicMock, patch
    from etl.general_etl import GeneralEtl

    # Create a temp .md file with a table whose headers are mostly 1-char
    md_content = (
        "# Test\n\n"
        "| a | b | c | Good Header |\n"
        "|---|---|---|---|\n"
        "| 1 | 2 | 3 | data |\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        md_file = os.path.join(tmpdir, "test_bad_table.md")
        with open(md_file, "w") as f:
            f.write(md_content)

        # Mock all dependencies
        mock_db = MagicMock()
        mock_embedding = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.uuid = "test-uuid"
        mock_embed_response.embedding = [0.1] * 384
        mock_embed_response.sparse = MagicMock(indices=[0], values=[1.0])
        mock_embedding.get_embedding_with_uuid.return_value = [mock_embed_response]
        mock_sql_db = MagicMock()

        etl = GeneralEtl(
            filepath=md_file,
            db_repository=mock_db,
            embedding_service=mock_embedding,
            sql_db_repo=mock_sql_db,
        )

        # Patch the OUTPUT_FOLDER so the ETL reads from the temp dir
        # GeneralEtl._read_markdown() reads from get_file_path() which returns
        # OUTPUT_FOLDER / f"{file.stem}.md" — so we need to patch it to return our temp file path
        import pathlib
        with patch.object(etl, 'get_file_path', return_value=pathlib.Path(md_file)):
            etl.transform()

        # The table has 3 of 4 single-char keys (75% > 60%) → must NOT call register_dataframe
        mock_sql_db.register_dataframe.assert_not_called()

# ==========================================
# 0. PROMPT CONTENT TESTS
# ==========================================
def test_query_planner_prompt_has_cross_domain_example():
    """QueryPlanner prompt must include a cross-domain compound hybrid example."""
    prompt = Prompts.get_query_planner_prompt(
        compact_catalog="games_2025(name:varchar, review:double) [20 rows]",
        available_sources=["games_2025", "lotr_lore"],
    )
    assert "Split Fiction" in prompt
    assert "Tolkien" in prompt

def test_query_planner_prompt_removed_metallica_vector_example():
    """Single-topic Metallica vector example must be removed to reduce small-model confusion."""
    prompt = Prompts.get_query_planner_prompt(
        compact_catalog="",
        available_sources=[],
    )
    assert 'Tell me about the history of Metallica' not in prompt

# ==========================================
# 1. QUERY PLANNER TESTS
# ==========================================
def test_query_planner_fallback_to_vector(rag_nodes, mock_llm):
    """Positive: When compact catalog is empty, falls back to vector decomposition."""
    rag_nodes._compact_catalog = ""  # force fallback path regardless of sql_db_repo

    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = MagicMock(queries=["keyword query", "conceptual query"])
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {
        'messages': [HumanMessage(content="original question")],
        'detected_source': None,
        'intent': Intent.RAG,
    }
    result = rag_nodes.query_planner_agent(state)

    assert 'rewritten_queries' in result
    assert "original question" in result['rewritten_queries']
    assert result['query_plan'].strategy == QueryStrategy.VECTOR

def test_query_planner_uses_sql_strategy(rag_nodes, mock_llm):
    """Positive: With SQL catalog present, planner can choose SQL strategy."""
    mock_plan = MagicMock()
    mock_plan.strategy = QueryStrategy.SQL
    mock_plan.sql_sources = ["games_2025"]
    mock_plan.sql_hint = "row with max review"
    mock_plan.vector_queries = []

    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = mock_plan
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {
        'messages': [HumanMessage(content="what is the highest rated game?")],
        'detected_source': None,
        'intent': Intent.RAG,
    }
    result = rag_nodes.query_planner_agent(state)

    assert result['query_plan'].strategy == QueryStrategy.SQL
    assert result['query_plan'].sql_sources == ["games_2025"]

def test_query_planner_uses_llm_structured_not_llm(mock_llm, mock_db, mock_embedding, mock_sql_db):
    """QueryPlanner must call with_structured_output on llm_structured, not llm."""
    mock_llm_structured = MagicMock()
    nodes = RagNodes(mock_llm, mock_db, mock_embedding, duck_db_repo=mock_sql_db, llm_structured=mock_llm_structured)

    mock_plan = MagicMock()
    mock_plan.strategy = QueryStrategy.VECTOR
    mock_plan.sql_sources = []
    mock_plan.vector_queries = ["glaciers Alps retreat"]

    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = mock_plan
    mock_llm_structured.with_structured_output.return_value = mock_runnable

    state = {
        'messages': [HumanMessage(content="What caused glacier retreat in the Alps?")],
        'detected_source': None,
        'intent': Intent.RAG,
    }
    nodes.query_planner_agent(state)

    mock_llm_structured.with_structured_output.assert_called_once_with(QueryPlan)
    mock_llm.with_structured_output.assert_not_called()

# ==========================================
# 2. RESEARCH WORKER TESTS
# ==========================================
def test_research_worker_negative_embedding_fail(rag_nodes, mock_embedding):
    """Negative: Embedding service returns None or error."""
    mock_embedding.get_embedding_with_uuid.return_value = []

    target = MagicMock()
    target.topics = []
    state = {'target': target, 'query': 'search text'}

    result = rag_nodes.research_worker(state)

    assert "Could not generate embeddings" in result['search_results'][0]

# ==========================================
# 3. RETRIEVAL GRADER TESTS
# ==========================================
def test_retrieval_grader_reranking(rag_nodes):
    """Positive: Reranks docs and returns top-N."""
    doc1 = MagicMock(id="1", text="Good Doc", metadata={"source": "test"})
    doc2 = MagicMock(id="2", text="Bad Doc", metadata={"source": "test"})

    state = {
        'messages': [HumanMessage(content='q')],
        'search_results': [doc1, doc2],
        'filtered_results': [],
        'intent': Intent.RAG,
    }

    result = rag_nodes.retrieval_grader_agent(state)

    assert len(result['filtered_results']) <= rag_nodes.distiller_top_n

def test_retrieval_grader_empty_input(rag_nodes):
    """Negative: Handles empty search results gracefully."""
    state = {
        'messages': [HumanMessage(content='q')],
        'search_results': [],
        'filtered_results': [],
        'intent': Intent.RAG,
    }
    result = rag_nodes.retrieval_grader_agent(state)
    assert result['filtered_results'] == []

def test_retrieval_grader_excludes_already_filtered(rag_nodes):
    """Reranker skips docs that are already in filtered_results from prior iterations."""
    existing_doc = MagicMock(id="1", text="Already filtered", metadata={"source": "test"})
    new_doc = MagicMock(id="2", text="New doc", metadata={"source": "test"})

    state = {
        'messages': [HumanMessage(content='q')],
        'search_results': [existing_doc, new_doc],
        'filtered_results': [existing_doc],
        'intent': Intent.RAG,
    }

    result = rag_nodes.retrieval_grader_agent(state)

    result_ids = [doc.id for doc in result['filtered_results']]
    assert "1" not in result_ids

def test_retrieval_grader_adaptive_topn_exhaustive(rag_nodes):
    """Exhaustive intent uses 3x top_n."""
    docs = [MagicMock(id=str(i), text=f"Doc {i}", metadata={"source": "test"}) for i in range(30)]

    state = {
        'messages': [HumanMessage(content='list all bands')],
        'search_results': docs,
        'filtered_results': [],
        'intent': Intent.RAG_EXHAUSTIVE,
    }

    result = rag_nodes.retrieval_grader_agent(state)

    assert len(result['filtered_results']) <= rag_nodes.distiller_top_n * 3

# ==========================================
# 4. SCROLL RETRIEVER TESTS
# ==========================================
def test_scroll_retriever_fetches_docs(rag_nodes, mock_db):
    """Scroll retriever fetches all docs from a source."""
    mock_docs = [MagicMock(id=str(i), text=f"Doc {i}", metadata={"source": "history_of_metal"}) for i in range(5)]
    mock_db.scroll_all_by_source.return_value = mock_docs

    state = {
        'detected_source': 'history_of_metal',
        'messages': [HumanMessage(content="summarize history_of_metal")],
    }
    result = rag_nodes.scroll_retriever(state)

    assert len(result['filtered_results']) == 5
    assert len(result['search_results']) == 5
    mock_db.scroll_all_by_source.assert_called_once_with('history_of_metal', limit=500)

def test_scroll_retriever_no_source(rag_nodes):
    """Scroll retriever returns empty when no source detected."""
    state = {'detected_source': None}
    result = rag_nodes.scroll_retriever(state)

    assert result['filtered_results'] == []

# ==========================================
# 5. ANALYTICAL QUERY TESTS
# ==========================================
def test_analytical_query_returns_fact(rag_nodes, mock_llm):
    """Positive: SQL query executes and result is stored as distilled fact."""
    import pandas as pd
    mock_sql_plan = MagicMock()
    mock_sql_plan.sql = 'SELECT name, review FROM games_2025 WHERE review = (SELECT MAX(review) FROM games_2025)'
    mock_sql_plan.explanation = "Game with highest review score"

    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = mock_sql_plan
    mock_llm.with_structured_output.return_value = mock_runnable

    rag_nodes.duck_db_repo.run_select.return_value = pd.DataFrame({"name": ["Game1"], "review": [9.4]})

    state = {
        'messages': [HumanMessage(content="what is the highest rated game?")],
        'query_plan': QueryPlan(strategy=QueryStrategy.SQL, sql_sources=["games_2025"], sql_hint="max review"),
    }
    result = rag_nodes.analytical_query_agent(state)

    assert len(result['distilled_facts']) == 1
    assert "games_2025" in result['distilled_facts'][0]
    assert result['sql_result'] is not None

# ==========================================
# 5b. SQL RETRY TESTS
# ==========================================
def test_analytical_query_retries_on_sql_error(rag_nodes, mock_llm):
    """When first SQL fails, agent retries with error context and returns result."""
    import pandas as pd
    mock_plan_fail = MagicMock()
    mock_plan_fail.sql = 'SELECT name INTERSECT SELECT name FROM games_2026'
    mock_plan_fail.explanation = "bad plan"

    mock_plan_ok = MagicMock()
    mock_plan_ok.sql = 'SELECT name FROM games_2025 UNION ALL SELECT name FROM games_2026'
    mock_plan_ok.explanation = "fixed plan"

    mock_runnable = MagicMock()
    mock_runnable.invoke.side_effect = [mock_plan_fail, mock_plan_ok]
    mock_llm.with_structured_output.return_value = mock_runnable

    rag_nodes.duck_db_repo.run_select.side_effect = [
        Exception("Parser Error: syntax error near INTERSECT"),
        pd.DataFrame({"name": ["Game A", "Game B"]}),
    ]

    state = {
        'messages': [HumanMessage(content="stealth games in 2025 and 2026?")],
        'query_plan': QueryPlan(strategy=QueryStrategy.SQL, sql_sources=["games_2025", "games_2026"], sql_hint="union both years"),
    }
    result = rag_nodes.analytical_query_agent(state)

    assert result['sql_result'] is not None
    assert "Game" in result['distilled_facts'][0]
    assert mock_runnable.invoke.call_count == 2
    # Verify the retry call included the failed SQL and error context
    second_call_messages = mock_runnable.invoke.call_args_list[1][0][0]
    human_msg = next(m for m in second_call_messages if isinstance(m, HumanMessage))
    assert "FAILED" in human_msg.content
    assert "Parser Error" in human_msg.content

def test_analytical_query_returns_error_after_two_failures(rag_nodes, mock_llm):
    """When both SQL attempts fail, returns error message (no exception raised)."""
    mock_plan = MagicMock()
    mock_plan.sql = 'BAD SQL'
    mock_plan.explanation = "broken"

    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = mock_plan
    mock_llm.with_structured_output.return_value = mock_runnable

    rag_nodes.duck_db_repo.run_select.side_effect = Exception("syntax error")

    state = {
        'messages': [HumanMessage(content="broken query")],
        'query_plan': QueryPlan(strategy=QueryStrategy.SQL, sql_sources=["games_2025"], sql_hint=""),
    }
    result = rag_nodes.analytical_query_agent(state)

    assert result['sql_result'] is None
    assert "could not be executed" in result['distilled_facts'][0]

# ==========================================
# 6. SYNTHESIZER TESTS
# ==========================================
def test_synthesizer_positive_clean(rag_nodes, mock_llm):
    """Positive: Generates answer from distilled facts."""
    mock_llm.invoke.return_value = AIMessage(content="Final Answer")

    state = {
        'messages': [HumanMessage(content="User Q")],
        'distilled_facts': ["[Sources: test]\nFact 1\nFact 2"],
        'hallucination_status': 'clean'
    }

    result = rag_nodes.synthesizer_agent(state)
    assert result['messages'][0].content == "Final Answer"

    args, _ = mock_llm.invoke.call_args
    user_prompt = args[0][1].content
    assert "CRITICAL WARNING" not in user_prompt

def test_synthesizer_retry_warning_injection(rag_nodes, mock_llm):
    """Positive: Injects warning if previous attempt was hallucinated."""
    mock_llm.invoke.return_value = AIMessage(content="Fixed Answer")

    state = {
        'messages': [HumanMessage(content="User Q")],
        'distilled_facts': ["[Sources: test]\nFact 1"],
        'hallucination_status': 'hallucinated'
    }

    rag_nodes.synthesizer_agent(state)

    args, _ = mock_llm.invoke.call_args
    user_prompt = args[0][1].content
    assert "CRITICAL WARNING" in user_prompt

def test_synthesizer_negative_no_facts(rag_nodes):
    """Negative: No distilled facts -> I cannot answer."""
    state = {
        'messages': [HumanMessage(content="User Q")],
        'distilled_facts': []
    }
    result = rag_nodes.synthesizer_agent(state)
    assert "could not find specific information" in result['messages'][0].content

# ==========================================
# 6. HALLUCINATION GRADER TESTS
# ==========================================
def test_hallucination_grader_positive_grounded(rag_nodes, mock_llm):
    """Positive: Document is grounded."""
    mock_grade = MagicMock()
    mock_grade.is_relevant = "yes"

    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = mock_grade
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {
        'filtered_results': [MagicMock(text='doc')],
        'distilled_facts': ["[Sources: test]\nSome fact"],
        'messages': [AIMessage(content='answer')]
    }

    result = rag_nodes.hallucination_grader_agent(state)
    assert result['hallucination_status'] == 'clean'

def test_hallucination_grader_negative_hallucinated(rag_nodes, mock_llm):
    """Negative: Hallucination detected, increments retries."""
    mock_grade = MagicMock()
    mock_grade.is_relevant = "no"

    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = mock_grade
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {
        'filtered_results': [MagicMock(text='doc')],
        'distilled_facts': ["[Sources: test]\nSome fact"],
        'messages': [AIMessage(content='answer')],
        'hallucination_retries': 1
    }

    result = rag_nodes.hallucination_grader_agent(state)
    assert result['hallucination_status'] == 'hallucinated'
    assert result['hallucination_retries'] == 2

# ==========================================
# 7. FACT EXTRACTOR TESTS
# ==========================================
def test_fact_extractor_calls_llm(rag_nodes, mock_llm):
    """FactExtractor calls the LLM and returns extracted facts with source tag."""
    mock_llm.invoke.return_value = MagicMock(content="The Valar are the Powers of Arda.\nManwë is the King of the Valar.")
    doc = MagicMock()
    doc.metadata = {'source': 'lotr_lore'}
    doc.text = "The Valar are the Powers of Arda. Manwë is the King of the Valar. Melkor was the first Dark Lord."

    state = {
        'messages': [HumanMessage(content="Who are the Valar?")],
        'filtered_results': [doc],
    }
    result = rag_nodes.fact_extractor_agent(state)
    mock_llm.invoke.assert_called_once()
    assert len(result['distilled_facts']) == 1

def test_fact_extractor_returns_relevant_sentences(rag_nodes, mock_llm):
    """FactExtractor returns LLM-extracted facts tagged with source and iteration."""
    mock_llm.invoke.return_value = MagicMock(content="The Valar are the Powers of Arda.\nManwë is the King of the Valar.")
    doc = MagicMock()
    doc.metadata = {'source': 'lotr_lore'}
    doc.text = (
        "The Valar are the Powers of Arda. "
        "Hobbits love second breakfast. "
        "Manwë is the King of the Valar."
    )

    state = {
        'messages': [HumanMessage(content="Who are the Valar?")],
        'filtered_results': [doc],
    }
    result = rag_nodes.fact_extractor_agent(state)

    assert len(result['distilled_facts']) == 1
    facts = result['distilled_facts'][0]
    assert "Valar" in facts
    assert "[Sources: lotr_lore]" in facts

def test_fact_extractor_no_facts_when_llm_returns_no_relevant(rag_nodes, mock_llm):
    """When LLM returns NO RELEVANT FACTS FOUND, distilled_facts is empty."""
    mock_llm.invoke.return_value = MagicMock(content="NO RELEVANT FACTS FOUND")
    doc = MagicMock()
    doc.metadata = {'source': 'history_of_metal'}
    doc.text = "xyz " * 200

    state = {
        'messages': [HumanMessage(content="Who are the Valar in Tolkien?")],
        'filtered_results': [doc],
    }
    result = rag_nodes.fact_extractor_agent(state)
    assert result['distilled_facts'] == []

def test_fact_extractor_empty_docs(rag_nodes):
    """Empty filtered_results → empty distilled_facts."""
    state = {
        'messages': [HumanMessage(content="anything")],
        'filtered_results': [],
    }
    result = rag_nodes.fact_extractor_agent(state)
    assert result['distilled_facts'] == []

# ==========================================
# 7b. ROUTING LOGIC (Static Methods)
# ==========================================
def test_route_query_plan_vector_fan_out():
    """Positive: VECTOR strategy fans out to RESEARCH_WORKER per query."""
    state = {
        'extracted_data': [],
        'rewritten_queries': ['q1', 'q2'],
        'messages': [HumanMessage(content='q')],
        'query_plan': QueryPlan(strategy=QueryStrategy.VECTOR, vector_queries=['q1', 'q2']),
    }

    result = RagNodes.route_query_plan(state)

    assert len(result) == 2
    assert isinstance(result[0], Send)
    assert result[0].node == NodeName.RESEARCH_WORKER

def test_route_query_plan_sql_routes_to_analytical():
    """SQL strategy routes to ANALYTICAL_QUERY node."""
    state = {
        'rewritten_queries': [],
        'messages': [HumanMessage(content='what is the highest rated game?')],
        'query_plan': QueryPlan(strategy=QueryStrategy.SQL, sql_sources=['games_2025'], sql_hint='max review'),
    }

    result = RagNodes.route_query_plan(state)
    assert result == NodeName.ANALYTICAL_QUERY

def test_route_query_plan_scroll_routes_to_scroll_retriever():
    """SCROLL strategy routes to SCROLL_RETRIEVER."""
    state = {
        'rewritten_queries': [],
        'messages': [HumanMessage(content='summarize history_of_metal')],
        'query_plan': QueryPlan(strategy=QueryStrategy.SCROLL),
    }

    result = RagNodes.route_query_plan(state)
    assert result == NodeName.SCROLL_RETRIEVER

def test_route_query_plan_no_plan_falls_back_to_vector():
    """No query_plan in state falls back to vector fan-out."""
    state = {
        'extracted_data': [],
        'rewritten_queries': ['q1'],
        'messages': [HumanMessage(content='question')],
        'query_plan': None,
    }

    result = RagNodes.route_query_plan(state)
    assert isinstance(result, list)
    assert isinstance(result[0], Send)

def test_route_hallucination_retry():
    """Positive: Loop back to Synthesizer."""
    state = {'hallucination_status': 'hallucinated', 'hallucination_retries': 1}
    assert RagNodes.route_hallucination(state) == NodeName.SYNTHESIZER

def test_route_hallucination_stop():
    """Negative: Max retries reached -> END."""
    state = {
        'hallucination_status': 'hallucinated',
        'hallucination_retries': 3,
        'messages': [AIMessage(content='some answer')],
    }
    assert RagNodes.route_hallucination(state) == END

# ==========================================
# 8. ROUTE COMPLETENESS CHECK
# ==========================================
def test_route_completeness_skip_for_summarization_scroll(rag_nodes):
    """Summarization scroll-based queries skip gap-check loop."""
    state = {
        'retrieval_iterations': 0,
        'completeness_follow_up_query': 'some follow up',
        'intent': Intent.RAG_SUMMARIZATION,
        'detected_source': 'history_of_metal',
        'search_results': [],
    }

    result = rag_nodes.route_completeness_check(state)
    assert result == NodeName.HALLUCINATION_GRADER_AGENT

def test_route_completeness_normal_follow_up(rag_nodes):
    """Normal RAG queries with follow-up trigger another iteration."""
    state = {
        'retrieval_iterations': 1,
        'completeness_follow_up_query': 'missing info keywords',
        'intent': Intent.RAG,
        'detected_source': None,
        'search_results': [],
    }

    result = rag_nodes.route_completeness_check(state)
    assert isinstance(result, list)
    assert isinstance(result[0], Send)

def test_route_completeness_exhaustive_uses_follow_up(rag_nodes):
    """Exhaustive + incomplete routes to RESEARCH_WORKER (same as RAG)."""
    state = {
        'retrieval_iterations': 1,
        'completeness_follow_up_query': 'more bands missing',
        'intent': Intent.RAG_EXHAUSTIVE,
        'detected_source': None,
        'search_results': [],
    }

    result = rag_nodes.route_completeness_check(state)
    assert isinstance(result, list)
    assert isinstance(result[0], Send)

# ==========================================
# 8. MODEL PARAMS TESTS
# ==========================================
from rag.agents.nodes.rag_nodes import ModelParams

def test_model_params_8192_context_top_n_is_15():
    """8192-context models must use top_n=15 to capture chunks ranked 11-15."""
    params = ModelParams.create_from_context_window(8192)
    assert params.top_n == 15, f"Expected top_n=15, got {params.top_n}"
    assert params.chars_per_doc == 1500, f"Expected chars_per_doc=1500, got {params.chars_per_doc}"

def test_model_params_4096_context_unchanged():
    """4096-context model params must remain unchanged."""
    params = ModelParams.create_from_context_window(4096)
    assert params.top_n == 5

def test_model_params_32768_context_unchanged():
    """32768-context model params must remain unchanged."""
    params = ModelParams.create_from_context_window(32768)
    assert params.top_n == 15
