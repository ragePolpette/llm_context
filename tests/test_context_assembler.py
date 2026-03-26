from rag_indexer.context_assembler import AssemblyOptions, assemble_functional_context


def test_assemble_functional_context_groups_matches_by_file_and_attaches_symbols():
    retrieval_results = [
        {
            "source_path": "pubblico/api/Controllers/Fattura.cs",
            "score": 0.92,
            "text_hash": "a1",
            "line_start": 10,
            "line_end": 30,
            "chunk_index": 0,
            "section_path": "",
            "snippet": "Gestione fatturazione studi.",
            "text": "public class FatturaController { ... }",
        },
        {
            "source_path": "pubblico/api/Controllers/Fattura.cs",
            "score": 0.74,
            "text_hash": "a2",
            "line_start": 40,
            "line_end": 70,
            "chunk_index": 1,
            "section_path": "",
            "snippet": "Metodo GeneraFattura.",
            "text": "public void GeneraFattura() { ... }",
        },
        {
            "source_path": "librerie/BpoFH/FatturazioneService.cs",
            "score": 0.68,
            "text_hash": "b1",
            "line_start": 15,
            "line_end": 40,
            "chunk_index": 0,
            "section_path": "",
            "snippet": "Servizio di supporto fatturazione.",
            "text": "public class FatturazioneService { ... }",
        },
    ]
    symbol_results = [
        {
            "source_path": "pubblico/api/Controllers/Fattura.cs",
            "name": "GeneraFattura",
            "kind": "method",
            "signature": "public void GeneraFattura()",
            "line_start": 44,
            "line_end": 60,
        }
    ]

    payload = assemble_functional_context(
        query_text="genera fattura",
        retrieval_results=retrieval_results,
        symbol_results=symbol_results,
    )

    assert payload["summary"]["core_file_count"] == 2
    assert payload["package_version"] == "functional_context_v1"
    assert payload["query"]["terms"] == ["genera", "fattura"]
    assert payload["core_files"][0]["source_path"] == "pubblico/api/Controllers/Fattura.cs"
    assert payload["core_files"][0]["match_count"] == 2
    assert payload["core_files"][0]["symbol_hits"][0]["name"] == "GeneraFattura"
    assert payload["core_files"][0]["query_overlap_terms"] == ["fattura"]
    assert "query_overlap=" in payload["core_files"][0]["selection_reason"]
    assert payload["entry_points"][0]["name"] == "GeneraFattura"
    assert payload["summary"]["query_term_count"] == 2
    assert "FILE pubblico/api/Controllers/Fattura.cs" in payload["assembled_context"]
    assert "selection_reason=" in payload["assembled_context"]


def test_assemble_functional_context_deduplicates_repeated_chunk_hashes():
    retrieval_results = [
        {
            "source_path": "pubblico/api/Controllers/Fattura.cs",
            "score": 0.9,
            "text_hash": "dup",
            "line_start": 1,
            "line_end": 10,
            "chunk_index": 0,
            "snippet": "A",
            "text": "A",
        },
        {
            "source_path": "pubblico/api/Controllers/Fattura.cs",
            "score": 0.8,
            "text_hash": "dup",
            "line_start": 1,
            "line_end": 10,
            "chunk_index": 0,
            "snippet": "A",
            "text": "A",
        },
    ]

    payload = assemble_functional_context(
        query_text="fattura",
        retrieval_results=retrieval_results,
        symbol_results=[],
    )

    assert payload["core_files"][0]["match_count"] == 1
    assert len(payload["core_files"][0]["matches"]) == 1


def test_assemble_functional_context_respects_limits_for_core_files_and_supporting_matches():
    retrieval_results = [
        {
            "source_path": f"file_{index}.cs",
            "score": 1.0 - index * 0.1,
            "text_hash": f"h{index}",
            "line_start": 1,
            "line_end": 5,
            "chunk_index": 0,
            "snippet": f"snippet {index}",
            "text": f"text {index}",
        }
        for index in range(6)
    ]

    payload = assemble_functional_context(
        query_text="feature x",
        retrieval_results=retrieval_results,
        symbol_results=[],
        options=AssemblyOptions(max_core_files=2, max_supporting_matches=2, max_chars=5000),
    )

    assert len(payload["core_files"]) == 2
    assert len(payload["supporting_matches"]) == 2


def test_assemble_functional_context_prefers_files_with_query_overlap_and_symbols():
    retrieval_results = [
        {
            "source_path": "pubblico/api/Controllers/Noise.cs",
            "score": 1.05,
            "text_hash": "noise",
            "line_start": 1,
            "line_end": 10,
            "chunk_index": 0,
            "snippet": "Gestione generica del menu.",
            "text": "public class MenuController { }",
        },
        {
            "source_path": "pubblico/api/Controllers/Fattura.cs",
            "score": 0.91,
            "text_hash": "fattura",
            "line_start": 40,
            "line_end": 70,
            "chunk_index": 0,
            "snippet": "Metodo GeneraFattura per la fatturazione studi.",
            "text": "public void GeneraFattura() { ... }",
        },
    ]
    symbol_results = [
        {
            "source_path": "pubblico/api/Controllers/Fattura.cs",
            "name": "GeneraFattura",
            "kind": "method",
            "signature": "public void GeneraFattura()",
            "line_start": 44,
            "line_end": 60,
        },
        {
            "source_path": "pubblico/api/Controllers/Noise.cs",
            "name": "ListMenu",
            "kind": "method",
            "signature": "public void ListMenu()",
            "line_start": 10,
            "line_end": 20,
        },
    ]

    payload = assemble_functional_context(
        query_text="GeneraFattura fattura",
        retrieval_results=retrieval_results,
        symbol_results=symbol_results,
    )

    assert payload["core_files"][0]["source_path"] == "pubblico/api/Controllers/Fattura.cs"
    assert payload["entry_points"][0]["source_path"] == "pubblico/api/Controllers/Fattura.cs"
    assert payload["core_files"][0]["evidence_kinds"] == ["retrieval", "symbol", "query_overlap"]


def test_assemble_functional_context_prioritizes_entry_points_from_core_files():
    retrieval_results = [
        {
            "source_path": "pubblico/api/Controllers/Fattura.cs",
            "score": 0.95,
            "text_hash": "fattura",
            "line_start": 30,
            "line_end": 50,
            "chunk_index": 0,
            "snippet": "Metodo GeneraFattura.",
            "text": "public void GeneraFattura() { ... }",
        }
    ]
    symbol_results = [
        {
            "source_path": "librerie/BpoFH/OtherService.cs",
            "name": "BootstrapService",
            "kind": "class",
            "signature": "public class BootstrapService",
            "line_start": 1,
            "line_end": 10,
        },
        {
            "source_path": "pubblico/api/Controllers/Fattura.cs",
            "name": "GeneraFattura",
            "kind": "method",
            "signature": "public void GeneraFattura()",
            "line_start": 44,
            "line_end": 60,
        },
    ]

    payload = assemble_functional_context(
        query_text="GeneraFattura",
        retrieval_results=retrieval_results,
        symbol_results=symbol_results,
    )

    assert payload["entry_points"][0]["name"] == "GeneraFattura"
