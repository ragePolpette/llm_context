from rag_indexer.context_assembler import AssemblyOptions, assemble_functional_context


def test_assemble_functional_context_groups_matches_by_file_and_attaches_symbols():
    retrieval_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "score": 0.92,
            "text_hash": "a1",
            "line_start": 10,
            "line_end": 30,
            "chunk_index": 0,
            "section_path": "",
            "snippet": "Billing workflow entry point.",
            "text": "public class BillingController { ... }",
        },
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "score": 0.74,
            "text_hash": "a2",
            "line_start": 40,
            "line_end": 70,
            "chunk_index": 1,
            "section_path": "",
            "snippet": "Metodo GenerateInvoice.",
            "text": "public void GenerateInvoice() { ... }",
        },
        {
            "source_path": "src/domain/services/BillingService.cs",
            "score": 0.68,
            "text_hash": "b1",
            "line_start": 15,
            "line_end": 40,
            "chunk_index": 0,
            "section_path": "",
            "snippet": "Billing service orchestration.",
            "text": "public class BillingService { ... }",
        },
    ]
    symbol_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "name": "GenerateInvoice",
            "kind": "method",
            "signature": "public void GenerateInvoice()",
            "line_start": 44,
            "line_end": 60,
        }
    ]

    payload = assemble_functional_context(
        query_text="GenerateInvoice billing",
        retrieval_results=retrieval_results,
        symbol_results=symbol_results,
    )

    assert payload["summary"]["core_file_count"] == 2
    assert payload["package_version"] == "functional_context_v1"
    assert payload["query"]["terms"] == ["generateinvoice", "billing"]
    assert payload["core_files"][0]["source_path"] == "src/api/controllers/BillingController.cs"
    assert payload["core_files"][0]["match_count"] == 2
    assert payload["core_files"][0]["functional_role"] == "entry_point"
    assert payload["core_files"][0]["symbol_hits"][0]["name"] == "GenerateInvoice"
    assert payload["core_files"][0]["query_overlap_terms"] == ["generateinvoice", "billing"]
    assert "query_overlap=" in payload["core_files"][0]["selection_reason"]
    assert payload["summary"]["role_counts"]["entry_point"] == 1
    assert payload["entry_points"][0]["name"] == "GenerateInvoice"
    assert payload["summary"]["query_term_count"] == 2
    assert "FILE src/api/controllers/BillingController.cs" in payload["assembled_context"]
    assert "functional_role=entry_point" in payload["assembled_context"]
    assert "selection_reason=" in payload["assembled_context"]


def test_assemble_functional_context_deduplicates_repeated_chunk_hashes():
    retrieval_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "score": 0.9,
            "text_hash": "dup",
            "line_start": 1,
            "line_end": 10,
            "chunk_index": 0,
            "snippet": "A",
            "text": "A",
        },
        {
            "source_path": "src/api/controllers/BillingController.cs",
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
        query_text="billing",
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
            "source_path": "src/ui/controllers/NoiseController.cs",
            "score": 1.05,
            "text_hash": "noise",
            "line_start": 1,
            "line_end": 10,
            "chunk_index": 0,
            "snippet": "Gestione generica del menu.",
            "text": "public class MenuController { }",
        },
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "score": 0.91,
            "text_hash": "billing",
            "line_start": 40,
            "line_end": 70,
            "chunk_index": 0,
            "snippet": "Method GenerateInvoice for the billing workflow.",
            "text": "public void GenerateInvoice() { ... }",
        },
    ]
    symbol_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "name": "GenerateInvoice",
            "kind": "method",
            "signature": "public void GenerateInvoice()",
            "line_start": 44,
            "line_end": 60,
        },
        {
            "source_path": "src/ui/controllers/NoiseController.cs",
            "name": "ListMenu",
            "kind": "method",
            "signature": "public void ListMenu()",
            "line_start": 10,
            "line_end": 20,
        },
    ]

    payload = assemble_functional_context(
        query_text="GenerateInvoice billing",
        retrieval_results=retrieval_results,
        symbol_results=symbol_results,
    )

    assert payload["core_files"][0]["source_path"] == "src/api/controllers/BillingController.cs"
    assert payload["entry_points"][0]["source_path"] == "src/api/controllers/BillingController.cs"
    assert payload["core_files"][0]["evidence_kinds"] == ["retrieval", "symbol", "query_overlap"]
    assert payload["core_files"][0]["functional_role"] == "entry_point"


def test_assemble_functional_context_prioritizes_entry_points_from_core_files():
    retrieval_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "score": 0.95,
            "text_hash": "billing",
            "line_start": 30,
            "line_end": 50,
            "chunk_index": 0,
            "snippet": "Metodo GenerateInvoice.",
            "text": "public void GenerateInvoice() { ... }",
        }
    ]
    symbol_results = [
        {
            "source_path": "src/domain/services/BootstrapService.cs",
            "name": "BootstrapService",
            "kind": "class",
            "signature": "public class BootstrapService",
            "line_start": 1,
            "line_end": 10,
        },
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "name": "GenerateInvoice",
            "kind": "method",
            "signature": "public void GenerateInvoice()",
            "line_start": 44,
            "line_end": 60,
        },
    ]

    payload = assemble_functional_context(
        query_text="GenerateInvoice",
        retrieval_results=retrieval_results,
        symbol_results=symbol_results,
    )

    assert payload["entry_points"][0]["name"] == "GenerateInvoice"


def test_assemble_functional_context_assigns_functional_roles_for_multi_file_flow():
    retrieval_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "score": 0.89,
            "text_hash": "controller-hit",
            "line_start": 10,
            "line_end": 30,
            "chunk_index": 0,
            "snippet": "Billing API entry point.",
            "text": "public class BillingController { ... }",
        },
        {
            "source_path": "src/domain/services/BillingService.cs",
            "score": 0.87,
            "text_hash": "service-hit",
            "line_start": 50,
            "line_end": 90,
            "chunk_index": 0,
            "snippet": "Service that executes GenerateInvoice and billing calculations.",
            "text": "public class BillingService { ... }",
        },
        {
            "source_path": "src/api/contracts/IBillingService.cs",
            "score": 0.78,
            "text_hash": "contract-hit",
            "line_start": 1,
            "line_end": 20,
            "chunk_index": 0,
            "snippet": "Contract for the billing API.",
            "text": "public interface IBillingService { ... }",
        },
    ]
    symbol_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "name": "GenerateInvoice",
            "kind": "method",
            "signature": "public void GenerateInvoice()",
            "line_start": 44,
            "line_end": 60,
        },
        {
            "source_path": "src/domain/services/BillingService.cs",
            "name": "BillingService",
            "kind": "class",
            "signature": "public class BillingService",
            "line_start": 1,
            "line_end": 120,
        },
        {
            "source_path": "src/api/contracts/IBillingService.cs",
            "name": "IBillingService",
            "kind": "interface",
            "signature": "public interface IBillingService",
            "line_start": 1,
            "line_end": 40,
        },
    ]

    payload = assemble_functional_context(
        query_text="GenerateInvoice billing api",
        retrieval_results=retrieval_results,
        symbol_results=symbol_results,
    )

    roles = {item["source_path"]: item["functional_role"] for item in payload["core_files"]}
    assert roles["src/api/controllers/BillingController.cs"] == "entry_point"
    assert roles["src/domain/services/BillingService.cs"] == "implementation"
    assert roles["src/api/contracts/IBillingService.cs"] == "contract"
    assert payload["summary"]["role_counts"]["entry_point"] == 1
    assert payload["summary"]["role_counts"]["implementation"] == 1
    assert payload["summary"]["role_counts"]["contract"] == 1


