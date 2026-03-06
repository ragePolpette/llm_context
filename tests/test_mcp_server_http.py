from mcp_server_http import _coerce_rpc_message


def test_raw_rpc_arguments_are_wrapped_for_rag_context():
    message = _coerce_rpc_message({"query_text": "bpofh", "top_k": 3})

    assert message["method"] == "tools/call"
    assert message["params"]["name"] == "rag_context"
    assert message["params"]["arguments"]["query_text"] == "bpofh"


def test_raw_rpc_arguments_reject_unknown_keys():
    try:
        _coerce_rpc_message({"query_text": "bpofh", "dsn": "postgres://evil"})
    except ValueError as exc:
        assert "Unsupported /rpc rag_context arguments" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported /rpc argument")


def test_raw_rpc_arguments_require_query_input():
    try:
        _coerce_rpc_message({"top_k": 3})
    except ValueError as exc:
        assert "requires 'query_text' or 'query_embedding'" in str(exc)
    else:
        raise AssertionError("Expected ValueError when query input is missing")


def test_jsonrpc_message_requires_valid_method_and_params():
    try:
        _coerce_rpc_message({"jsonrpc": "2.0", "method": "", "params": []})
    except ValueError as exc:
        assert "non-empty string 'method'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid JSON-RPC envelope")


def test_tools_call_requires_object_arguments():
    try:
        _coerce_rpc_message(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "rag_context", "arguments": "bad"},
            }
        )
    except ValueError as exc:
        assert "params.arguments" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid tools/call arguments")

