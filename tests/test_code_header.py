import unittest

from rag_indexer.ingest import _build_code_header


class TestCodeHeader(unittest.TestCase):
    def test_python_header_contains_symbols_and_scope(self):
        text = (
            "class BillingService:\n"
            "    def compute_total(self):\n"
            "        return 42\n"
            "\n"
            "async def export_invoice(invoice_id):\n"
            "    return invoice_id\n"
        )
        header = _build_code_header(
            text=text,
            relative_path="services/billing/invoice.py",
            language="python",
        )

        self.assertIn("LANGUAGE: python", header)
        self.assertIn("PROJECT: services", header)
        self.assertIn("SUBPROJECT: billing", header)
        self.assertIn("CLASSES: BillingService", header)
        self.assertIn("FUNCTIONS:", header)
        self.assertIn("compute_total", header)
        self.assertIn("export_invoice", header)

    def test_javascript_header_contains_named_functions(self):
        text = (
            "export class UserApi {}\n"
            "export async function loadUsers() { return []; }\n"
            "const buildPayload = (id) => ({ id });\n"
        )
        header = _build_code_header(
            text=text,
            relative_path="frontend/api/users.ts",
            language="typescript",
        )

        self.assertIn("LANGUAGE: typescript", header)
        self.assertIn("CLASSES: UserApi", header)
        self.assertIn("FUNCTIONS:", header)
        self.assertIn("loadUsers", header)
        self.assertIn("buildPayload", header)


if __name__ == "__main__":
    unittest.main()


