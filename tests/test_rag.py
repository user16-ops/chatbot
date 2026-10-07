"""API 비용 없이 인용 검증과 검색 의도 보존을 점검합니다."""

import unittest
from unittest.mock import MagicMock, Mock, patch

import httpx
from langchain_core.documents import Document
from openai import APIConnectionError

import app


class APIKeyTests(unittest.TestCase):
    """실제 API 키를 읽거나 출력하지 않고 키 선택 순서를 확인합니다."""

    def test_local_env_key_has_priority(self):
        secrets = MagicMock()
        with patch.object(app, "dotenv_values", return_value={"OPENAI_API_KEY": " local-test-key "}):
            with patch.object(app.st, "secrets", secrets):
                self.assertEqual(app.get_api_key(), "local-test-key")
        secrets.__getitem__.assert_not_called()

    def test_cloud_secrets_key_is_used_without_local_key(self):
        for local_values in ({}, {"OPENAI_API_KEY": "   "}):
            with self.subTest(local_values=local_values):
                with patch.object(app, "dotenv_values", return_value=local_values):
                    with patch.object(app.st, "secrets", {"OPENAI_API_KEY": " cloud-test-key "}):
                        self.assertEqual(app.get_api_key(), "cloud-test-key")

    def test_missing_secrets_returns_empty_string(self):
        for error in (FileNotFoundError, KeyError):
            with self.subTest(error=error.__name__):
                secrets = MagicMock()
                secrets.__getitem__.side_effect = error("테스트용 누락")
                with patch.object(app, "dotenv_values", return_value={}):
                    with patch.object(app.st, "secrets", secrets):
                        self.assertEqual(app.get_api_key(), "")


class RAGRegressionTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document(
            page_content="근무지내 국내출장은 정액으로 지급한다.\n별도의 식비는 지급하지 아니함.",
            metadata={"source": "기준.pdf", "page": 41, "basis_date": "2024년"},
        )

    def test_expansion_keeps_original_and_does_not_invent_conditions(self):
        question = "근무지내 출장 시 점심은 자비 부담이니? 출장비 정산 가능하니?"
        queries = app.expand_queries(question)
        self.assertEqual(queries[0], question)
        self.assertTrue(any("근무지 내 국내출장" in q and "식비" in q for q in queries))
        self.assertTrue(any("정액 지급" in q for q in queries))
        self.assertFalse(any("2km" in q or "4시간" in q or "공용차량" in q for q in queries))
        self.assertEqual(app.expand_queries("파이썬 문법은 무엇인가요?"), ["파이썬 문법은 무엇인가요?"])

    def test_quotes_allow_whitespace_only_and_preserve_original(self):
        self.assertEqual(
            app.find_original_quote(self.doc.page_content, "근무지 내 국내출장은 정액으로 지급한다."),
            "근무지내 국내출장은 정액으로 지급한다.",
        )
        self.assertIsNone(app.find_original_quote(self.doc.page_content, "별도의 식비도 지급한다."))
        self.assertIsNone(app.find_original_quote("4시간 이상인 경우 2만원을 지급한다.", "4시간 이상인 경우 3만원을 지급한다."))

    def test_only_claims_depending_on_invalid_citations_are_removed(self):
        good = app.SupportedClaim(
            section="일반 원칙", text="정액으로 지급합니다.",
            evidence=[app.Evidence(source_id="D1", quote="근무지내 국내출장은 정액으로 지급한다.")],
        )
        bad = app.SupportedClaim(
            section="예외", text="원문에 없는 주장입니다.",
            evidence=[good.evidence[0], app.Evidence(source_id="D1", quote="식비도 무제한으로 지급한다.")],
        )
        result = app.verify_answer(app.RAGAnswer(claims=[good, bad], questions=[]), {"D1": self.doc})
        self.assertEqual(result.status, "PARTIAL_CITATION")
        self.assertIn("정액으로 지급합니다. [1]", result.answer)
        self.assertNotIn("원문에 없는 주장", result.answer)
        self.assertEqual(len(result.evidence), 1)

    def test_failure_reasons_are_distinct(self):
        insufficient = app.verify_answer(app.RAGAnswer(claims=[], questions=[]), {"D1": self.doc})
        invalid = app.RAGAnswer(claims=[app.SupportedClaim(
            section="결론", text="잘못된 인용의 답변", evidence=[app.Evidence(source_id="D9", quote="존재하지 않는 문서의 원문입니다.")],
        )], questions=[])
        self.assertEqual(insufficient.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(app.verify_answer(invalid, {"D1": self.doc}).status, "CITATION_FAILED")
        with patch.object(app, "retrieve_pages", return_value=([], {})):
            self.assertEqual(app.answer_question("질문", Mock(), "not-a-key").status, "RETRIEVAL_EMPTY")
        with patch.object(app, "retrieve_pages", side_effect=RuntimeError("sensitive-value")):
            with self.assertLogs("rag_chatbot", level="WARNING") as log:
                result = app.answer_question("질문", Mock(), "not-a-key")
            self.assertEqual(result.status, "RETRIEVAL_ERROR")
            self.assertNotIn("sensitive-value", "".join(log.output))

    def test_retrieval_includes_neighbor_pages_once(self):
        exception = Document(page_content="예외 규정", metadata={"source": "기준.pdf", "page": 42})
        store = Mock()
        store.embeddings.embed_documents.return_value = [[0.1]]
        store.similarity_search_with_score_by_vector.return_value = [(self.doc, 0.9)]
        index = app.SearchIndex(store, {("기준.pdf", 41): self.doc, ("기준.pdf", 42): exception}, [self.doc])
        docs, _ = app.retrieve_pages("정액 기준", index)
        self.assertEqual([doc.metadata["page"] for doc in docs], [41, 42])

    def test_unrelated_low_similarity_question_is_insufficient_evidence(self):
        store = Mock()
        store.embeddings.embed_documents.return_value = [[0.1]]
        store.similarity_search_with_score_by_vector.return_value = [(self.doc, 0.15)]
        index = app.SearchIndex(store, {("기준.pdf", 41): self.doc}, [self.doc])
        result = app.answer_question("파이썬 문법을 알려줘", index, "not-a-key")
        self.assertEqual(result.status, "INSUFFICIENT_EVIDENCE")
        self.assertFalse(result.evidence)

    def test_selected_passages_are_exact_source_text(self):
        _, passages = app.make_context([self.doc])
        for doc in passages.values():
            self.assertIsNotNone(app.find_original_quote(self.doc.page_content, doc.page_content))

    def test_semantic_review_drops_only_rejected_sentence(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = app.GroundingReview(decisions=[
            app.ClaimCheck(claim_number=1, issue="없음", supported=True),
            app.ClaimCheck(claim_number=2, issue="근거에 없는 조건", supported=False),
        ])
        draft = app.DraftAnswer(claims=[app.DraftClaim(
            section="일반 원칙", evidence_ids=["D1"],
            text="정액으로 지급합니다. 모든 식비도 별도로 지급합니다.",
        )])
        generated = app.review_claims(draft, {"D1": self.doc}, model)
        result = app.verify_answer(generated, {"D1": self.doc})
        self.assertEqual(result.status, "PARTIAL_EVIDENCE")
        self.assertIn("정액으로 지급합니다.", result.answer)
        self.assertNotIn("모든 식비", result.answer)

    def test_api_errors_do_not_expose_exception_or_key(self):
        error = APIConnectionError(
            message="secret-value", request=httpx.Request("POST", "https://api.openai.com"),
        )
        with patch.object(app, "retrieve_pages", side_effect=error):
            with self.assertLogs("rag_chatbot", level="WARNING") as log:
                result = app.answer_question("질문", Mock(), "private-key")
        self.assertEqual(result.status, "API_ERROR")
        self.assertNotIn("secret-value", result.answer + "".join(log.output))
        self.assertNotIn("private-key", result.answer + "".join(log.output))

    def test_missing_conditions_are_questions_not_assumptions(self):
        evidence = [{"quote": "4시간, 왕복 2km, 공용차량과 증거서류를 확인한다."}]
        self.assertEqual(len(app.missing_condition_questions("출장 식비", evidence)), 3)
        questions = app.missing_condition_questions("공용차량으로 왕복 2km, 4시간 출장", evidence)
        self.assertEqual(len(questions), 1)
        self.assertIn("증빙", questions[0])

    def test_reimbursement_scope_and_fraction_are_not_expanded(self):
        evidence = [app.Evidence(source_id="D1", quote=(
            "식비(1/3) 범위 내 실비 지급. 실비 상한액은 2만원이다. "
            "다만, 운임은 증거서류 구비가 어려울 경우 출장확인서를 근거로 지급한다."
        ))]
        self.assertFalse(app.preserves_scope("증거서류 없이 출장확인서로 정산할 수 있습니다.", evidence))
        self.assertTrue(app.preserves_scope("운임은 출장확인서로 정산할 수 있습니다.", evidence))
        self.assertFalse(app.preserves_scope("운임 및 식비를 2만원 한도로 실비 지급합니다.", evidence))
        self.assertTrue(app.preserves_scope("운임 및 식비(1/3)를 2만원 한도로 실비 지급합니다.", evidence))
        self.assertFalse(app.preserves_scope("운임 실비로 2만원을 지급합니다.", evidence))


if __name__ == "__main__":
    unittest.main()
