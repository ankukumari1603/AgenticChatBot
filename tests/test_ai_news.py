import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.langGraphAgenticAi.news.ai_news_summarizer import (
    REPORTS,
    UNKNOWN_DATE,
    NewsArticle,
    generate_news_report,
    render_markdown_report,
    save_news_report,
)


class AINewsReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.today = date(2026, 11, 4)
        self.search_results = [
            {
                "title": "AI research breakthrough",
                "url": "https://example.com/article?utm_source=feed",
                "published_date": "2026-11-04",
                "content": "Researchers describe a new AI research result.",
            },
            {
                "title": "  AI research BREAKTHROUGH ",
                "url": "https://www.example.com/article/",
                "published_date": "2026-11-04",
                "content": "Duplicate coverage of the same article.",
            },
            {
                "title": "New AI product",
                "url": "https://news.example.org/product",
                "published_date": "2026-10-30",
                "content": "A company announced a new AI product.",
            },
            {
                "title": "AI research update",
                "url": "https://science.example.net/update",
                "published_date": "2026-10-10",
                "content": "An update to earlier AI research.",
            },
            {
                "title": "AI news without a date",
                "url": "https://example.edu/undated",
                "content": "The source provides no publication date.",
            },
            {
                "title": "Unsupported result",
                "url": "javascript:alert(1)",
                "content": "Invalid source link.",
            },
        ]

    def _generate(self, report_type: str):
        client = Mock()
        client.search.return_value = {"results": self.search_results}
        model = FakeListChatModel(
            responses=["A short, source-grounded news summary."] * 5
        )
        articles = generate_news_report(
            report_type,
            client,
            model,
            today=self.today,
        )
        return client, articles

    def test_daily_weekly_and_monthly_reports_use_correct_search_ranges(self) -> None:
        expected_counts = {
            "Daily AI News Summary": 2,
            "Weekly AI News Summary": 3,
            "Monthly AI News Summary": 4,
        }
        for report_type, expected_count in expected_counts.items():
            with self.subTest(report_type=report_type):
                client, articles = self._generate(report_type)
                self.assertEqual(len(articles), expected_count)
                self.assertEqual(
                    client.search.call_args.kwargs["time_range"],
                    REPORTS[report_type][1],
                )
                self.assertEqual(
                    client.search.call_args.kwargs["topic"],
                    "news",
                )

    def test_missing_publication_date_is_not_invented(self) -> None:
        _, articles = self._generate("Monthly AI News Summary")
        undated = next(article for article in articles if "without a date" in article.title)
        self.assertEqual(undated.publication_date, UNKNOWN_DATE)

    def test_reports_are_saved_to_the_correct_markdown_paths(self) -> None:
        article = NewsArticle(
            title="AI | research",
            publication_date="2026-11-04",
            summary="A summary | with detail.",
            url="https://example.com/story",
        )
        expected_names = {
            "Daily AI News Summary": "daily_summary.md",
            "Weekly AI News Summary": "weekly_summary.md",
            "Monthly AI News Summary": "monthly_summary.md",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for report_type, filename in expected_names.items():
                with self.subTest(report_type=report_type):
                    path = save_news_report(
                        report_type,
                        [article],
                        output_dir=Path(temp_dir),
                    )
                    self.assertEqual(path.name, filename)
                    contents = path.read_text(encoding="utf-8")
                    self.assertIn("Article title", contents)
                    self.assertIn("Publication date", contents)
                    self.assertIn("Short summary", contents)
                    self.assertIn("Open article", contents)
                    self.assertIn("AI \\| research", contents)

    def test_empty_report_still_has_table_headers(self) -> None:
        markdown = render_markdown_report(
            "Daily AI News Summary",
            [],
            generated_at=datetime(2026, 11, 4, tzinfo=timezone.utc),
        )
        self.assertIn("No articles with usable titles", markdown)
        self.assertIn("| Article title | Publication date | Short summary | Source |", markdown)


if __name__ == "__main__":
    unittest.main()
