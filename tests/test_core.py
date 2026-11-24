import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from services.album_recommendation_service import AlbumRecommendationService
from services.recommendation_evaluation_service import RecommendationEvaluationService


class CoreTest(unittest.TestCase):

    def test_precision_recall_f1(self):
        rankings = np.array([[1, 2, 3]])
        metrics = RecommendationEvaluationService.precision_recall_f1(rankings, [np.array([1, 4])], k=3)
        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertAlmostEqual(metrics["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["f1"], 0.4)

    def test_precision_recall_f1_without_hits(self):
        rankings = np.array([[1, 2, 3]])
        metrics = RecommendationEvaluationService.precision_recall_f1(rankings, [np.array([4, 5])], k=3)
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)

    def test_title_problem_rejects_bad_titles(self):
        self.assertEqual(AlbumRecommendationService.title_problem(""), "sin título")
        self.assertEqual(AlbumRecommendationService.title_problem("Tema A / Tema B"), "doble tema con barra")
        self.assertEqual(AlbumRecommendationService.title_problem("http://example.com"), "texto raro")
        self.assertIsNone(AlbumRecommendationService.title_problem("Kind of Blue"))

    def test_edition_key_normalizes_remastered_titles(self):
        first = AlbumRecommendationService.edition_key("Álbum (Deluxe Edition)")
        second = AlbumRecommendationService.edition_key("Album [Remastered]")
        self.assertEqual(first, "album")
        self.assertEqual(second, "album")


if __name__ == "__main__":
    unittest.main()
