import unittest
import warnings
from unittest.mock import patch

import pandas as pd

from src import train_margin_model as trainer


class MarginTrainingTests(unittest.TestCase):
    def test_training_uses_supported_metric_api_and_saves_artifacts(self):
        data = pd.DataFrame({
            "season": [2018] * 8 + [2019] * 4,
            "form": list(range(8)) + [1, 3, 5, 7],
            "point_diff": [-7, -3, 0, 1, 3, 7, 10, 14, -2, 2, 8, 12],
        })
        with patch.object(trainer, "ensure_directories"), \
             patch.object(trainer.pd, "read_parquet", return_value=data), \
             patch.object(trainer.joblib, "dump") as dump, \
             warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            trainer.train_margin_model()
        self.assertEqual(dump.call_count, 2)
        model, imputer = [call.args[0] for call in dump.call_args_list]
        self.assertNotIn("point_diff", imputer.feature_names_in_)
        self.assertEqual(len(model.predict(imputer.transform(data[imputer.feature_names_in_]))), 12)


if __name__ == "__main__":
    unittest.main()
