"""Tests de la métrique MAE et du rapport baseline."""
import numpy as np

from defia.evaluation.metrics import mae, mae_report


def test_mae_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mae_basic():
    assert mae([0, 0, 0], [1, 1, 1]) == 1.0


def test_mae_report_gain_positive_when_model_beats_baseline():
    y = np.array([1.0, 1.0, 10.0])
    # modèle parfait vs baseline constant = 1
    rep = mae_report(y, y, baseline_value=1.0)
    assert rep["mae"] == 0.0
    assert rep["mae_baseline"] > 0.0
    assert rep["gain_rel"] == 1.0
