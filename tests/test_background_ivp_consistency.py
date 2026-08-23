import csv


def test_independent_ivp_campaign_converged_and_agrees():
    with open(
        "reports/background/background_ivp_consistency.csv", newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert all(row["ivp_success"] == "True" for row in rows)
    for row in rows:
        assert float(row["mass_scaled_max_difference"]) < 3e-5
        assert float(row["alpha_scaled_max_difference"]) < 3e-6
        assert float(row["psi_scaled_max_difference"]) < 3e-5
        assert float(row["dpsi_scaled_max_difference"]) < 4e-5
