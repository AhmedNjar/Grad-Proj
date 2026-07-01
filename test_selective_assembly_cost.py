from selective_assembly import SpindleCostModel


def test_total_cost_uses_external_tolerance_cost():
    cost_model = SpindleCostModel()
    var_dict = {
        "L1": 100.0,
        "L2": 200.0,
        "L3": 100.0,
        "L4": 80.0,
        "R1": 20.0,
        "R2": 30.0,
        "R3": 20.0,
        "R4": 15.0,
        "ri": 15.0,
        "rho": 7.85,
        "E": 2.1e5,
        "R2": 30.0,
    }

    costs = cost_model.total_cost(var_dict, sa_results=None, tolerance_cost_usd=123.45)

    assert costs["machining_usd"] == 123.45
    assert costs["total_usd"] > 123.45
