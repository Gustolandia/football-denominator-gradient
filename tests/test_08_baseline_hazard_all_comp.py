def test_08_imports(load_src_module):
    module = load_src_module("08_baseline_hazard_all_comp.py")
    assert module.MIN_MINUTES_FOR_ANALYSIS == 900.0
