def test_10_imports(load_src_module):
    module = load_src_module("10_hazard_by_fragility_45min.py")
    assert module.MIN_EVENTS_FOR_GLM == 200
