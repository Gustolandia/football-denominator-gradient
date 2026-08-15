def test_05_imports(load_src_module):
    module = load_src_module("05_clean_injuries.py")
    assert callable(module.main)
