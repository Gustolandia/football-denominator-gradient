def test_02_imports(load_src_module):
    module = load_src_module("02_build_player_index.py")
    assert callable(module.main)
