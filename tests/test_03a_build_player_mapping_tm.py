def test_03a_imports(load_src_module):
    module = load_src_module("03a_build_player_mapping_tm.py")
    assert callable(module.main)
