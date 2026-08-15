def test_00_imports(load_src_module):
    module = load_src_module("00_list_result_columns.py")
    assert callable(module.main)
