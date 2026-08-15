import config


def test_config_paths_and_constants():
    assert config.DATA_RAW.name == "raw"
    assert config.DATA_PROCESSED.name == "processed"
    assert config.TM_COMP_ID == "GB1"
    assert "1718" in config.SEASONS_FBREF
    assert config.ANALYSIS_START_SEASON == 2017
    assert config.ANALYSIS_START_DATE == "2017-07-01"
