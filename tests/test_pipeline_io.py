import pandas as pd
import pytest


def test_merge_and_restrict_fragility(load_src_module, tmp_path):
    module = load_src_module("pipeline_io.py")
    panel = pd.DataFrame(
        {
            "tm_player_id": [1],
            "date": ["2024-01-01"],
            "fragility_group": ["old"],
            "q1_freq_frag": [99],
        }
    )
    with pytest.raises(FileNotFoundError):
        module.merge_day_fragility(panel, tmp_path)

    day = pd.DataFrame(
        {
            "tm_player_id": [1],
            "date": ["2024-01-01"],
            "fragility_group": ["regular"],
            "prior_minutes_played": [900],
            "prior_n_spells": [1],
            "prior_total_days_injured": [2],
            "prior_max_spell_duration_days": [2],
            "prior_years_at_risk": [1],
            "prior_injuries_per_year": [1],
            "prior_injuries_per_10000min": [1],
            "q1_freq": [0],
            "q3_freq": [2],
            "q1_sev": [0],
            "q3_sev": [2],
        }
    )
    day.to_csv(tmp_path / "player_day_fragility.csv", index=False)
    merged = module.merge_day_fragility(panel, tmp_path)
    assert merged.loc[0, "fragility_group"] == "regular"
    assert "q1_freq_frag" not in merged.columns

    clean_panel = pd.DataFrame({"tm_player_id": [1], "date": ["2024-01-01"]})
    clean_merged = module.merge_day_fragility(clean_panel, tmp_path)
    assert clean_merged.loc[0, "prior_minutes_played"] == 900

    with pytest.raises(KeyError):
        module.restrict_to_fragility_risk_set(pd.DataFrame({"x": [1]}))
    restricted = module.restrict_to_fragility_risk_set(
        pd.DataFrame({"fragility_group": ["regular", "low_exposure"]})
    )
    assert restricted["fragility_group"].tolist() == ["regular"]

    with pytest.raises(KeyError):
        module.restrict_to_available_risk_set(pd.DataFrame({"x": [1]}))
    available = module.restrict_to_available_risk_set(
        pd.DataFrame({"available_for_injury_risk": [1, 0, True]})
    )
    assert available["available_for_injury_risk"].tolist() == [1, True]


def test_45min_bins_and_estimability(load_src_module):
    module = load_src_module("pipeline_io.py")
    with pytest.raises(KeyError):
        module.add_45min_load_bins(pd.DataFrame({"x": [1]}))

    panel = module.add_45min_load_bins(
        pd.DataFrame(
            {
                "all_minutes_last_7d": [0, 50, 50, 275],
                "injury_event": [0, 1, 0, 0],
            }
        )
    )
    assert panel["all_minutes7d_bin"].astype(str).tolist() == [
        "0-45",
        "46-90",
        "46-90",
        "271-300",
    ]

    labels, counts = module.estimable_bin_labels(panel, min_days=1, min_events=1)
    assert labels == ["46-90"]
    assert counts.loc[counts["all_minutes7d_bin"].astype(str) == "271-300", "estimable"].iloc[0] == False

    with pytest.raises(KeyError):
        module.estimable_bin_labels(pd.DataFrame({"all_minutes7d_bin": ["0-45"]}))


def test_publication_history_labels(load_src_module):
    module = load_src_module("pipeline_io.py")

    labelled = module.add_publication_history_labels(
        pd.DataFrame({"history": ["regular", "custom"]}),
        "history",
    )
    assert labelled["publication_history_stratum"].tolist() == [
        "intermediate prior-injury-history",
        "custom",
    ]
    collapsed = module.add_publication_history_labels(
        pd.DataFrame({"history": ["lower_intermediate_history"]}),
        "history",
    )
    assert collapsed["publication_history_stratum"].iloc[0] == (
        "lower/intermediate prior-injury-history"
    )


def test_injury_episode_reconciliation_and_collapse(load_src_module):
    module = load_src_module("pipeline_io.py")
    injuries = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 1, 2, None],
            "injury_spell_id": ["a", "b", "c", "d", "e", "bad"],
            "start_date": [
                "2024-01-01",
                "2024-01-03",
                "2024-01-10",
                "2024-01-20",
                "2024-01-05",
                "2024-01-01",
            ],
            "end_date": [
                "2024-01-15",
                "2024-01-04",
                "2024-01-15",
                "2024-01-19",
                None,
                "2024-01-02",
            ],
            "injury_desc": ["A", "B", "C", "D", None, "bad"],
            "missedGamesCount": [2, 3, 4, None, 1, 1],
        }
    )
    appearances = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 2, None],
            "date": ["2024-01-01", "2024-01-10", "2024-02-01", "2024-01-07", None],
        }
    )
    episodes = module.build_injury_episodes(
        injuries,
        appearance_days=appearances,
        min_date=pd.Timestamp("2024-01-01"),
        max_date=pd.Timestamp("2024-01-31"),
    )

    player1 = episodes[episodes["tm_player_id"] == 1].reset_index(drop=True)
    assert len(player1) == 3
    assert player1.loc[0, "source_spell_ids"] == "a;b"
    assert player1.loc[0, "missedGamesCount"] == 3
    assert player1.loc[0, "end_date"] == pd.Timestamp("2024-01-09")
    assert bool(player1.loc[0, "return_truncated"])
    assert player1.loc[1, "start_date"] == pd.Timestamp("2024-01-10")
    assert player1.loc[2, "duration_days"] == 1
    assert episodes.loc[episodes["tm_player_id"] == 2, "injury_desc"].iloc[0] == ""

    starts = module.injury_episode_start_table(episodes)
    assert starts["n_injury_spells"].sum() == len(episodes)
    unavailable = module.expand_injury_episode_days(
        episodes,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
    )
    assert not ((unavailable["tm_player_id"] == 1) & (unavailable["date"] == pd.Timestamp("2024-01-10"))).any()


def test_injury_episode_defensive_paths(load_src_module):
    module = load_src_module("pipeline_io.py")
    with pytest.raises(KeyError, match="start_date"):
        module.build_injury_episodes(pd.DataFrame({"tm_player_id": [1]}))

    empty = module.build_injury_episodes(
        pd.DataFrame(columns=["tm_player_id", "start_date"])
    )
    assert list(empty.columns) == module.INJURY_EPISODE_COLUMNS

    filtered = module.build_injury_episodes(
        pd.DataFrame(
            {
                "tm_player_id": [1, 1],
                "start_date": [None, "2023-01-01"],
            }
        ),
        min_date=pd.Timestamp("2024-01-01"),
    )
    assert filtered.empty

    with pytest.raises(KeyError, match="appearance_days"):
        module.build_injury_episodes(
            pd.DataFrame({"tm_player_id": [1], "start_date": ["2024-01-01"]}),
            appearance_days=pd.DataFrame({"tm_player_id": [1]}),
        )

    no_return = module.build_injury_episodes(
        pd.DataFrame(
            {
                "tm_player_id": [1, 1],
                "start_date": ["2024-01-01", "2024-01-05"],
                "end_date": ["2024-01-02", "2024-01-06"],
            }
        ),
        appearance_days=pd.DataFrame(
            {"tm_player_id": [1], "date": ["2024-02-01"]}
        ),
    )
    assert len(no_return) == 2
    assert not no_return["return_truncated"].any()

    assert module.injury_episode_start_table(empty).empty
    with pytest.raises(KeyError, match="injury_episode_id"):
        module.injury_episode_start_table(
            pd.DataFrame(
                {
                    "tm_player_id": [1],
                    "start_date": ["2024-01-01"],
                    "injury_desc": ["x"],
                }
            )
        )
    assert module.expand_injury_episode_days(
        empty, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")
    ).empty
    with pytest.raises(KeyError, match="end_date"):
        module.expand_injury_episode_days(
            pd.DataFrame({"tm_player_id": [1], "start_date": ["2024-01-01"]}),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-31"),
        )
    no_days = module.expand_injury_episode_days(
        pd.DataFrame(
            {
                "tm_player_id": [1],
                "start_date": pd.to_datetime(["2024-01-01"]),
                "end_date": pd.to_datetime(["2024-01-01"]),
            }
        ),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
    )
    assert no_days.empty
