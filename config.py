"""Central path and legacy constants for the EPL congestion pipeline.

The current production pipeline reads local Transfermarkt dataset dumps under
``external_data/transfermarkt`` and writes generated files under ``data/``.
Some constant names still reference FBref because early versions of the project
used a FBref-oriented design. Treat those names as backwards-compatible labels,
not as evidence that the current pipeline calls FBref.
"""

from pathlib import Path

# Base directories used by every numbered pipeline script.
DATA_RAW = Path("data") / "raw"
DATA_PROCESSED = Path("data") / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Legacy season list retained for compatibility with early FBref-oriented work.
# The active manuscript analysis starts in 2017-18 because earlier public
# Transfermarkt coverage is treated as lower confidence for this project.
SEASONS_FBREF = ["1718", "1819", "1920", "2021", "2122", "2223", "2324"]
ANALYSIS_START_SEASON = 2017
ANALYSIS_START_DATE = "2017-07-01"

# Transfermarkt competition ID for Premier League
TM_COMP_ID = "GB1"
