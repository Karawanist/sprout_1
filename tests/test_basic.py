import pandas as pd
from src.knowledge import compare_to_profile

PROFILE = {"temp_optimal_c": [18, 27], "verified": False}

def test_hot_weather_flagged():
    frame = pd.DataFrame({"T2M_MAX": [36, 37, 38], "T2M_MIN": [24, 25, 24]})
    notes = compare_to_profile(frame, PROFILE)
    assert any("above" in note for note in notes)

def test_unverified_profile_warns():
    frame = pd.DataFrame({"T2M_MAX": [22], "T2M_MIN": [15]})
    assert any("not been verified" in note for note in compare_to_profile(frame, PROFILE))
