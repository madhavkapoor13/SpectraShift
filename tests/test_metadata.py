import pandas as pd
import pytest

from spectrashift.data.metadata import enrich_metadata, parse_patch_ids


def test_parse_patch_id_contract() -> None:
    patch_id = "S2B_MSIL2A_20180421T100029_N9999_R122_T33TWM_00_07"
    row = parse_patch_ids(pd.Series([patch_id])).iloc[0]
    assert row["mgrs_tile"] == "T33TWM"
    assert row["orbit"] == "122"
    assert row["h_order"] == 0
    assert row["v_order"] == 7
    assert row["location_key"] == "T33TWM_0_7"


def test_repeated_acquisitions_share_location_key() -> None:
    ids = pd.Series(
        [
            "S2A_MSIL2A_20170613T101031_N9999_R022_T33TWM_03_05",
            "S2B_MSIL2A_20180701T101029_N9999_R022_T33TWM_03_05",
        ]
    )
    parsed = parse_patch_ids(ids)
    assert parsed["location_key"].nunique() == 1


def test_invalid_id_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unparseable"):
        parse_patch_ids(pd.Series(["bad-id"]))


def test_duplicate_patch_ids_fail() -> None:
    row = {
        "patch_id": "S2B_MSIL2A_20180421T100029_N9999_R122_T33TWM_00_07",
        "labels": ["Pastures"],
        "split": "train",
        "country": "Austria",
    }
    with pytest.raises(ValueError, match="duplicate"):
        enrich_metadata(pd.DataFrame([row, row]))

