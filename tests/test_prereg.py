"""Pre-registration gate tests (task §10): missing file and each missing key."""

from __future__ import annotations

import yaml
import pytest

from failure_mech import prereg as PR


def test_missing_file_raises_naming_it(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(PR.PreregError) as exc:
        PR.load_prereg(missing)
    assert str(missing) in str(exc.value)


def test_valid_file_loads(prereg_file):
    cfg = PR.load_prereg(prereg_file)
    assert PR.get(cfg, "sampling.stage1_n_per_cell") == 30
    assert PR.get(cfg, "causal_criteria.alpha") == 0.05


@pytest.mark.parametrize("dotted", PR.REQUIRED_KEYS)
def test_each_missing_key_raises_naming_it(tmp_path, valid_prereg_dict, dotted):
    # delete exactly one required key, write, expect PreregError naming it
    cfg = valid_prereg_dict
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur[p]
    del cur[parts[-1]]
    path = tmp_path / "preregistration.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(PR.PreregError) as exc:
        PR.load_prereg(path)
    assert dotted in str(exc.value)


def test_null_value_counts_as_missing(tmp_path, valid_prereg_dict):
    valid_prereg_dict["causal_criteria"]["alpha"] = None
    path = tmp_path / "preregistration.yaml"
    path.write_text(yaml.safe_dump(valid_prereg_dict), encoding="utf-8")
    with pytest.raises(PR.PreregError) as exc:
        PR.load_prereg(path)
    assert "causal_criteria.alpha" in str(exc.value)
