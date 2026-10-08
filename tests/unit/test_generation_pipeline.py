"""Unit tests for generation pipeline schema wiring."""

from pathlib import Path

import pandas as pd
import pytest

from synthdata.generation.pipeline import run_generation

pytestmark = pytest.mark.unit


def _configure_tabpfn_only(cfg):
    cfg.generation.synthcity.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.tabpfn.enabled = True
    cfg.generation.tabpfn.variants = ["custom"]
    cfg.generation.tabpfn.data_variants = ["raw"]
    cfg.generation.match_class_prior = False


def _set_schema(dataset, *, target_kind):
    dataset.variable_schema = {
        column: {
            "kind": (
                target_kind
                if column == "target"
                else "categorical"
                if column == "smoker"
                else "continuous"
            ),
            "ordinal_order": None,
        }
        for column in dataset.feature_columns + [dataset.target_column]
    }
    dataset.nominal_columns = ["smoker"]
    dataset.ordinal_columns = []


def test_tabpfn_generation_forwards_schema_derived_feature_roles(make_config, make_dataset, mocker):
    cfg = make_config()
    _configure_tabpfn_only(cfg)
    dataset = make_dataset(nominal_columns=["smoker"])
    _set_schema(dataset, target_kind="categorical")
    generated = dataset.train_df.copy()
    mock_generate = mocker.patch(
        "synthdata.generation.pipeline.tpfn.generate_tabpfn_custom",
        return_value=(generated, None),
    )

    result = run_generation(cfg, dataset)

    mock_generate.assert_called_once()
    arguments, keyword_arguments = mock_generate.call_args
    assert arguments[1] == ["smoker"]
    assert "group" not in arguments[1]
    assert arguments[2] == "target"
    assert arguments[3] == cfg.generation.n_samples
    assert keyword_arguments["target_is_categorical"] is True
    assert keyword_arguments["variable_schema_fingerprint"] is None
    assert result["tabpfn_custom"].equals(generated)


def test_continuous_target_fails_before_tabpfn_cache_lookup(make_config, make_dataset, mocker):
    cfg = make_config()
    _configure_tabpfn_only(cfg)
    dataset = make_dataset()
    _set_schema(dataset, target_kind="continuous")

    output_path = Path(cfg.generation.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cached_path = output_path / "tabpfn_custom.csv"
    cached_contents = "cached,target\n1,0\n"
    cached_path.write_text(cached_contents)
    mock_generate = mocker.patch("synthdata.generation.pipeline.tpfn.generate_tabpfn_custom")

    with pytest.raises(ValueError, match="target column 'target'.*continuous"):
        run_generation(cfg, dataset)

    mock_generate.assert_not_called()
    assert cached_path.read_text() == cached_contents


def test_each_replicate_is_built_with_its_own_seed(make_config, make_dataset, mocker):
    cfg = make_config()
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.n_replicates = 3
    dataset = make_dataset()
    dataset.train_imputed_df = dataset.train_df
    mocker.patch("synthdata.generation.pipeline.sc.make_loader")
    mock_fit = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate",
        side_effect=lambda *args, **kwargs: dataset.train_df.assign(seed=args[4]),
    )

    result = run_generation(cfg, dataset)

    assert sorted(result) == ["ctgan", "ctgan__rep1", "ctgan__rep2"]
    assert [call.args[4] for call in mock_fit.call_args_list] == [
        cfg.seed,
        cfg.seed + 1,
        cfg.seed + 2,
    ]
    assert result["ctgan__rep2"]["seed"].eq(cfg.seed + 2).all()
    output_dir = Path(cfg.generation.output_dir)
    assert (output_dir / "ctgan__rep1.csv").exists()


def test_saved_synthetic_data_gets_the_train_class_shares(make_config, make_dataset, mocker):
    cfg = make_config()
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.synthcity.names = ["ctgan"]
    dataset = make_dataset()
    dataset.train_imputed_df = dataset.train_df
    target = dataset.target_column
    classes = sorted(dataset.train_df[target].unique())
    # A generator that rebalances: equal rows per class.
    balanced = pd.concat(
        [
            dataset.train_df[dataset.train_df[target] == c].sample(30, replace=True, random_state=0)
            for c in classes
        ]
    )
    mocker.patch("synthdata.generation.pipeline.sc.make_loader")
    mocker.patch("synthdata.generation.pipeline.sc.fit_generate", return_value=balanced)

    result = run_generation(cfg, dataset)

    shares = result["ctgan"][target].value_counts(normalize=True)
    expected = dataset.train_df[target].value_counts(normalize=True)
    assert len(result["ctgan"]) == len(balanced)
    for c in classes:
        assert abs(shares[c] - expected[c]) < 1 / len(balanced) + 1e-9
