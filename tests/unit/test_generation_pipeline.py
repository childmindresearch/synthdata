"""Unit tests for generation pipeline schema wiring."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from synthdata.data import semantic_context_payload
from synthdata.generation import synthcity_backend as sc
from synthdata.generation.pipeline import refit_selected_model, run_generation

pytestmark = pytest.mark.unit


def _pategan_generator_metadata(n_samples: int, params: dict | None = None) -> dict:
    parameters = dict(params or {})
    requested_epsilon = parameters.get("epsilon", 1.0)
    requested_delta = parameters.get("delta")
    requested_alpha = parameters.get("alpha", 100)
    requested_lamda = parameters.get("lamda", 0.001)
    accounting = {
        "schema_version": "pate-accounting-v1",
        "privacy_claim_type": "formal_dp",
        "accountant": "pate_moments_v1",
        "requested_epsilon": requested_epsilon,
        "requested_delta": requested_delta,
        "requested_alpha": requested_alpha,
        "requested_lamda": requested_lamda,
        "resolved_epsilon": requested_epsilon,
        "resolved_delta": requested_delta or 1e-6,
        "resolved_alpha": requested_alpha,
        "resolved_lamda": requested_lamda,
        "effective_epsilon": 1.2,
        "effective_delta": requested_delta or 1e-6,
        "effective_alpha": requested_alpha,
        "effective_lamda": requested_lamda,
        "iterations": 2,
        "max_iter": 10,
        "stopping_state": "epsilon_reached",
    }
    return {
        "schema_version": sc.GENERATOR_METADATA_SCHEMA_VERSION,
        "generator_context": sc.generator_metadata_context("pategan", parameters),
        "plugin_name": "pategan",
        "plugin_fqdn": "privacy.pategan",
        "requested_parameters": parameters,
        "n_samples": n_samples,
        "random_state": 42,
        "privacy_accounting": accounting,
    }


def _configure_tabpfn_only(cfg):
    cfg.generation.synthcity.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.tabpfn.enabled = True
    cfg.generation.tabpfn.variants = ["custom"]
    cfg.generation.tabpfn.data_variants = ["raw"]


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
    cfg.generation.n_samples = len(generated)
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
    assert keyword_arguments["semantic_context"] == semantic_context_payload(
        dataset,
        classification_score=cfg.evaluation.synthcity.classification_score,
        roles=("train",),
    )
    assert result["tabpfn_custom"].equals(generated)


def test_tabpfgen_generation_forwards_complete_semantic_context(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.enabled = False
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = True
    cfg.generation.tabpfgen.variants = ["standard"]
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    generated = dataset.role_frame("train", imputed=True).head(cfg.generation.n_samples).copy()
    mock_generate = mocker.patch(
        "synthdata.generation.tabpfgen_backend.generate_tabpfgen_standard",
        return_value=generated,
    )

    run_generation(cfg, dataset)

    keyword_arguments = mock_generate.call_args.kwargs
    assert keyword_arguments["variable_schema_fingerprint"] == dataset.variable_schema_fingerprint
    assert keyword_arguments["semantic_context"] == semantic_context_payload(
        dataset,
        classification_score=cfg.evaluation.synthcity.classification_score,
        roles=("train",),
    )


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


def test_continuous_target_fails_before_tabpfgen_cache_lookup(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.enabled = False
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = True
    cfg.generation.tabpfgen.variants = ["standard", "custom"]
    cfg.generation.hpo.enabled = False
    dataset = make_canonical_dataset()
    dataset.variable_schema["target"]["kind"] = "continuous"
    standard_generate = mocker.patch(
        "synthdata.generation.tabpfgen_backend.generate_tabpfgen_standard"
    )
    custom_generate = mocker.patch("synthdata.generation.tabpfgen_backend.generate_tabpfgen_custom")

    with pytest.raises(ValueError, match="TabPFGen generation requires a categorical target"):
        run_generation(cfg, dataset)

    standard_generate.assert_not_called()
    custom_generate.assert_not_called()


def test_tabpfgen_parameter_changes_invalidate_ordinary_caches(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.enabled = False
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = True
    cfg.generation.tabpfgen.variants = ["standard", "custom"]
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    cfg.generation.tabpfgen.standard_params = {"temperature": 0.1}
    cfg.generation.tabpfgen.custom_params = {"n_sgld_steps": 10, "noise": 0.2}
    dataset = make_canonical_dataset()
    generated = dataset.role_frame("train", imputed=True).head(cfg.generation.n_samples).copy()
    standard_generate = mocker.patch(
        "synthdata.generation.tabpfgen_backend.generate_tabpfgen_standard",
        return_value=generated,
    )
    custom_generate = mocker.patch(
        "synthdata.generation.tabpfgen_backend.generate_tabpfgen_custom",
        return_value=generated,
    )

    run_generation(cfg, dataset)
    run_generation(cfg, dataset)

    assert standard_generate.call_count == 1
    assert custom_generate.call_count == 1

    cfg.generation.tabpfgen.standard_params["temperature"] = 0.9
    cfg.generation.tabpfgen.custom_params["noise"] = 0.8
    run_generation(cfg, dataset)

    assert standard_generate.call_count == 2
    assert custom_generate.call_count == 2
    standard_metadata = json.loads(
        (Path(cfg.generation.output_dir) / "tabpfgen_standard.cache.json").read_text()
    )
    custom_metadata = json.loads(
        (Path(cfg.generation.output_dir) / "tabpfgen_custom.cache.json").read_text()
    )
    assert standard_metadata["resolved_parameters"] == {"temperature": 0.9}
    assert custom_metadata["resolved_parameters"] == {"n_sgld_steps": 10, "noise": 0.8}


def test_generation_cache_rejects_undersized_frame(make_config, make_canonical_dataset, mocker):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).head(cfg.generation.n_samples).copy()
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )

    run_generation(cfg, dataset)
    metadata_path = Path(cfg.generation.output_dir) / "ctgan.cache.json"
    data_path = Path(cfg.generation.output_dir) / "ctgan.csv"
    cache_metadata = json.loads(metadata_path.read_text())
    data_path.write_text(
        data_path.read_text().splitlines()[0]
        + "\n"
        + "\n".join(data_path.read_text().splitlines()[1:4])
        + "\n"
    )
    cache_metadata["row_count"] = 3
    cache_metadata["synthetic_data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(cache_metadata))

    fit_generate.reset_mock()
    run_generation(cfg, dataset)

    fit_generate.assert_called_once()


def test_non_private_generation_cache_requires_generator_metadata(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).head(cfg.generation.n_samples).copy()
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )

    run_generation(cfg, dataset)
    metadata_path = Path(cfg.generation.output_dir) / "ctgan.cache.json"
    cache_metadata = json.loads(metadata_path.read_text())
    cache_metadata.pop("generator_metadata")
    metadata_path.write_text(json.dumps(cache_metadata))

    fit_generate.reset_mock()
    run_generation(cfg, dataset)

    fit_generate.assert_called_once()


def test_non_private_final_refit_cache_requires_generator_metadata(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).head(cfg.generation.n_samples).copy()
    mocker.patch("synthdata.generation.pipeline.sc.make_loader", return_value=object())
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )
    refit_dir = Path(cfg.evaluation.output_dir) / "final_refit"

    _result, metadata = refit_selected_model(cfg, dataset, "ctgan", output_dir=refit_dir)
    cache_metadata = json.loads(Path(metadata["metadata_path"]).read_text())
    cache_metadata.pop("generator_metadata")
    Path(metadata["metadata_path"]).write_text(json.dumps(cache_metadata))

    fit_generate.reset_mock()
    refit_selected_model(cfg, dataset, "ctgan", output_dir=refit_dir)

    fit_generate.assert_called_once()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("plugin_name", "wrong_generator"),
        ("requested_parameters", {"unexpected": True}),
    ],
)
def test_generation_cache_rejects_mismatched_generator_metadata(
    make_config, make_canonical_dataset, mocker, field, value
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).head(cfg.generation.n_samples).copy()
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )

    run_generation(cfg, dataset)
    metadata_path = Path(cfg.generation.output_dir) / "ctgan.cache.json"
    cache_metadata = json.loads(metadata_path.read_text())
    cache_metadata["generator_metadata"][field] = value
    metadata_path.write_text(json.dumps(cache_metadata))

    fit_generate.reset_mock()
    run_generation(cfg, dataset)

    fit_generate.assert_called_once()


def test_final_refit_cache_rejects_mismatched_generator_metadata(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).head(cfg.generation.n_samples).copy()
    mocker.patch("synthdata.generation.pipeline.sc.make_loader", return_value=object())
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )
    refit_dir = Path(cfg.evaluation.output_dir) / "final_refit"

    _result, metadata = refit_selected_model(cfg, dataset, "ctgan", output_dir=refit_dir)
    metadata_path = Path(metadata["metadata_path"])
    cache_metadata = json.loads(metadata_path.read_text())
    cache_metadata["generator_metadata"]["random_state"] = cfg.seed + 1
    metadata_path.write_text(json.dumps(cache_metadata))

    fit_generate.reset_mock()
    refit_selected_model(cfg, dataset, "ctgan", output_dir=refit_dir)

    fit_generate.assert_called_once()


def test_generation_cache_rejects_changed_implementation_fingerprint(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).head(cfg.generation.n_samples).copy()
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )
    fingerprint = mocker.patch(
        "synthdata.generation.pipeline.sc.generator_implementation_fingerprint",
        side_effect=["implementation-a", "implementation-b"],
    )

    run_generation(cfg, dataset)
    fit_generate.reset_mock()
    run_generation(cfg, dataset)

    assert fingerprint.call_count == 2
    fit_generate.assert_called_once()


def test_final_refit_cache_rejects_changed_implementation_fingerprint(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).head(cfg.generation.n_samples).copy()
    mocker.patch("synthdata.generation.pipeline.sc.make_loader", return_value=object())
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )
    fingerprint = mocker.patch(
        "synthdata.generation.pipeline.sc.generator_implementation_fingerprint",
        side_effect=["implementation-a", "implementation-b"],
    )
    refit_dir = Path(cfg.evaluation.output_dir) / "final_refit"

    refit_selected_model(cfg, dataset, "ctgan", output_dir=refit_dir)
    fit_generate.reset_mock()
    refit_selected_model(cfg, dataset, "ctgan", output_dir=refit_dir)

    assert fingerprint.call_count == 2
    fit_generate.assert_called_once()


def test_final_refit_uses_train_and_tuning_and_resumes_cache(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    captured = {}
    fit_loader = object()
    synthetic = pd.DataFrame(
        {
            "feature": [100.0, 101.0, 102.0, 103.0],
            "protected": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
        }
    )

    def fake_make_loader(frame, *args, **kwargs):
        captured["fit_frame"] = frame.copy()
        return fit_loader

    fake_fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )
    mocker.patch("synthdata.generation.pipeline.sc.make_loader", side_effect=fake_make_loader)

    refit_dir = Path(cfg.evaluation.output_dir) / "final_refit"
    result, metadata = refit_selected_model(cfg, dataset, "ctgan", output_dir=refit_dir)

    expected_fit = pd.concat(
        [dataset.role_frame("train", imputed=True), dataset.role_frame("tuning", imputed=True)],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(captured["fit_frame"], expected_fit)
    pd.testing.assert_frame_equal(result, synthetic)
    assert metadata["fit_roles"] == ["train", "tuning"]
    assert metadata["cache_state"] == "generated"
    assert metadata["fit_frame_fingerprint"] == metadata["fit_frame_fingerprints"]["imputed"]
    assert metadata["generator_metadata"]["plugin_name"] == "ctgan"
    assert metadata["generator_metadata"]["privacy_accounting"] is None
    assert dataset.role_frame("final_holdout", imputed=True).iloc[0]["feature"] not in set(
        captured["fit_frame"]["feature"]
    )
    fake_fit_generate.reset_mock()

    cached_result, cached_metadata = refit_selected_model(
        cfg, dataset, "ctgan", output_dir=refit_dir
    )

    pd.testing.assert_frame_equal(cached_result, synthetic)
    assert cached_metadata["cache_state"] == "hit"
    fake_fit_generate.assert_not_called()

    cached_result.loc[0, "feature"] = -999.0
    cached_result.to_csv(cached_metadata["path"], index=False)

    regenerated_result, regenerated_metadata = refit_selected_model(
        cfg, dataset, "ctgan", output_dir=refit_dir
    )

    pd.testing.assert_frame_equal(regenerated_result, synthetic)
    assert regenerated_metadata["cache_state"] == "generated"
    assert fake_fit_generate.call_count == 1


def test_pategan_generation_cache_persists_and_requires_accounting_metadata(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["pategan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = dataset.full_df.head(cfg.generation.n_samples).copy()
    generator_metadata = _pategan_generator_metadata(cfg.generation.n_samples)
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate",
        return_value=(synthetic, generator_metadata),
    )

    run_generation(cfg, dataset)

    metadata_path = Path(cfg.generation.output_dir) / "pategan.cache.json"
    cache_metadata = json.loads(metadata_path.read_text())
    assert cache_metadata["generator_context"] == generator_metadata["generator_context"]
    assert cache_metadata["generator_metadata"] == generator_metadata

    fit_generate.reset_mock()
    cache_metadata.pop("generator_metadata")
    metadata_path.write_text(json.dumps(cache_metadata))

    run_generation(cfg, dataset)

    fit_generate.assert_called_once()


def test_fit_generate_returns_structured_generator_metadata(mocker):
    synthetic = pd.DataFrame({"feature": [1.0, 2.0], "target": [0, 1]})
    accounting = _pategan_generator_metadata(2)["privacy_accounting"]
    captured = {}

    class GeneratedLoader:
        def dataframe(self):
            return synthetic

    class FakePATEGAN:
        def fit(self, loader):
            captured["fit_loader"] = loader

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return GeneratedLoader()

        def fqdn(self):
            return "privacy.pategan"

        def get_accounting_metadata(self):
            return accounting

    mocker.patch("synthcity.plugins.Plugins.get", return_value=FakePATEGAN())

    result, metadata = sc.fit_generate(
        "pategan",
        {},
        object(),
        n_samples=2,
        random_state=42,
    )

    pd.testing.assert_frame_equal(result, synthetic)
    assert metadata["schema_version"] == sc.GENERATOR_METADATA_SCHEMA_VERSION
    assert metadata["privacy_accounting"] == accounting
    assert captured["generate_kwargs"]["_group_namespace"] == "synthetic"


def test_pategan_generator_metadata_rejects_unfitted_or_mismatched_accounting():
    expected_context = sc.generator_metadata_context("pategan", {})
    metadata = _pategan_generator_metadata(2)

    assert sc.generator_metadata_is_valid(metadata, expected_context)

    unversioned = json.loads(json.dumps(metadata))
    unversioned["privacy_accounting"].pop("schema_version")
    assert not sc.generator_metadata_is_valid(unversioned, expected_context)

    incomplete = json.loads(json.dumps(metadata))
    incomplete["privacy_accounting"]["effective_epsilon"] = None
    assert not sc.generator_metadata_is_valid(incomplete, expected_context)

    mismatched = json.loads(json.dumps(metadata))
    mismatched["privacy_accounting"]["requested_epsilon"] = 2.0
    assert not sc.generator_metadata_is_valid(mismatched, expected_context)


def test_final_refit_cache_ignores_final_holdout_changes(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 2
    dataset = make_canonical_dataset()
    synthetic = pd.DataFrame(
        {
            "feature": [100.0, 101.0],
            "protected": ["A", "B"],
            "target": [0, 1],
        }
    )
    mocker.patch("synthdata.generation.pipeline.sc.make_loader", return_value=object())
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )
    refit_dir = Path(cfg.evaluation.output_dir) / "final_refit"

    _result, first_metadata = refit_selected_model(cfg, dataset, "ctgan", output_dir=refit_dir)
    dataset.imputed_roles["final_holdout"] = dataset.imputed_roles["final_holdout"].copy()
    dataset.imputed_roles["final_holdout"].loc[:, "feature"] += 1000
    dataset.roles["final_holdout"] = dataset.roles["final_holdout"].copy()
    dataset.roles["final_holdout"].loc[:, "feature"] += 2000

    _cached_result, second_metadata = refit_selected_model(
        cfg, dataset, "ctgan", output_dir=refit_dir
    )

    assert first_metadata["cache_key"] == second_metadata["cache_key"]
    assert second_metadata["cache_state"] == "hit"
    fit_generate.assert_called_once()


def test_hpo_generated_cache_rejects_changed_objective_context(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.n_trials = 1
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = pd.DataFrame(
        {
            "feature": [100.0, 101.0, 102.0, 103.0],
            "protected": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
        }
    )
    mocker.patch("synthdata.generation.pipeline.sc.make_loader", return_value=object())
    mocker.patch("synthdata.generation.pipeline.sc.plugin_accepts", return_value=False)
    mocker.patch(
        "synthdata.generation.pipeline.sc.build_synthcity_objective",
        return_value=lambda trial: 0.5,
    )
    run_study = mocker.patch(
        "synthdata.generation.pipeline.hpo_mod.run_study", return_value={"n_iter": 3}
    )
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )

    run_generation(cfg, dataset)
    assert fit_generate.call_count == 2
    assert run_study.call_count == 1

    fit_generate.reset_mock()
    run_study.reset_mock()
    run_generation(cfg, dataset)
    fit_generate.assert_not_called()
    run_study.assert_not_called()

    cfg.generation.hpo.n_iter_cap += 1
    run_generation(cfg, dataset)

    assert [call.args[0] for call in fit_generate.call_args_list] == ["ctgan"]
    run_study.assert_called_once()
    assert run_study.call_args.kwargs["hpo_context"]["objective_context"]["n_iter_cap"] == 301


def test_generation_cache_rejects_changed_semantic_context(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.enabled = False
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).head(4).copy()
    mocker.patch("synthdata.generation.pipeline.sc.make_loader", return_value=object())
    fit_generate = mocker.patch(
        "synthdata.generation.pipeline.sc.fit_generate", return_value=synthetic
    )

    run_generation(cfg, dataset)
    cfg.evaluation.synthcity.classification_score = "macro_f1"
    run_generation(cfg, dataset)

    assert fit_generate.call_count == 2
    cache_metadata = json.loads((Path(cfg.generation.output_dir) / "ctgan.cache.json").read_text())
    assert cache_metadata["semantic_context"]["classification_score"] == "macro_f1"


def test_patient_group_hpo_rejects_row_only_objective_before_generation(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.evaluation.group_mode = "patient_group"
    cfg.evaluation.group_column = "patient_id"
    cfg.generation.synthcity.names = ["ctgan"]
    cfg.generation.tabpfn.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.hpo.metric_config = {"stats": ["wasserstein_dist"]}
    dataset = make_canonical_dataset()

    fit_generate = mocker.patch("synthdata.generation.pipeline.sc.fit_generate")

    with pytest.raises(ValueError, match="canonical HPO allowlist"):
        run_generation(cfg, dataset)

    fit_generate.assert_not_called()
