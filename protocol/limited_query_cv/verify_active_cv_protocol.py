from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "limited_query_cv" / "active_cv_protocol.json"
RUN_ID_PATTERN = re.compile(
    r"^v5_batch(?P<batch>[0-9]+)_run_(?P<run>[0-9]{3,})_goal_"
    r"(?P<goal>[0-9]+)$"
)


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_prompt_capture(
    run: dict,
    registration: dict,
    run_directory: Path,
) -> None:
    ledger_capture = run.get("kickstart_prompt")
    registration_capture = registration.get("kickstart_prompt")
    if ledger_capture is None and registration_capture is None:
        if (
            registration.get("protocol_revision", 0) >= 8
            or "kickstart_prompt_sha256" not in registration
        ):
            raise RuntimeError("missing V5 run prompt-capture metadata")
        return
    if (
        not isinstance(ledger_capture, dict)
        or ledger_capture != registration_capture
        or ledger_capture.get("scope")
        != "literal user kickstart message only"
        or ledger_capture.get("authorization_role") != "audit metadata only"
        or ledger_capture.get("validation_gate") is not False
    ):
        raise RuntimeError("invalid V5 run prompt-capture metadata")

    saved = ledger_capture.get("saved")
    if saved is True:
        prompt_relative = ledger_capture.get("path")
        if (
            not isinstance(prompt_relative, str)
            or ROOT / prompt_relative
            != run_directory / "kickstart_prompt.txt"
            or sha256(prompt_relative) != ledger_capture.get("sha256")
        ):
            raise RuntimeError("V5 saved kickstart prompt changed")
    elif saved is False:
        if (
            ledger_capture.get("path") is not None
            or ledger_capture.get("sha256") is not None
        ):
            raise RuntimeError("invalid empty V5 prompt capture")
    else:
        raise RuntimeError("invalid V5 prompt saved flag")


def main() -> None:
    active = json.loads(ACTIVE.read_text())
    if active["status"] != "active" or active["protocol_name"] != "cv-protocol-v5":
        raise RuntimeError("CV Protocol v5 is not the active pointer")

    policy = load(active["policy"])
    ledger = load(active["ledger"])
    parent_registry = load(policy["frozen_previous_best_registry"])
    if policy["status"] != "active_development":
        raise RuntimeError("active policy is not in development status")
    if ledger["status"] != "active_development":
        raise RuntimeError("active ledger is not in development status")
    if policy["protocol_name"] != active["protocol_name"]:
        raise RuntimeError("active policy name mismatch")
    if ledger["protocol_name"] != active["protocol_name"]:
        raise RuntimeError("active ledger name mismatch")
    if ledger["protocol"] != active["policy"]:
        raise RuntimeError("active ledger policy path mismatch")
    if policy["ledger"] != active["ledger"]:
        raise RuntimeError("active policy ledger path mismatch")
    parent_policy = policy.get("parent_activation", {})
    parent_activation = parent_registry.get("activation_policy", {})
    parent_id = parent_registry.get("active_parent_id")
    parents = parent_registry.get("parents", {})
    active_parents = [
        record
        for record in parents.values()
        if record.get("status") == "active"
    ]
    active_parent_record = parents.get(parent_id, {})
    alignment_record = active_parent_record.get("alignment_record")
    activation_history = parent_registry.get("activation_history", [])
    if (
        parent_registry.get("protocol_name") != active["protocol_name"]
        or parent_id not in parents
        or len(active_parents) != 1
        or active_parents[0].get("solution_id") != parent_id
        or active_parents[0].get("user_approved") is not True
        or not isinstance(
            active_parents[0].get("selection_clean_pooled_oof_rmse"),
            (int, float),
        )
        or not isinstance(
            active_parents[0].get("kaggle_submission_reference"), int
        )
        or not isinstance(active_parents[0].get("public_lb_rmse"), (int, float))
        or parent_policy.get("automatic") is not False
        or parent_policy.get("promotion_changes_parent") is not False
        or parent_policy.get("submission_changes_parent") is not False
        or parent_policy.get(
            "requires_explicit_user_approval_after_submitted_solution_cv_lb_review"
        )
        is not True
        or parent_policy.get("effective_scope") != "future runs only"
        or parent_activation.get("automatic") is not False
        or parent_activation.get("promotion_changes_parent") is not False
        or parent_activation.get("submission_changes_parent") is not False
        or parent_activation.get("effective_scope") != "future runs only"
        or not activation_history
        or activation_history[-1].get("solution_id") != parent_id
        or (
            alignment_record is not None
            and sha256(alignment_record)
            != active_parent_record.get("alignment_record_sha256")
        )
    ):
        raise RuntimeError("invalid V5 frozen previous-best parent registry")
    if not (
        active["protocol_revision"]
        == policy["protocol_revision"]
        == ledger["protocol_revision"]
        == 14
    ):
        raise RuntimeError("active V5 revision mismatch")

    promotion = policy.get("promotion_command_semantics", {})
    kaggle = policy.get("kaggle_boundary", {})
    if (
        promotion.get("terminal_workflow") is not True
        or promotion.get("numeric_score_is_research_context_not_hard_gate")
        is not True
        or promotion.get(
            "authorizes_one_candidate_specific_protected_audit_when_needed"
        )
        is not True
        or promotion.get("creates_reusable_reveal_budget") is not False
        or promotion.get("ledger_must_record_audit_before_reveal") is not True
        or promotion.get(
            "authorizes_uniform_deployment_completeness_repairs"
        )
        is not True
        or promotion.get("authorizes_private_dataset_create_or_update")
        is not True
        or promotion.get(
            "authorizes_private_kernel_create_or_update_and_run"
        )
        is not True
        or promotion.get("must_continue_past_intermediate_ready_states")
        is not True
        or promotion.get("final_verdict_labels")
        != ["LEGIT SUBMISSION: YES", "LEGIT SUBMISSION: NO"]
        or promotion.get("private_packaging_gate")
        != "promotion_rules decision is accept"
        or promotion.get("hold_or_reject_action")
        != (
            "return LEGIT SUBMISSION: NO; do not create or push a Kaggle "
            "Dataset or kernel"
        )
        or promotion.get("does_not_authorize_competition_submission")
        is not True
        or kaggle.get("promote_authorizes_private_dataset_create_or_update")
        is not True
        or kaggle.get(
            "promote_authorizes_private_kernel_create_or_update_and_run"
        )
        is not True
        or kaggle.get("private_packaging_requires_accept_decision")
        is not True
        or kaggle.get("hold_or_reject_forbids_private_dataset_and_kernel_push")
        is not True
        or kaggle.get("private_artifacts_require_separate_followup_after_promote")
        is not False
        or kaggle.get("competition_submission_requires_separate_user_instruction")
        is not True
        or kaggle.get("competition_submission_trigger") != "submit"
    ):
        raise RuntimeError("invalid V5 terminal promotion command semantics")

    comparison = promotion.get("final_report_previous_best_comparison", {})
    if (
        comparison.get("required_for_legit_submission_yes") is not True
        or comparison.get("required_for_every_completed_protected_audit")
        is not True
        or comparison.get("reference_must_not_be_selected_after_candidate_reveal")
        is not True
        or comparison.get("reference_must_not_change_during_run") is not True
        or comparison.get("identical_folds_rows_masks_and_metric_required")
        is not True
        or comparison.get("rows")
        != ["F0", "F1", "F2", "F3", "F4", "pooled"]
        or comparison.get("columns")
        != ["previous_best_rmse", "candidate_rmse", "gain"]
        or comparison.get("gain_definition")
        != "previous_best_rmse - candidate_rmse; positive is improvement"
        or comparison.get(
            "reference_name_and_immutable_artifact_or_commit_required"
        )
        is not True
    ):
        raise RuntimeError("invalid V5 previous-best CV report semantics")

    decision = policy.get("promotion_rules", {})
    if (
        decision.get("numeric_thresholds_role") != "diagnostic_only"
        or decision.get("accept", {}).get(
            "selection_clean_pooled_gain_over_frozen_previous_best_must_be_positive"
        )
        is not True
        or decision.get("accept", {}).get("integrity_rules_must_pass") is not True
        or decision.get("accept", {}).get(
            "no_material_overfitting_or_transfer_reversal"
        )
        is not True
        or decision.get("prospective_audit", {}).get("exact_sign_is_hard_gate")
        is not False
        or decision.get("prospective_audit", {}).get(
            "tiny_regression_within_uncertainty_may_be_accepted"
        )
        is not True
        or decision.get("prospective_audit", {}).get(
            "material_regression_or_transfer_reversal_rejects"
        )
        is not True
        or decision.get("public_leaderboard", {}).get("used_for_in_run_tuning")
        is not False
        or decision.get("public_leaderboard", {}).get(
            "automatic_parent_activation"
        )
        is not False
        or decision.get("public_leaderboard", {}).get(
            "manual_parent_activation_requires_user_cv_lb_approval"
        )
        is not True
        or decision.get("public_leaderboard", {}).get(
            "single_small_regression_may_be_accepted"
        )
        is not True
        or decision.get("public_leaderboard", {}).get(
            "repeated_cv_up_lb_down_triggers_validation_review"
        )
        is not True
        or decision.get("production_promotion_requires_accept_decision")
        is not True
    ):
        raise RuntimeError("invalid V5 evidence-based promotion decision")

    amendments = policy["amendments"]
    if len(amendments) != 8:
        raise RuntimeError("expected exactly eight active V5 amendments")
    by_id = {record["id"]: record for record in amendments}
    expected_ids = {
        "fold-ensemble-production",
        "post-promotion-checkpoint-refinement",
        "manual-analysis-firewall-and-symmetric-pair-rerun",
        "kickstart-configurable-development-goal",
        "batch3-single-reveal-extension",
        "batch3-run2-buda-dip-family-extension",
        "batch5-target-free-boundary-extension",
        "batch6-single-reveal-extension",
    }
    if set(by_id) != expected_ids:
        raise RuntimeError("active V5 amendment IDs changed")
    if set(active["active_amendments"]) != {
        record["policy"] for record in amendments
    }:
        raise RuntimeError("active amendment pointers mismatch")
    supersession = policy.get("revision_14_supersession", {})
    if (
        supersession.get("numeric_thresholds_are_diagnostics") is not True
        or supersession.get("structural_integrity_rules_remain_active") is not True
        or supersession.get("historical_amendment_files_remain_immutable")
        is not True
    ):
        raise RuntimeError("invalid V5 revision-14 supersession boundary")
    for amendment_record in amendments:
        if sha256(amendment_record["policy"]) != amendment_record[
            "policy_sha256"
        ]:
            raise RuntimeError("active amendment policy changed")
        if (
            sha256(amendment_record["documentation"])
            != amendment_record["documentation_sha256"]
        ):
            raise RuntimeError("active amendment documentation changed")

    startup = policy["agent_startup_surface"]
    expected_human_reading = {
        "COOKBOOK.md",
        "limited_query_cv/CV_PROTOCOL_V5.md",
    }
    expected_machine_sources = {
        "limited_query_cv/cv_protocol_v5.json",
        "limited_query_cv/cv_query_ledger_v5.json",
        "limited_query_cv/frozen_previous_best_v5.json",
        "limited_query_cv/verify_active_cv_protocol.py",
    }
    if (
        set(startup["required_human_reading"]) != expected_human_reading
        or set(startup["machine_sources"]) != expected_machine_sources
        or startup["amendment_files_are_required_startup_reading"] is not False
        or startup[
            "historical_policy_and_result_files_are_required_startup_reading"
        ]
        is not False
    ):
        raise RuntimeError("invalid consolidated V5 startup surface")
    for startup_file in expected_human_reading | expected_machine_sources:
        if not (ROOT / startup_file).is_file():
            raise RuntimeError(f"missing V5 startup file: {startup_file}")

    fold_amendment = load(by_id["fold-ensemble-production"]["policy"])
    if (
        fold_amendment["status"] != "active"
        or fold_amendment["effective_from_batch"] != 2
        or fold_amendment["checkpoint_selection"][
            "held_outer_fold_may_select_its_checkpoint"
        ]
        is not False
        or fold_amendment["production"]["full_data_retraining"] is not False
        or fold_amendment["production"]["reuse_frozen_outer_checkpoints"]
        is not True
        or fold_amendment["fold_geometry"][
            "normal_deployed_checkpoints_per_accepted_trainable_family"
        ]
        != 5
    ):
        raise RuntimeError("invalid V5 fold-ensemble amendment")

    refinement = load(
        by_id["post-promotion-checkpoint-refinement"]["policy"]
    )
    if (
        refinement["status"] != "active"
        or refinement["effective_from_batch"] != 2
        or refinement["clean_view"]["promotion_eligible"] is not True
        or refinement["snapshot_registration"][
            "maximum_eligible_snapshots_per_fold"
        ]
        != 3
        or refinement["snapshot_registration"][
            "window_determined_without_held_outer_fold"
        ]
        is not True
        or refinement["freeze_boundary"][
            "every_eligible_held_prediction_frozen_before_reveal"
        ]
        is not True
        or refinement["conditional_production_selection"][
            "requires_clean_gate_pass"
        ]
        is not True
        or refinement["conditional_production_selection"][
            "clean_fold_weight_retained"
        ]
        is not True
        or refinement["production"]["full_data_retraining"] is not False
        or refinement["production"][
            "normal_deployed_checkpoints_per_accepted_trainable_family"
        ]
        != 5
    ):
        raise RuntimeError("invalid V5 post-promotion refinement amendment")

    firewall = load(
        by_id[
            "manual-analysis-firewall-and-symmetric-pair-rerun"
        ]["policy"]
    )
    geometry = firewall["nested_rerun"]
    outer_folds = geometry["outer_folds_m"]
    inner_roles = geometry["inner_roles_per_outer_n"]
    expected_pairs = outer_folds * (outer_folds - 1) // 2
    if (
        firewall["status"] != "active"
        or firewall["effective_from_batch"] != 2
        or firewall["historical_boundary"][
            "batch_1_remains_immutable"
        ]
        is not True
        or firewall["historical_boundary"][
            "audit_fold_was_examined_in_historical_research"
        ]
        is not True
        or firewall["historical_boundary"][
            "firewall_is_prospective_only"
        ]
        is not True
        or firewall["manual_development"]["development_folds"]
        != [0, 1, 2, 3]
        or firewall["manual_development"]["audit_fold"] != 4
        or firewall["development_readiness"][
            "minimum_pooled_rmse_gain"
        ]
        != 0.2
        or firewall["audit_firewall"][
            "pre_reveal_target_derived_access_by_agent"
        ]
        is not False
        or firewall["audit_firewall"][
            "sealed_automated_label_use_after_global_freeze"
        ]["agent_facing_target_derived_output_before_all_outer_caches_frozen"]
        is not False
        or firewall["global_pipeline_freeze"][
            "one_pipeline_for_all_outer_contexts"
        ]
        is not True
        or inner_roles != outer_folds - 1
        or geometry["logical_inner_roles"] != outer_folds * inner_roles
        or geometry["unordered_excluded_fold_pairs"] != expected_pairs
        or geometry["unique_inner_training_trajectories"] != expected_pairs
        or geometry["outer_training_trajectories"] != outer_folds
        or geometry["total_logical_training_roles"]
        != outer_folds * inner_roles + outer_folds
        or geometry["total_unique_model_training_trajectories"]
        != expected_pairs + outer_folds
        or geometry[
            "all_inner_and_outer_outputs_frozen_before_joint_reveal"
        ]
        is not True
        or firewall["joint_reveal_and_promotion"][
            "audit_fold_regression_vetoes_promotion"
        ]
        is not True
        or firewall["production"][
            "normal_deployed_outer_checkpoints_per_accepted_trainable_family"
        ]
        != 5
        or firewall["production"]["inner_training_trajectories_packaged"]
        is not False
    ):
        raise RuntimeError("invalid V5 manual-analysis firewall amendment")

    policy_firewall = policy["manual_analysis_firewall"]
    policy_rerun = policy["symmetric_pair_nested_rerun"]
    if (
        policy_firewall["development_folds"] != [0, 1, 2, 3]
        or policy_firewall["prospective_audit_fold"] != 4
        or policy_firewall[
            "target_derived_audit_fold_access_before_joint_reveal"
        ]
        is not False
        or policy_firewall["development_readiness"][
            "prompt_selected_goal_allowed"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "goal_environment_variable"
        ]
        != "ROGII_V5_DEVELOPMENT_GOAL"
        or policy_firewall["development_readiness"]["goal_resolver"]
        != "limited_query_cv/resolve_v5_goal.py"
        or policy_firewall["development_readiness"][
            "automatic_run_allocation_flag"
        ]
        != "--new-run"
        or policy_firewall["development_readiness"]["prompt_capture_flag"]
        != "--prompt-save"
        or policy_firewall["development_readiness"][
            "prompt_capture_filename"
        ]
        != "kickstart_prompt.txt"
        or policy_firewall["development_readiness"]["prompt_capture_scope"]
        != "literal user kickstart message only"
        or policy_firewall["development_readiness"][
            "prompt_digest_computed_by_resolver"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "prompt_metadata_is_validation_gate"
        ]
        is not False
        or policy_firewall["development_readiness"][
            "prompt_metadata_mismatch_may_consume_run_number"
        ]
        is not False
        or policy_firewall["development_readiness"][
            "pre_revision8_hash_only_records_grandfathered"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "protected_batch_and_development_run_are_independent"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "run_numbering_epoch_required"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "current_run_numbering_epoch"
        ]
        != 2
        or policy_firewall["development_readiness"][
            "run_number_unique_scope"
        ]
        != "within protected batch and numbering epoch"
        or policy_firewall["development_readiness"][
            "run_number_high_watermark_is_durable"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "run_number_reuse_after_cleanup"
        ]
        is not False
        or policy_firewall["development_readiness"][
            "explicit_user_authorized_full_numbering_reset_allowed"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "full_numbering_reset_requires_empty_run_namespace"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "archived_numbering_epoch_1_high_watermark"
        ]
        != 9
        or policy_firewall["development_readiness"][
            "default_requested_goal"
        ]
        != 0.2
        or policy_firewall["development_readiness"]["protocol_floor"] != 0.2
        or policy_firewall["development_readiness"][
            "effective_readiness_target"
        ]
        != "max(requested_development_rmse_gain, protocol_floor)"
        or policy_firewall["development_readiness"][
            "repository_policy_edit_required_for_clear_prompt_value"
        ]
        is not False
        or policy_firewall["development_readiness"][
            "frozen_goal_record_required_before_target_derived_metrics"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "effective_readiness_target_role"
        ]
        != "research aspiration and prioritization; not a reveal or promotion cutoff"
        or policy_firewall["development_readiness"][
            "selection_clean_pooled_development_gain_must_be_positive_to_advance"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "advance_below_effective_readiness_target_allowed"
        ]
        is not True
        or policy_firewall["development_readiness"][
            "diagnostic_reference_values_are_hard_gates"
        ]
        is not False
        or policy_firewall["goal_policy"]
        != by_id["kickstart-configurable-development-goal"]["policy"]
        or policy_rerun["logical_inner_roles"] != 20
        or policy_rerun["unique_inner_training_trajectories"] != 10
        or policy_rerun["outer_training_trajectories"] != 5
        or policy_rerun["total_unique_model_training_trajectories"] != 15
        or policy["promotion_rules"]["prospective_audit"][
            "exact_sign_is_hard_gate"
        ]
        is not False
    ):
        raise RuntimeError(
            "active policy does not enforce V5 geometry and goal resolution"
        )

    kickstart_goal = load(
        by_id["kickstart-configurable-development-goal"]["policy"]
    )
    if (
        kickstart_goal["status"] != "active"
        or kickstart_goal["effective_from_batch"] != 2
        or kickstart_goal["goal_parameter"][
            "explicit_natural_language_value_allowed"
        ]
        is not True
        or kickstart_goal["goal_parameter"]["environment_variable"]
        != "ROGII_V5_DEVELOPMENT_GOAL"
        or kickstart_goal["goal_parameter"]["resolver"]
        != "limited_query_cv/resolve_v5_goal.py"
        or kickstart_goal["goal_parameter"]["default_when_omitted"] != 0.2
        or kickstart_goal["goal_parameter"]["must_be_finite"] is not True
        or kickstart_goal["goal_parameter"]["must_be_positive"] is not True
        or kickstart_goal["readiness_resolution"]["protocol_floor"] != 0.2
        or kickstart_goal["readiness_resolution"][
            "prompt_value_above_floor_raises_gate"
        ]
        is not True
        or kickstart_goal["readiness_resolution"][
            "prompt_value_below_floor_lowers_gate"
        ]
        is not False
        or kickstart_goal["runtime_interface"][
            "agent_transfers_clear_prompt_value_to_environment"
        ]
        is not True
        or kickstart_goal["runtime_interface"][
            "environment_variable_is_durable_record"
        ]
        is not False
        or kickstart_goal["runtime_interface"][
            "frozen_resolver_output_is_durable_record"
        ]
        is not True
        or kickstart_goal["prompt_capture"]["resolver_flag"] != "--prompt-save"
        or kickstart_goal["prompt_capture"]["saved_filename"]
        != "kickstart_prompt.txt"
        or kickstart_goal["prompt_capture"]["input_scope"]
        != "literal user kickstart message only"
        or kickstart_goal["prompt_capture"][
            "hidden_system_or_developer_context_captured"
        ]
        is not False
        or kickstart_goal["prompt_capture"]["digest_computed_by_resolver"]
        is not True
        or kickstart_goal["prompt_capture"]["user_supplied_digest_allowed"]
        is not False
        or kickstart_goal["prompt_capture"]["authorization_role"]
        != "audit metadata only"
        or kickstart_goal["prompt_capture"]["validation_gate"] is not False
        or kickstart_goal["prompt_capture"][
            "mismatch_may_abandon_or_renumber_run"
        ]
        is not False
        or kickstart_goal["prompt_capture"][
            "existing_hash_only_records_grandfathered"
        ]
        is not True
        or kickstart_goal["run_allocation"]["resolver_flag"] != "--new-run"
        or kickstart_goal["run_allocation"][
            "protected_batch_and_development_run_are_independent"
        ]
        is not True
        or kickstart_goal["run_allocation"]["run_number_scope"]
        != "within protected batch and numbering epoch"
        or kickstart_goal["run_allocation"][
            "run_numbering_epoch_recorded"
        ]
        is not True
        or kickstart_goal["run_allocation"]["current_numbering_epoch"] != 2
        or kickstart_goal["run_allocation"]["run_number_monotonic"] is not True
        or kickstart_goal["run_allocation"][
            "run_number_reuse_after_cleanup"
        ]
        is not False
        or kickstart_goal["run_allocation"][
            "run_number_reuse_across_explicit_reset_epochs"
        ]
        is not True
        or kickstart_goal["run_allocation"][
            "explicit_user_authorized_full_numbering_reset_allowed"
        ]
        is not True
        or kickstart_goal["run_allocation"][
            "full_numbering_reset_requires_empty_run_namespace"
        ]
        is not True
        or kickstart_goal["run_allocation"][
            "durable_high_watermark_in_ledger"
        ]
        is not True
        or kickstart_goal["run_allocation"][
            "archived_numbering_epoch_1_high_watermark"
        ]
        != 9
        or kickstart_goal["run_allocation"]["numbering_epoch_2_first_run"] != 1
        or kickstart_goal["run_registration"][
            "repository_policy_edit_required_for_clear_prompt_value"
        ]
        is not False
        or kickstart_goal["run_registration"][
            "required_before_target_derived_development_metrics"
        ]
        is not True
        or kickstart_goal["run_registration"][
            "prompt_capture_recommended_but_not_authoritative"
        ]
        is not True
        or kickstart_goal["run_registration"][
            "silent_post_metric_goal_reduction_allowed"
        ]
        is not False
        or kickstart_goal["unchanged_boundaries"][
            "f4_manual_analysis_firewall"
        ]
        is not True
        or kickstart_goal["unchanged_boundaries"]["logical_inner_roles"] != 20
        or kickstart_goal["unchanged_boundaries"][
            "unique_inner_training_trajectories"
        ]
        != 10
        or kickstart_goal["unchanged_boundaries"][
            "normal_deployed_outer_checkpoints"
        ]
        != 5
    ):
        raise RuntimeError("invalid V5 kickstart-goal amendment")
    if not (ROOT / kickstart_goal["goal_parameter"]["resolver"]).is_file():
        raise RuntimeError("missing V5 kickstart-goal resolver")

    batch3 = load(by_id["batch3-single-reveal-extension"]["policy"])
    batch3_authorization = load(batch3["authorization"])
    if (
        batch3["status"] != "active"
        or batch3["effective_from_batch"] != 3
        or batch3["budget"]["previous_maximum_batches"] != 2
        or batch3["budget"]["new_maximum_batches"] != 3
        or batch3["budget"][
            "maximum_joint_outer_oof_reveals_for_batch3"
        ]
        != 1
        or batch3["budget"]["maximum_total_joint_outer_oof_reveals"] != 3
        or batch3["budget"]["fold_pilot_reveals_allowed"] is not False
        or batch3["budget"]["post_reveal_recipe_changes_allowed"] is not False
        or batch3["candidate_scope"]["family"]
        != "c016_residual_gr_path_pool"
        or batch3["candidate_scope"]["requested_development_rmse_gain"]
        != 0.2
        or batch3["candidate_scope"]["effective_readiness_target"] != 0.2
        or batch3["unchanged_boundaries"]["development_folds"]
        != [0, 1, 2, 3]
        or batch3["unchanged_boundaries"]["audit_fold"] != 4
        or batch3["unchanged_boundaries"][
            "f4_manual_analysis_before_reveal"
        ]
        is not False
        or batch3["unchanged_boundaries"][
            "symmetric_pair_nested_rerun_required"
        ]
        is not True
        or batch3["unchanged_boundaries"][
            "all_five_outer_predictions_frozen_before_reveal"
        ]
        is not True
        or batch3["unchanged_boundaries"]["kaggle_submission_authorized"]
        is not False
        or batch3_authorization["status"] != "authorized"
        or batch3_authorization["scope"]["batch"] != 3
        or batch3_authorization["scope"]["candidate_family"]
        != "c016_residual_gr_path_pool"
        or batch3_authorization["scope"][
            "additional_protected_joint_outer_reveals"
        ]
        != 1
        or batch3_authorization["scope"][
            "kaggle_competition_submission_authorized"
        ]
        is not False
        or sha256(batch3_authorization["source_review"])
        != batch3_authorization["source_review_sha256"]
        or sha256(batch3_authorization["experiment_proposal"])
        != batch3_authorization["experiment_proposal_sha256"]
    ):
        raise RuntimeError("invalid V5 Batch 3 authorization amendment")

    batch3_run2 = load(
        by_id["batch3-run2-buda-dip-family-extension"]["policy"]
    )
    batch3_run2_authorization = load(batch3_run2["authorization"])
    if (
        batch3_run2["status"] != "active"
        or batch3_run2["effective_from_batch"] != 3
        or batch3_run2["budget"]["batch3_reveal_previously_authorized"] != 1
        or batch3_run2["budget"]["batch3_reveals_previously_spent"] != 0
        or batch3_run2["budget"]["batch3_reveals_added"] != 0
        or batch3_run2["budget"][
            "maximum_joint_outer_oof_reveals_for_batch3"
        ]
        != 1
        or batch3_run2["budget"]["reuse_unspent_reveal"] is not True
        or batch3_run2["candidate_scope"]["run_id"]
        != "v5_batch3_run_002_goal_020"
        or batch3_run2["candidate_scope"]["family"]
        != "c016_buda_delayed_dip_cluster_surface"
        or batch3_run2["candidate_scope"]["readiness_baseline_rmse"]
        != 5.869105573928044
        or batch3_run2["candidate_scope"][
            "requested_incremental_rmse_gain"
        ]
        != 0.2
        or batch3_run2["candidate_scope"]["maximum_candidate_rmse"]
        != 5.669105573928044
        or batch3_run2["candidate_scope"][
            "equivalent_minimum_gain_over_protected_v20"
        ]
        != 0.2710854363677842
        or batch3_run2["feature_firewall"]["buda_is_train_only"] is not True
        or batch3_run2["feature_firewall"][
            "buda_allowed_as_role_purged_training_surface"
        ]
        is not True
        or batch3_run2["feature_firewall"][
            "held_well_buda_may_enter_held_prediction"
        ]
        is not False
        or batch3_run2["feature_firewall"][
            "held_well_ancc_may_enter_held_prediction"
        ]
        is not False
        or batch3_run2["feature_firewall"][
            "held_hidden_tvt_may_enter_held_prediction"
        ]
        is not False
        or batch3_run2["unchanged_boundaries"]["development_folds"]
        != [0, 1, 2, 3]
        or batch3_run2["unchanged_boundaries"]["audit_fold"] != 4
        or batch3_run2["unchanged_boundaries"][
            "f4_manual_analysis_before_reveal"
        ]
        is not False
        or batch3_run2["unchanged_boundaries"][
            "all_five_outer_predictions_frozen_before_reveal"
        ]
        is not True
        or batch3_run2["unchanged_boundaries"][
            "kaggle_submission_authorized"
        ]
        is not False
        or batch3_run2_authorization["status"] != "authorized"
        or batch3_run2_authorization["scope"]["batch"] != 3
        or batch3_run2_authorization["scope"]["run_number"] != 2
        or batch3_run2_authorization["scope"]["candidate_family"]
        != "c016_buda_delayed_dip_cluster_surface"
        or batch3_run2_authorization["scope"][
            "reuse_batch3_unspent_protected_joint_outer_reveal"
        ]
        is not True
        or batch3_run2_authorization["scope"][
            "additional_protected_joint_outer_reveals"
        ]
        != 0
        or batch3_run2_authorization["scope"][
            "kaggle_competition_submission_authorized"
        ]
        is not False
    ):
        raise RuntimeError("invalid V5 Batch 3 Run 2 family extension")

    batch5 = load(
        by_id["batch5-target-free-boundary-extension"]["policy"]
    )
    batch5_authorization = load(batch5["authorization"])
    boundary_record = batch5["validation_boundary"]
    boundary = load(boundary_record["boundary"])
    if (
        batch5["status"] != "active"
        or batch5["effective_from_batch"] != 5
        or batch5["budget"]["previous_maximum_batches"] != 4
        or batch5["budget"]["new_maximum_batches"] != 5
        or batch5["budget"][
            "additional_protected_joint_outer_reveals"
        ]
        != 1
        or batch5["budget"]["maximum_total_joint_outer_oof_reveals"]
        != 5
        or batch5["budget"]["fold_pilot_reveals_allowed"] is not False
        or batch5["budget"]["post_reveal_recipe_changes_allowed"]
        is not False
        or boundary_record["target_fields_read"] is not False
        or boundary_record["complete_well_split"] is not True
        or boundary_record[
            "same_typewell_cluster_kept_inside_one_fold"
        ]
        is not True
        or boundary_record["development_folds"] != [0, 1, 2, 3]
        or boundary_record["prospective_audit_fold"] != 4
        or boundary_record[
            "previous_batch4_F4_evidence_reused_as_fresh_gate"
        ]
        is not False
        or sha256(boundary_record["boundary"])
        != boundary_record["boundary_sha256"]
        or sha256(boundary_record["builder"])
        != boundary_record["builder_sha256"]
        or boundary["status"]
        != "frozen_before_any_batch5_target_metric"
        or boundary["target_fields_read"] is not False
        or boundary["horizontal_TVT_read"] is not False
        or boundary["train_only_horizontal_columns_read"] is not False
        or boundary["development_folds"] != [0, 1, 2, 3]
        or boundary["prospective_audit_fold"] != 4
        or boundary["fold_count"] != 5
        or len(boundary["assignments"]) != 773
        or set(boundary["assignments"].values()) != {0, 1, 2, 3, 4}
        or batch5_authorization["status"] != "authorized"
        or batch5_authorization["scope"]["batch"] != 5
        or batch5_authorization["scope"][
            "additional_protected_joint_outer_reveals"
        ]
        != 1
        or batch5_authorization["scope"][
            "kaggle_competition_submission_authorized"
        ]
        is not False
    ):
        raise RuntimeError("invalid V5 Batch 5 boundary extension")
    for members in boundary["cluster_members"].values():
        if len({boundary["assignments"][well_id] for well_id in members}) != 1:
            raise RuntimeError("Batch 5 typewell cluster split across folds")

    batch6 = load(by_id["batch6-single-reveal-extension"]["policy"])
    batch6_authorization = load(batch6["authorization"])
    batch6_boundary_record = batch6["validation_boundary"]
    batch6_boundary = load(batch6_boundary_record["boundary"])
    if (
        batch6["status"] != "active"
        or batch6["effective_from_batch"] != 6
        or batch6["budget"]["previous_maximum_batches"] != 5
        or batch6["budget"]["new_maximum_batches"] != 6
        or batch6["budget"]["additional_protected_joint_outer_reveals"] != 1
        or batch6["budget"]["maximum_total_joint_outer_oof_reveals"] != 6
        or batch6["budget"]["fold_pilot_reveals_allowed"] is not False
        or batch6["budget"]["post_reveal_recipe_changes_allowed"] is not False
        or batch6_boundary_record["target_fields_read"] is not False
        or batch6_boundary_record["complete_well_split"] is not True
        or batch6_boundary_record["development_folds"] != [0, 1, 2, 3]
        or batch6_boundary_record["prospective_audit_fold"] != 4
        or batch6_boundary_record["audit_well_count"] != 155
        or sha256(batch6_boundary_record["boundary"])
        != batch6_boundary_record["boundary_sha256"]
        or sha256(batch6_boundary_record["source"])
        != batch6_boundary_record["source_sha256"]
        or batch6_boundary["fold_count"] != 5
        or len(batch6_boundary["assignments"]) != 773
        or set(batch6_boundary["assignments"].values()) != {0, 1, 2, 3, 4}
        or batch6_boundary["target_fields_read"] is not False
        or batch6_boundary["horizontal_TVT_read"] is not False
        or batch6_boundary["train_only_horizontal_columns_read"] is not False
        or batch6_authorization["status"] != "authorized"
        or batch6_authorization["scope"]["batch"] != 6
        or batch6_authorization["scope"]["additional_protected_joint_outer_reveals"] != 1
        or batch6_authorization["scope"]["kaggle_competition_submission_authorized"] is not False
    ):
        raise RuntimeError("invalid V5 Batch 6 extension")

    ledger_amendments = {
        record["id"]: record for record in ledger["active_amendments"]
    }
    if set(ledger_amendments) != expected_ids:
        raise RuntimeError("ledger amendment IDs changed")
    for amendment_id, record in by_id.items():
        ledger_record = ledger_amendments[amendment_id]
        if (
            ledger_record["policy"] != record["policy"]
            or ledger_record["policy_sha256"] != record["policy_sha256"]
        ):
            raise RuntimeError("ledger amendment pointer mismatch")

    numbering = ledger["development_run_numbering"]
    current_numbering_epoch = int(numbering["current_epoch"])
    archived_epochs = ledger["archived_development_run_numbering_epochs"]
    if (
        current_numbering_epoch != 2
        or numbering["registration_field"] != "run_numbering_epoch"
        or numbering["run_number_scope"]
        != "within protected batch and numbering epoch"
        or numbering["run_ids_may_repeat_across_epochs"] is not True
        or len(archived_epochs) != 1
        or archived_epochs[0]["numbering_epoch"] != 1
        or archived_epochs[0]["high_watermark"] != 9
    ):
        raise RuntimeError("invalid V5 development run-numbering reset")

    development_runs = ledger["development_runs"]
    run_keys: set[tuple[int, int, int]] = set()
    maximum_run_by_batch: dict[int, int] = {}
    for run in development_runs:
        batch = int(run["batch"])
        run_epoch = int(run["run_numbering_epoch"])
        run_number = int(run["run_number"])
        key = (run_epoch, batch, run_number)
        if key in run_keys:
            raise RuntimeError("duplicate V5 development run number")
        run_keys.add(key)
        if run_epoch != current_numbering_epoch:
            raise RuntimeError("active V5 run belongs to an archived epoch")
        maximum_run_by_batch[batch] = max(
            maximum_run_by_batch.get(batch, 0),
            run_number,
        )

        match = RUN_ID_PATTERN.fullmatch(run["run_id"])
        if (
            match is None
            or int(match.group("batch")) != batch
            or int(match.group("run")) != run_number
            or match.group("goal") != run["goal_code"]
            or run["effective_readiness_target"]
            < max(run["requested_development_rmse_gain"], 0.2)
        ):
            raise RuntimeError("invalid V5 development run identity")

        registration_path = run["goal_registration"]
        if sha256(registration_path) != run["goal_registration_sha256"]:
            raise RuntimeError("V5 development goal registration changed")
        if run["legacy_layout"]:
            if (
                run["run_id"] != "v5_batch2_run_001_goal_050"
                or run["status"]
                != "stopped_before_preregistration_readiness_failed"
            ):
                raise RuntimeError("invalid legacy V5 development run")
        else:
            run_directory = ROOT / run["run_directory"]
            if (
                run_directory.name != run["run_id"]
                or (ROOT / registration_path).parent != run_directory
            ):
                raise RuntimeError("V5 run directory does not match run ID")
            registration = load(registration_path)
            if (
                registration["batch"] != batch
                or registration["run_numbering_epoch"] != run_epoch
                or registration["run_number"] != run_number
                or registration["run_id"] != run["run_id"]
                or registration["run_directory"] != run["run_directory"]
                or registration["requested_development_rmse_gain"]
                != run["requested_development_rmse_gain"]
                or registration["effective_readiness_target"]
                != run["effective_readiness_target"]
            ):
                raise RuntimeError("V5 run ledger and goal record mismatch")
            verify_prompt_capture(run, registration, run_directory)

        if "development_result" in run:
            if sha256(run["development_result"]) != run[
                "development_result_sha256"
            ]:
                raise RuntimeError("V5 development result changed")

    high_watermarks = {
        int(batch): int(run_number)
        for batch, run_number in ledger[
            "development_run_high_watermarks"
        ].items()
    }
    for batch, maximum_run in maximum_run_by_batch.items():
        if high_watermarks.get(batch, 0) < maximum_run:
            raise RuntimeError("V5 development run high-watermark regressed")

    batch_policy = policy["batch_policy"]
    if ledger["maximum_batches"] != batch_policy["maximum_batches"]:
        raise RuntimeError("batch budget mismatch")
    if (
        ledger["maximum_candidate_families_per_batch"]
        != batch_policy["maximum_candidate_families_per_batch"]
    ):
        raise RuntimeError("candidate-family budget mismatch")
    if (
        ledger["maximum_joint_outer_oof_reveals_per_batch"]
        != batch_policy["maximum_joint_outer_oof_reveals_per_batch"]
    ):
        raise RuntimeError("outer-OOF budget mismatch")
    if ledger["completed_batches"] + ledger["remaining_batches"] != ledger[
        "maximum_batches"
    ]:
        raise RuntimeError("ledger batch accounting mismatch")
    available = ledger["maximum_batches"] - ledger[
        "total_protected_outer_oof_reveals"
    ]
    if available != active["protected_outer_oof_reveals_available"]:
        raise RuntimeError("active outer-OOF availability mismatch")
    if (
        ledger["total_protected_outer_oof_reveals"]
        != active["protected_outer_oof_reveals_used"]
    ):
        raise RuntimeError("active outer-OOF usage mismatch")
    if len(ledger["research_queue"]) != active[
        "candidate_families_registered_total"
    ]:
        raise RuntimeError("registered candidate count mismatch")
    inactive_statuses = {
        "rejected_by_outer_gate",
        "promoted",
        "withdrawn",
    }
    active_candidates = sum(
        candidate["status"] not in inactive_statuses
        for candidate in ledger["research_queue"]
    )
    if active_candidates != active["active_candidate_families"]:
        raise RuntimeError("active candidate count mismatch")
    c016_candidates = [
        candidate
        for candidate in ledger["research_queue"]
        if candidate["candidate"]
        == "clean_c016_without_quarantined_d072"
    ]
    if len(c016_candidates) == 1:
        c016 = c016_candidates[0]
        if (
            c016["batch"] != 2
            or c016["status"]
            not in {
                "exception_audit_authorized_preregistration_pending",
                "exception_audit_paused_scope_decision_required",
                "exception_audit_leak_repaired_scope_decision_required",
                "exception_audit_preregistered_confirmations_pending",
                "exception_audit_confirmations_passed_nested_training_authorized",
                "exception_audit_preregistered",
                "exception_nested_rerun_complete_reveal_pending",
                "rejected_by_outer_gate",
                "promoted",
                "withdrawn",
            }
            or sha256(c016["authorization"])
            != c016["authorization_sha256"]
            or (
                "lineage_correction" in c016
                and sha256(c016["lineage_correction"])
                != c016["lineage_correction_sha256"]
            )
            or (
                "leak_repair" in c016
                and sha256(c016["leak_repair"])
                != c016["leak_repair_sha256"]
            )
            or (
                "clean_scope_authorization" in c016
                and sha256(c016["clean_scope_authorization"])
                != c016["clean_scope_authorization_sha256"]
            )
            or (
                "promotion_preregistration" in c016
                and sha256(c016["promotion_preregistration"])
                != c016["promotion_preregistration_sha256"]
            )
            or (
                "implementation_gate" in c016
                and sha256(c016["implementation_gate"])
                != c016["implementation_gate_sha256"]
            )
        ):
            raise RuntimeError("invalid C016 exception authorization")
        if "promotion_preregistration" in c016:
            clean_authorization = load(c016["clean_scope_authorization"])
            preregistration = load(c016["promotion_preregistration"])
            implementation_gate = load(c016["implementation_gate"])
            if (
                clean_authorization["status"]
                != "authorized_for_preregistered_promotion_audit"
                or abs(
                    clean_authorization["candidate"][
                        "development_gain_versus_v20"
                    ]
                    - 0.1210103266375624
                )
                > 1e-12
                or preregistration["status"]
                != "preregistered_clean_c016_exception_audit"
                or preregistration["baseline_commit"] != "f32712b"
                or preregistration["exception_authorization_sha256"]
                != c016["clean_scope_authorization_sha256"]
                or preregistration["implementation_gate"][
                    "result_sha256"
                ]
                != c016["implementation_gate_sha256"]
                or preregistration["reveal"][
                    "protected_outer_reveal_spent"
                ]
                is not False
                or preregistration["reveal"]["f4_metric_computed"] is not False
                or preregistration["kaggle_dataset_authorized"] is not False
                or preregistration["kaggle_kernel_authorized"] is not False
                or preregistration["kaggle_submission_authorized"] is not False
                or implementation_gate["status"] != "passed"
                or implementation_gate["f4_loaded"] is not False
                or implementation_gate["f4_metric_computed"] is not False
                or implementation_gate["checks"][
                    "f4_real_target_not_loaded"
                ]
                is not True
            ):
                raise RuntimeError("invalid clean C016 preregistration")
            for source, expected in preregistration[
                "frozen_source_sha256"
            ].items():
                if sha256(source) != expected:
                    raise RuntimeError(
                        f"clean C016 preregistered source changed: {source}"
                    )
            if "confirmation_result" in c016:
                confirmation = load(c016["confirmation_result"])
                if (
                    sha256(c016["confirmation_result"])
                    != c016["confirmation_result_sha256"]
                    or confirmation["status"] != "confirmations_passed"
                    or confirmation["all_confirmations_passed"] is not True
                    or confirmation["audit_fold_loaded"] is not False
                    or confirmation["f4_loaded"] is not False
                    or confirmation["f4_metric_computed"] is not False
                    or confirmation["source_sha256"]
                    != preregistration["frozen_source_sha256"][
                        "limited_query_cv/assemble_v5_run25_c016_nested.py"
                    ]
                    or any(
                        confirmation["results"][group][
                            "gain_vs_v20"
                        ]
                        < 0.10
                        for group in ("balanced", "independent")
                    )
                ):
                    raise RuntimeError(
                        "invalid clean C016 confirmation result"
                    )
            if "outer_prediction_manifest" in c016:
                for key in (
                    "candidate_manifest",
                    "selector_record",
                    "outer_prediction",
                    "outer_prediction_manifest",
                ):
                    if sha256(c016[key]) != c016[f"{key}_sha256"]:
                        raise RuntimeError(
                            f"clean C016 frozen {key} changed"
                        )
                candidate_manifest = load(c016["candidate_manifest"])
                prediction_manifest = load(
                    c016["outer_prediction_manifest"]
                )
                if (
                    candidate_manifest["status"]
                    != "frozen_before_selector_or_protected_metric"
                    or candidate_manifest["artifact_count"] != 915
                    or candidate_manifest["trainable_trajectories"][
                        "total"
                    ]
                    != 75
                    or candidate_manifest["f4_metric_computed"] is not False
                    or prediction_manifest["status"]
                    != "frozen_ready_for_single_joint_reveal"
                    or prediction_manifest["outer_folds"] != 5
                    or prediction_manifest["truth_stored_in_prediction"]
                    is not False
                    or prediction_manifest["protected_metric_computed"]
                    is not False
                    or prediction_manifest["f4_metric_computed"] is not False
                    or prediction_manifest["candidate_manifest_sha256"]
                    != c016["candidate_manifest_sha256"]
                    or prediction_manifest["selector_sha256"]
                    != c016["selector_record_sha256"]
                    or prediction_manifest["prediction_sha256"]
                    != c016["outer_prediction_sha256"]
                ):
                    raise RuntimeError(
                        "invalid clean C016 frozen prediction manifest"
                    )
            if "promotion_result" in c016:
                promotion_result = load(c016["promotion_result"])
                if (
                    sha256(c016["promotion_result"])
                    != c016["promotion_result_sha256"]
                    or sha256(c016["promotion_oof"])
                    != c016["promotion_oof_sha256"]
                    or promotion_result["status"]
                    != "complete_with_promotion"
                    or promotion_result["accepted"] is not True
                    or promotion_result["strong_gate"]["passed"] is not True
                    or promotion_result["f4_nonnegative_gain_veto_passed"]
                    is not True
                    or promotion_result["protected_outer_reveal_spent"]
                    is not True
                    or promotion_result["protected_outer_reveal_count"] != 1
                    or promotion_result["kaggle_dataset_authorized"] is not False
                    or promotion_result["kaggle_kernel_authorized"] is not False
                    or promotion_result["kaggle_submission_authorized"] is not False
                    or abs(
                        promotion_result["oof_gain"]
                        - 0.06486559414696291
                    )
                    > 1e-12
                    or abs(
                        promotion_result["f4_audit"]["gain"]
                        - 0.044604920598778186
                    )
                    > 1e-12
                ):
                    raise RuntimeError(
                        "invalid clean C016 promotion result"
                    )

    batch3_candidates = [
        candidate
        for candidate in ledger["research_queue"]
        if candidate["candidate"] == "c016_residual_gr_path_pool"
    ]
    if len(batch3_candidates) != 1:
        raise RuntimeError("missing V5 Batch 3 candidate")
    batch3_candidate = batch3_candidates[0]
    if (
        batch3_candidate["batch"] != 3
        or batch3_candidate["status"]
        not in {
            "authorized_development",
            "registered_development",
            "metric_free_gate_passed",
            "development_in_progress",
            "development_readiness_passed",
            "nested_rerun_in_progress",
            "nested_rerun_complete_reveal_pending",
            "rejected_before_protected_reveal",
            "rejected_by_outer_gate",
            "promoted",
            "withdrawn",
        }
        or batch3_candidate["requested_development_rmse_gain"] != 0.2
        or batch3_candidate["effective_readiness_target"] != 0.2
        or batch3_candidate["maximum_protected_outer_reveals"] != 1
        or batch3_candidate["kaggle_competition_submission_authorized"]
        is not False
        or sha256(batch3_candidate["source_review"])
        != batch3_candidate["source_review_sha256"]
        or sha256(batch3_candidate["experiment_proposal"])
        != batch3_candidate["experiment_proposal_sha256"]
        or sha256(batch3_candidate["authorization"])
        != batch3_candidate["authorization_sha256"]
        or sha256(batch3_candidate["family_registration"])
        != batch3_candidate["family_registration_sha256"]
    ):
        raise RuntimeError("invalid V5 Batch 3 candidate authorization")
    batch3_registration = load(batch3_candidate["family_registration"])
    if (
        batch3_registration["status"]
        not in {
            "frozen_before_implementation_or_target_metric",
            "metric_free_gate_passed_before_target_metric",
        }
        or batch3_registration["run_id"]
        != "v5_batch3_run_001_goal_020"
        or batch3_registration["family"]
        != "c016_residual_gr_path_pool"
        or batch3_registration["goal"]["effective_readiness_target"] != 0.2
        or batch3_registration["parent"]["prediction_sha256"]
        != "737711c483cc3ce81ba94fa8e5dfb9c307f75c7691fd06be33631d80c355fad1"
        or len(batch3_registration["registered_variants"]) < 3
        or batch3_registration["protected_and_kaggle_boundaries"][
            "kaggle_submission_authorized"
        ]
        is not False
    ):
        raise RuntimeError("invalid V5 Batch 3 family registration")
    if batch3_candidate["status"] == "metric_free_gate_passed":
        batch3_gate = load(batch3_candidate["implementation_gate"])
        if (
            sha256(batch3_candidate["implementation_source"])
            != batch3_candidate["implementation_source_sha256"]
            or sha256(batch3_candidate["implementation_gate"])
            != batch3_candidate["implementation_gate_sha256"]
            or batch3_registration["status"]
            != "metric_free_gate_passed_before_target_metric"
            or batch3_gate["status"] != "passed"
            or batch3_gate["hidden_truth_loaded"] is not False
            or batch3_gate["F4_loaded"] is not False
            or batch3_gate["protected_reveal_spent"] is not False
        ):
            raise RuntimeError("invalid V5 Batch 3 implementation gate")

    batch3_run2_candidates = [
        candidate
        for candidate in ledger["research_queue"]
        if candidate["candidate"]
        == "c016_buda_delayed_dip_cluster_surface"
    ]
    if len(batch3_run2_candidates) != 1:
        raise RuntimeError("missing V5 Batch 3 Run 2 candidate")
    batch3_run2_candidate = batch3_run2_candidates[0]
    if (
        batch3_run2_candidate["batch"] != 3
        or batch3_run2_candidate["status"]
        not in {
            "registered_development",
            "metric_free_gate_passed",
            "development_in_progress",
            "development_readiness_passed",
            "development_repair_required",
            "nested_rerun_in_progress",
            "nested_rerun_complete_reveal_pending",
            "rejected_before_protected_reveal",
            "rejected_by_outer_gate",
            "promoted",
            "withdrawn",
        }
        or batch3_run2_candidate["requested_development_rmse_gain"] != 0.2
        or batch3_run2_candidate["effective_readiness_target"] != 0.2
        or batch3_run2_candidate["readiness_baseline_rmse"]
        != 5.869105573928044
        or batch3_run2_candidate["maximum_candidate_rmse"]
        != 5.669105573928044
        or batch3_run2_candidate["maximum_protected_outer_reveals"] != 1
        or batch3_run2_candidate["protected_outer_reveal_spent"] is not False
        or batch3_run2_candidate["F4_loaded"] is not False
        or batch3_run2_candidate[
            "kaggle_competition_submission_authorized"
        ]
        is not False
        or sha256(batch3_run2_candidate["source_review"])
        != batch3_run2_candidate["source_review_sha256"]
        or sha256(batch3_run2_candidate["experiment_proposal"])
        != batch3_run2_candidate["experiment_proposal_sha256"]
        or sha256(batch3_run2_candidate["authorization"])
        != batch3_run2_candidate["authorization_sha256"]
        or sha256(batch3_run2_candidate["family_registration"])
        != batch3_run2_candidate["family_registration_sha256"]
    ):
        raise RuntimeError("invalid V5 Batch 3 Run 2 candidate")
    batch3_run2_registration = load(
        batch3_run2_candidate["family_registration"]
    )
    if (
        batch3_run2_registration["status"]
        not in {
            "frozen_before_implementation_or_target_metric",
            "metric_free_gate_passed_before_target_metric",
        }
        or batch3_run2_registration["run_id"]
        != "v5_batch3_run_002_goal_020"
        or batch3_run2_registration["family"]
        != "c016_buda_delayed_dip_cluster_surface"
        or batch3_run2_registration["goal"]["baseline_rmse"]
        != 5.869105573928044
        or batch3_run2_registration["goal"]["maximum_candidate_rmse"]
        != 5.669105573928044
        or batch3_run2_registration["parent"]["prediction_sha256"]
        != "737711c483cc3ce81ba94fa8e5dfb9c307f75c7691fd06be33631d80c355fad1"
        or len(batch3_run2_registration["registered_variants"]) < 3
        or batch3_run2_registration["protected_and_kaggle_boundaries"][
            "protected_reveals_added"
        ]
        != 0
        or batch3_run2_registration["protected_and_kaggle_boundaries"][
            "kaggle_submission_authorized"
        ]
        is not False
    ):
        raise RuntimeError("invalid V5 Batch 3 Run 2 registration")
    if batch3_run2_candidate["status"] == "metric_free_gate_passed":
        batch3_run2_gate = load(
            batch3_run2_candidate["implementation_gate"]
        )
        if (
            sha256(batch3_run2_candidate["implementation_source"])
            != batch3_run2_candidate["implementation_source_sha256"]
            or sha256(batch3_run2_candidate["implementation_gate"])
            != batch3_run2_candidate["implementation_gate_sha256"]
            or batch3_run2_registration["status"]
            != "metric_free_gate_passed_before_target_metric"
            or batch3_run2_gate["status"] != "passed"
            or batch3_run2_gate["hidden_truth_loaded"] is not False
            or batch3_run2_gate["F4_loaded"] is not False
            or batch3_run2_gate["protected_reveal_spent"] is not False
        ):
            raise RuntimeError(
                "invalid V5 Batch 3 Run 2 implementation gate"
            )

    mode_bank_invalidations = [
        record
        for record in ledger.get("active_lineage_invalidations", [])
        if record["id"] == "c016-mode-bank-stratified-fold-exposure"
    ]
    if len(mode_bank_invalidations) != 1:
        raise RuntimeError("missing C016 mode-bank lineage invalidation")
    mode_bank_invalidation = mode_bank_invalidations[0]
    if (
        mode_bank_invalidation["status"]
        != "active_lineage_quarantine_repaired"
        or mode_bank_invalidation["protected_outer_reveal_spent"] is not False
        or mode_bank_invalidation["f4_metric_computed"] is not False
        or sha256(mode_bank_invalidation["record"])
        != mode_bank_invalidation["record_sha256"]
        or sha256(mode_bank_invalidation["canonical_repair"])
        != mode_bank_invalidation["canonical_repair_sha256"]
    ):
        raise RuntimeError("invalid C016 mode-bank invalidation pointer")

    mode_bank_record = load(mode_bank_invalidation["record"])
    if (
        mode_bank_record["status"] != "active_lineage_quarantine_repaired"
        or mode_bank_record["protected_outer_reveal_spent"] is not False
        or mode_bank_record["f4_metric_computed"] is not False
        or mode_bank_record[
            "discovered_before_confirmation_or_protected_outer_reveal"
        ]
        is not True
    ):
        raise RuntimeError("invalid C016 mode-bank quarantine record")
    for source in mode_bank_record["fail_closed_code"]:
        if sha256(source["path"]) != source["sha256"]:
            raise RuntimeError("C016 fail-closed source changed")

    r094_invalidations = [
        record
        for record in ledger.get("active_lineage_invalidations", [])
        if record["id"]
        == "batch3-r094-robust-transitive-mode-bank-exposure"
    ]
    if len(r094_invalidations) != 1:
        raise RuntimeError("missing R094-robust lineage invalidation")
    r094_invalidation = r094_invalidations[0]
    r094_record = load(r094_invalidation["record"])
    if (
        r094_invalidation["status"] != "active_lineage_quarantine"
        or sha256(r094_invalidation["record"])
        != r094_invalidation["record_sha256"]
        or r094_invalidation["protected_outer_reveal_spent"] is not False
        or r094_invalidation["f4_metric_computed"] is not False
        or r094_record["status"]
        != "r094_robust_invalidated_before_protected_metric"
        or r094_record["sealed_attempt"]["protected_metric_computed"]
        is not False
        or r094_record["sealed_attempt"]["protected_reveal_spent"]
        is not False
        or r094_record["clean_reassessment"][
            "clean_candidate_clears_authorized_exception"
        ]
        is not False
        or sha256(r094_record["fail_closed_source"]["path"])
        != r094_record["fail_closed_source"]["sha256"]
    ):
        raise RuntimeError("invalid R094-robust lineage invalidation")

    canonical = mode_bank_record["canonical_repair"]
    if (
        canonical["parent_count"] != 15
        or canonical["candidate_count"] != 240
        or canonical["removed_parent"] != "strat"
        or canonical["stratified_parent_loaded"] is not False
        or canonical["protected_outer_evidence_used"] is not False
    ):
        raise RuntimeError("invalid C016 canonical repair contract")
    for artifact in (
        "builder",
        "finalizer",
        "mode_result",
        "mode_archive",
        "c016_result",
        "c016_archive",
    ):
        if sha256(canonical[artifact]) != canonical[f"{artifact}_sha256"]:
            raise RuntimeError(f"C016 canonical {artifact} changed")

    repair = load(mode_bank_invalidation["canonical_repair"])
    controls = repair["controls"]
    if (
        repair["status"] != "leak_repaired_scope_decision_pending"
        or repair["invalidation"] != mode_bank_invalidation["record"]
        or repair["invalidation_sha256"]
        != mode_bank_invalidation["record_sha256"]
        or repair["protected_outer_reveal_spent"] is not False
        or repair["kaggle_dataset_created"] is not False
        or repair["kaggle_kernel_created"] is not False
        or repair["kaggle_submission_made"] is not False
        or not all(
            controls[key] is expected
            for key, expected in {
                "all_checkpoint_paths_strat_free": True,
                "all_candidate_names_strat_free": True,
                "all_fold_banks_rebuilt_from_checkpoints": True,
                "old_tainted_bank_rejected_by_legacy_clean_loader": True,
                "old_tainted_bank_rejected_by_mode_selector_loader": True,
                "canonical_output_forbidden_stratified_parent_loaded": False,
                "confirmation_regroupings_loaded": False,
                "audit_fold_loaded": False,
                "f4_metric_computed": False,
            }.items()
        )
    ):
        raise RuntimeError("invalid C016 canonical repair record")
    if repair["canonical_outputs"] != {
        key: canonical[key]
        for key in (
            "mode_result",
            "mode_result_sha256",
            "mode_archive",
            "mode_archive_sha256",
            "c016_result",
            "c016_result_sha256",
            "c016_archive",
            "c016_archive_sha256",
        )
    }:
        raise RuntimeError("C016 canonical output pointers disagree")
    for fold_bank in repair["fold_banks"]:
        if sha256(fold_bank["result"]) != fold_bank["result_sha256"]:
            raise RuntimeError("C016 original-safe fold-bank record changed")

    mode_result = load(canonical["mode_result"])
    c016_result = load(canonical["c016_result"])
    expected_gain = 0.1210103266375624
    if (
        mode_result["parent_count"] != 15
        or mode_result["candidate_count"] != 240
        or mode_result["removed_parent"] != "strat"
        or mode_result["forbidden_stratified_parent_loaded"] is not False
        or mode_result["confirmation_regroupings_loaded"] is not False
        or mode_result["audit_fold_loaded"] is not False
        or c016_result["forbidden_stratified_parent_loaded"] is not False
        or c016_result["confirmation_regroupings_loaded"] is not False
        or c016_result["audit_fold_loaded"] is not False
        or c016_result["positive_total_gain_folds"] != 4
        or abs(c016_result["gain_versus_v20"] - expected_gain) > 1e-12
        or abs(
            repair["eligible_development_result"]["gain_versus_v20"]
            - expected_gain
        )
        > 1e-12
    ):
        raise RuntimeError("invalid C016 original-fold-safe result")
    if ledger["currently_authorized_kaggle_submissions"] != active[
        "kaggle_submissions_authorized"
    ]:
        raise RuntimeError("Kaggle authorization mismatch")

    predecessor = policy["naming_boundary"]
    if sha256(predecessor["sealed_predecessor_result"]) != predecessor[
        "sealed_predecessor_result_sha256"
    ]:
        raise RuntimeError("sealed predecessor result changed")
    if policy["baseline"]["historical_v20_parity_oof_rmse"] != active[
        "baseline_oof_rmse"
    ]:
        raise RuntimeError("protected baseline mismatch")

    active_batch_number = (
        ledger["completed_batches"] + 1
        if ledger["remaining_batches"] > 0
        else None
    )
    next_run_number = (
        high_watermarks.get(active_batch_number, 0) + 1
        if active_batch_number is not None
        else None
    )
    print(
        "cv_protocol_v5 active "
        f"registered_families={len(ledger['research_queue'])} "
        f"active_families={active_candidates} "
        f"outer_reveals={ledger['total_protected_outer_oof_reveals']}/"
        f"{ledger['maximum_batches']} "
        f"active_batch={active_batch_number} "
        f"run_numbering_epoch={current_numbering_epoch} "
        f"next_development_run={next_run_number} "
        f"kaggle_submissions="
        f"{ledger['currently_authorized_kaggle_submissions']}"
    )


if __name__ == "__main__":
    main()
