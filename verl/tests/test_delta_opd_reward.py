import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import (
    _compute_delta_opd_metrics,
    _compute_delta_opd_sampled_metrics,
    _compute_delta_opd_sampled_tensors,
    _compute_ref_teacher_overlap_metrics,
    _compute_student_weighted_teacher_logprob_metrics,
    _pop_direct_opd_rollout_options,
)
from verl.workers.actor.dp_actor import (
    _build_bottom_fraction_token_mask,
    _build_random_fraction_token_mask,
    _build_top_fraction_token_mask,
    _compute_delta_opd_rm_scores,
    _compute_topk_js_divergence,
    _derive_random_token_selection_seed,
)
from verl.workers.config.actor import ActorConfig


def test_direct_opd_reward_uses_detached_student_topk_weights():
    student_logp = torch.tensor([[[-0.1, -2.0, -3.0]]], requires_grad=True)
    teacher_rl_logp = torch.tensor([[[-0.2, -1.0, -4.0]]])
    teacher_ref_logp = torch.tensor([[[-0.5, -2.0, -2.5]]])

    rm_scores, delta = _compute_delta_opd_rm_scores(
        student_logp=student_logp,
        teacher_rl_logp=teacher_rl_logp,
        teacher_ref_logp=teacher_ref_logp,
    )

    expected_delta = teacher_rl_logp - teacher_ref_logp
    expected_weights = torch.softmax(student_logp.detach(), dim=-1)

    assert rm_scores.shape == torch.Size([1, 1, 3])
    assert torch.allclose(delta, expected_delta)
    assert torch.allclose(rm_scores, expected_weights * expected_delta)
    assert not rm_scores.requires_grad
    assert not delta.requires_grad


def test_direct_opd_reward_masks_invalid_candidates_before_weighting():
    student_logp = torch.tensor([[[-0.1, -2.0, -3.0]]], requires_grad=True)
    teacher_rl_logp = torch.tensor([[[-0.2, -torch.inf, -4.0]]])
    teacher_ref_logp = torch.tensor([[[-0.5, -2.0, -2.5]]])
    valid_mask = torch.tensor([[[True, False, True]]])

    rm_scores, delta = _compute_delta_opd_rm_scores(
        student_logp=student_logp,
        teacher_rl_logp=teacher_rl_logp,
        teacher_ref_logp=teacher_ref_logp,
        valid_mask=valid_mask,
    )

    expected_delta = (teacher_rl_logp - teacher_ref_logp).masked_fill(~valid_mask, 0.0)
    masked_student_logp = student_logp.detach().masked_fill(~valid_mask, -torch.inf)
    expected_weights = torch.softmax(masked_student_logp, dim=-1)
    expected_weights = torch.nan_to_num(expected_weights, nan=0.0, posinf=0.0, neginf=0.0)

    assert torch.isfinite(rm_scores).all()
    assert torch.isfinite(delta).all()
    assert torch.allclose(delta, expected_delta)
    assert torch.allclose(rm_scores, expected_weights * expected_delta)
    assert rm_scores[0, 0, 1].item() == 0.0
    assert not rm_scores.requires_grad


def test_direct_opd_reward_rejects_mismatched_shapes():
    student_logp = torch.zeros(2, 3, 4)
    teacher_rl_logp = torch.zeros(2, 3, 4)
    teacher_ref_logp = torch.zeros(2, 3, 5)

    try:
        _compute_delta_opd_rm_scores(
            student_logp=student_logp,
            teacher_rl_logp=teacher_rl_logp,
            teacher_ref_logp=teacher_ref_logp,
        )
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected mismatched Direct-OPD shapes to raise ValueError")


def test_direct_opd_rollout_options_are_removed_before_rollout_config_instantiation():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "reward_mode": "delta_opd",
                    "top_k_strategy": "only_stu",
                }
            }
        }
    )

    options = _pop_direct_opd_rollout_options(config)

    assert options == {"reward_mode": "delta_opd"}
    assert "reward_mode" not in config.actor_rollout_ref.rollout
    assert config.actor_rollout_ref.rollout.top_k_strategy == "only_stu"


def test_delta_opd_sampled_rollout_mode_is_preserved_as_direct_opd_option():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "reward_mode": "delta_opd_sampled",
                    "log_prob_top_k": 0,
                }
            }
        }
    )

    options = _pop_direct_opd_rollout_options(config)

    assert options == {"reward_mode": "delta_opd_sampled"}
    assert "reward_mode" not in config.actor_rollout_ref.rollout
    assert config.actor_rollout_ref.rollout.log_prob_top_k == 0


def test_delta_opd_sampled_tensors_use_sampled_teacher_ref_gap():
    teacher_sampled_log_probs = torch.tensor([[-0.2, -1.0, -3.0]])
    teacher_ref_sampled_log_probs = torch.tensor([[-0.5, -0.7, -2.0]])
    response_mask = torch.tensor([[1, 1, 0]])

    tensors = _compute_delta_opd_sampled_tensors(
        teacher_sampled_log_probs=teacher_sampled_log_probs,
        teacher_ref_sampled_log_probs=teacher_ref_sampled_log_probs,
        response_mask=response_mask,
    )

    expected_delta = teacher_sampled_log_probs - teacher_ref_sampled_log_probs
    expected_rm_scores = expected_delta * response_mask

    assert torch.allclose(tensors["delta_opd_sampled_log_ratio"], expected_delta)
    assert torch.allclose(tensors["rm_scores"], expected_rm_scores)
    assert torch.allclose(tensors["teacher_ref_sampled_log_probs"], teacher_ref_sampled_log_probs)
    assert not tensors["rm_scores"].requires_grad


def test_delta_opd_sampled_metrics_use_sampled_prefix_and_response_mask():
    sampled_delta = torch.tensor([[0.3, -0.3, 1.0]])
    response_mask = torch.tensor([[1, 1, 0]])
    rm_scores = sampled_delta * response_mask

    metrics = _compute_delta_opd_sampled_metrics(
        sampled_delta=sampled_delta,
        rm_scores=rm_scores,
        response_mask=response_mask,
    )

    assert metrics["delta_opd_sampled/log_ratio_mean"] == 0.0
    assert metrics["delta_opd_sampled/log_ratio_pos_frac"] == 0.5
    assert metrics["delta_opd_sampled/reward_mean"] == 0.0
    assert metrics["delta_opd_sampled/reward_token_sum_mean"] == 0.0
    assert "delta_opd_sampled/log_ratio_mean_chunk_0_3" in metrics


def test_direct_opd_metrics_include_weighted_pos_frac_and_teacher_gap():
    student_topk_log_probs = torch.log(torch.tensor([[[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]]))
    teacher_on_student = torch.log(torch.tensor([[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]]))
    teacher_ref_on_student = torch.log(torch.tensor([[[0.3, 0.3, 0.4], [0.1, 0.7, 0.2]]]))
    response_mask = torch.tensor([[1, 1]])
    delta = teacher_on_student - teacher_ref_on_student
    rm_scores = torch.softmax(student_topk_log_probs, dim=-1) * delta

    metrics = {}
    metrics.update(
        _compute_delta_opd_metrics(
            delta=delta,
            rm_scores=rm_scores,
            response_mask=response_mask,
            student_topk_log_probs=student_topk_log_probs,
        )
    )
    metrics.update(
        _compute_student_weighted_teacher_logprob_metrics(
            student_topk_log_probs=student_topk_log_probs,
            teacher_on_student_log_probs=teacher_on_student,
            teacher_ref_on_student_log_probs=teacher_ref_on_student,
            response_mask=response_mask,
        )
    )
    metrics.update(
        _compute_ref_teacher_overlap_metrics(
            teacher_ref_on_student_log_probs=teacher_ref_on_student,
            teacher_ref_overlap_mask=torch.tensor([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            response_mask=response_mask,
        )
    )

    assert "delta_opd/weighted_reward_mean" in metrics
    assert "delta_opd/topk_log_ratio_weighted_pos_frac" in metrics
    assert "delta_opd/student_weighted_pos_frac" in metrics
    assert "delta_opd/student_weighted_teacher_logprob" in metrics
    assert "delta_opd/student_weighted_teacher_ref_logprob" in metrics
    assert "delta_opd/student_weighted_teacher_logprob_gap" in metrics
    assert "delta_opd/ref_teacher_overlap_ratio" in metrics


def test_topk_js_divergence_is_zero_for_identical_distributions():
    log_probs = torch.log(torch.tensor([[[0.4, 0.2], [0.1, 0.3]]], requires_grad=True))

    js_divergence = _compute_topk_js_divergence(base_logp=log_probs, rl_logp=log_probs)

    assert js_divergence.shape == torch.Size([1, 2])
    assert torch.allclose(js_divergence, torch.zeros_like(js_divergence), atol=1e-7)
    assert not js_divergence.requires_grad


def test_topk_js_divergence_includes_remaining_mass_as_tail_bucket():
    base_probs = torch.tensor([0.4, 0.1, 0.5])
    rl_probs = torch.tensor([0.1, 0.3, 0.6])
    mixture = 0.5 * (base_probs + rl_probs)
    expected = 0.5 * (base_probs * (base_probs.log() - mixture.log())).sum()
    expected += 0.5 * (rl_probs * (rl_probs.log() - mixture.log())).sum()

    js_divergence = _compute_topk_js_divergence(
        base_logp=base_probs[:2].log().view(1, 1, 2),
        rl_logp=rl_probs[:2].log().view(1, 1, 2),
    )

    assert torch.allclose(js_divergence, expected.view(1, 1), atol=1e-7)


def test_top_fraction_js_mask_is_computed_per_response_with_ceil():
    js_divergence = torch.arange(42, dtype=torch.float32).view(2, 21)
    response_mask = torch.ones(2, 21, dtype=torch.long)
    response_mask[0, -1] = 0

    selected = _build_top_fraction_token_mask(
        js_divergence=js_divergence,
        response_mask=response_mask,
        top_fraction=0.05,
    )

    assert selected[0].sum().item() == 1  # ceil(20 * 0.05)
    assert selected[1].sum().item() == 2  # ceil(21 * 0.05)
    assert selected[0, 19]
    assert not selected[0, 20]
    assert selected[1, 19]
    assert selected[1, 20]


def test_bottom_fraction_js_mask_is_computed_per_response_with_ceil():
    js_divergence = torch.arange(42, dtype=torch.float32).view(2, 21)
    response_mask = torch.ones(2, 21, dtype=torch.long)
    response_mask[0, -1] = 0

    selected = _build_bottom_fraction_token_mask(
        js_divergence=js_divergence,
        response_mask=response_mask,
        bottom_fraction=0.10,
    )

    assert selected[0].sum().item() == 2  # ceil(20 * 0.10)
    assert selected[1].sum().item() == 3  # ceil(21 * 0.10)
    assert selected[0, 0]
    assert selected[0, 1]
    assert not selected[0, 20]
    assert selected[1, 0]
    assert selected[1, 1]
    assert selected[1, 2]


def test_random_fraction_token_mask_uses_valid_tokens_and_ceil():
    response_mask = torch.ones(3, 21, dtype=torch.long)
    response_mask[0, -1] = 0
    response_mask[2] = 0

    selected = _build_random_fraction_token_mask(
        response_mask=response_mask,
        top_fraction=0.10,
        seed=42,
    )

    assert selected[0].sum().item() == 2  # ceil(20 * 0.10)
    assert selected[1].sum().item() == 3  # ceil(21 * 0.10)
    assert selected[2].sum().item() == 0
    assert not selected[~response_mask.bool()].any()


def test_random_fraction_token_mask_is_reproducible_and_changes_by_step():
    response_mask = torch.ones(4, 100, dtype=torch.long)
    step_1_seed = _derive_random_token_selection_seed(
        base_seed=42,
        global_step=1,
        data_parallel_rank=0,
        data_parallel_world_size=8,
    )
    step_2_seed = _derive_random_token_selection_seed(
        base_seed=42,
        global_step=2,
        data_parallel_rank=0,
        data_parallel_world_size=8,
    )

    selected_1 = _build_random_fraction_token_mask(response_mask, top_fraction=0.10, seed=step_1_seed)
    selected_1_repeat = _build_random_fraction_token_mask(response_mask, top_fraction=0.10, seed=step_1_seed)
    selected_2 = _build_random_fraction_token_mask(response_mask, top_fraction=0.10, seed=step_2_seed)

    assert step_1_seed == 42
    assert step_2_seed == 50
    assert torch.equal(selected_1, selected_1_repeat)
    assert not torch.equal(selected_1, selected_2)


def test_random_fraction_token_mask_does_not_follow_js_ranking():
    response_mask = torch.ones(1, 100, dtype=torch.long)
    js_divergence = torch.arange(100, dtype=torch.float32).view(1, 100)

    random_selected = _build_random_fraction_token_mask(response_mask, top_fraction=0.10, seed=42)
    top_js_selected = _build_top_fraction_token_mask(js_divergence, response_mask, top_fraction=0.10)

    assert random_selected.sum().item() == top_js_selected.sum().item() == 10
    assert not torch.equal(random_selected, top_js_selected)


def test_actor_config_accepts_bottom_js_token_selection_mode():
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        js_token_selection_mode="bottom_js",
    )

    assert config.js_token_selection_mode == "bottom_js"


def test_actor_config_rejects_invalid_js_token_selection_mode():
    try:
        ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
            js_token_selection_mode="largest",
        )
    except ValueError as exc:
        assert "js_token_selection_mode" in str(exc)
    else:
        raise AssertionError("Expected invalid js_token_selection_mode to raise ValueError")
