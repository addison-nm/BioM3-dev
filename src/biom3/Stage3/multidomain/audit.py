"""Pre-flight gates for multidomain training.

Three properties are checked before a run is allowed to start:

* the composed decoder is at its additive null, so it begins bit-exactly equal to
  the independent experts — and, conversely, that the coupling is actually wired
  into the forward pass at all;
* exactly the intended parameters are trainable, with prior references frozen and
  no stray tensor in the optimizer;
* each expert's drift from its reference is zero at initialisation.

Failures raise rather than assert: ``assert`` statements vanish under ``python
-O``, and these gates stand between a typo and a corrupted multi-day run.
"""

import torch

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


class AuditFailure(RuntimeError):
    """A pre-flight gate rejected the model."""


def assert_additive_null(model, *, batch=2, seq_len=None, device=None, seed=0):
    """Verify the composed forward equals the independent experts, bit-exactly.

    Checks, for every domain d:
      composed(couple=True)[:, d] == composed(couple=False)[:, d]
                                  == expert d's own standalone forward

    Then perturbs one output projection and requires the logits to move, so a
    coupling that was never connected cannot pass by being trivially null.

    Bit-exactness (not a tolerance) is the right bar: with zeroed output
    projections the cross term is a matmul against an exactly-zero matrix, so the
    sum is exactly ``out + 0``. A tolerance would let a genuinely non-null
    coupling through.
    """
    if not model.coupling.is_additive_null(tol=0.0):
        raise AuditFailure(
            "coupling is not at the additive null: its output projections are "
            "non-zero, so the composed model does not start equal to the experts"
        )

    transformer = model.experts[0].transformer
    seq_len = seq_len or transformer.max_seq_len
    num_domains = model.num_domains
    device = device or next(model.parameters()).device

    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_classes = transformer.out.out_features
    x = torch.randint(0, num_classes, (batch, num_domains, seq_len),
                      generator=generator).to(device)
    t = torch.randint(0, seq_len, (batch,), generator=generator).float().to(device)
    y_c = torch.randn(batch, num_domains, transformer.y_mlp[0].in_features,
                      generator=generator).to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            coupled = model(x, t, y_c, couple=True)
            uncoupled = model(x, t, y_c, couple=False)
            for d in range(num_domains):
                standalone = model.standalone_logits(d, x[:, d], t, y_c[:, d])
                if not torch.equal(coupled[:, d], uncoupled[:, d]):
                    raise AuditFailure(
                        f"domain {d}: coupled and uncoupled forwards differ at the "
                        "additive null"
                    )
                if not torch.equal(coupled[:, d], standalone):
                    raise AuditFailure(
                        f"domain {d}: composed forward differs from the standalone "
                        "expert forward at the additive null"
                    )

            # Converse: the cross term must be reachable from the forward pass.
            probe = model.coupling.out_proj[
                next(iter(model.coupling.out_proj))][0]
            saved = probe.weight.detach().clone()
            probe.weight.fill_(0.05)
            perturbed = model(x, t, y_c, couple=True)
            probe.weight.copy_(saved)
        if torch.equal(coupled, perturbed):
            raise AuditFailure(
                "perturbing a coupling output projection did not change the "
                "logits — the cross term is not wired into the forward pass"
            )
    finally:
        model.train(was_training)

    logger.info("Additive-null gate passed (bit-exact over %s domains)", num_domains)
    return True


def expert_delta_norms(model, references) -> dict:
    """Relative drift ``||W - W_ref||_F / ||W_ref||_F`` for each expert."""
    deltas = {}
    for d, (expert, reference) in enumerate(zip(model.experts, references)):
        numerator = 0.0
        denominator = 0.0
        for param, ref in zip(expert.parameters(), reference):
            numerator += (param.detach() - ref).pow(2).sum().item()
            denominator += ref.pow(2).sum().item()
        deltas[d] = (numerator ** 0.5) / (denominator ** 0.5 + 1e-12)
    return deltas


def audit_trainable_parameters(model, *, train_experts: bool,
                               optimizer=None, expected_trainable=None) -> dict:
    """Inventory the trainable parameters, partitioned by role.

    Returns a report; :func:`enforce_audit` decides whether it is acceptable. The
    report is written to the run's artifacts so a run's freeze state is auditable
    after the fact.
    """
    coupling_ids = {id(p) for p in model.coupling.parameters()}
    expert_ids = {id(p) for expert in model.experts for p in expert.parameters()}

    report = {
        "num_domains": model.num_domains,
        "coupling": model.coupling.parameter_report(),
        "train_experts": bool(train_experts),
        "coupling_trainable": 0,
        "experts_trainable": 0,
        "other_trainable": 0,
        "other_trainable_names": [],
        "frozen_expert_tensors": 0,
        "expected_trainable": expected_trainable,
    }

    for name, param in model.named_parameters():
        if id(param) in coupling_ids:
            if param.requires_grad:
                report["coupling_trainable"] += param.numel()
        elif id(param) in expert_ids:
            if param.requires_grad:
                report["experts_trainable"] += param.numel()
            else:
                report["frozen_expert_tensors"] += 1
        elif param.requires_grad:
            report["other_trainable"] += param.numel()
            report["other_trainable_names"].append(name)

    report["total_trainable"] = (
        report["coupling_trainable"] + report["experts_trainable"]
        + report["other_trainable"]
    )

    # Aliased re-registration inflates state_dict() without showing up in
    # named_parameters(), so compare the two counts directly.
    param_entries = sum(1 for _ in model.parameters())
    state_param_keys = {name for name, _ in model.named_parameters()}
    report["parameter_tensors"] = param_entries
    report["named_parameter_keys"] = len(state_param_keys)
    report["state_dict_entries"] = len(model.state_dict())

    if optimizer is not None:
        report["optimizer"] = _audit_optimizer(model, optimizer, train_experts)
    return report


def _audit_optimizer(model, optimizer, train_experts) -> dict:
    """Check the optimizer holds only intended parameters, in sane groups.

    When experts are regularized toward a reference by ``lambda ||W - W_ref||^2``,
    they must sit in a ``weight_decay=0`` group: decay would pull the same weights
    toward zero while the prior pulls them toward the reference, which is two
    conflicting priors on one tensor.
    """
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    expert_ids = {id(p) for expert in model.experts for p in expert.parameters()}

    seen = set()
    decayed_experts = 0
    for group in optimizer.param_groups:
        weight_decay = group.get("weight_decay", 0.0)
        for param in group["params"]:
            seen.add(id(param))
            if train_experts and id(param) in expert_ids and weight_decay:
                decayed_experts += 1

    return {
        "params_in_optimizer": len(seen),
        "trainable_params": len(trainable_ids),
        "missing_from_optimizer": len(trainable_ids - seen),
        "frozen_in_optimizer": len(seen - trainable_ids),
        "prior_regularized_with_weight_decay": decayed_experts,
    }


def enforce_audit(report: dict, *, require_null_coupling=True):
    """Raise :class:`AuditFailure` when the inventory is not what was intended."""
    problems = []

    if report["other_trainable"]:
        problems.append(
            f"{len(report['other_trainable_names'])} tensor(s) outside the coupling "
            f"and experts are trainable, e.g. {report['other_trainable_names'][:5]}"
        )
    if not report["coupling_trainable"]:
        problems.append("the coupling has no trainable parameters")
    if require_null_coupling and not report["coupling"]["additive_null"]:
        problems.append("the coupling is not at the additive null")

    if report["train_experts"]:
        if not report["experts_trainable"]:
            problems.append("train_experts is set but no expert parameter is trainable")
    elif report["experts_trainable"]:
        problems.append(
            f"train_experts is off but {report['experts_trainable']} expert "
            "parameters are trainable"
        )

    if report["named_parameter_keys"] != report["parameter_tensors"]:
        problems.append(
            "parameter tensors are registered under more than one name — aliased "
            "registration inflates every checkpoint"
        )

    expected = report.get("expected_trainable")
    if expected is not None and report["total_trainable"] != expected:
        problems.append(
            f"trainable parameter count {report['total_trainable']} != the "
            f"{expected} declared in the config"
        )

    optimizer = report.get("optimizer")
    if optimizer:
        if optimizer["missing_from_optimizer"]:
            problems.append(
                f"{optimizer['missing_from_optimizer']} trainable parameter(s) are "
                "not in the optimizer"
            )
        if optimizer["frozen_in_optimizer"]:
            problems.append(
                f"{optimizer['frozen_in_optimizer']} frozen parameter(s) are in the "
                "optimizer"
            )
        if optimizer["prior_regularized_with_weight_decay"]:
            problems.append(
                f"{optimizer['prior_regularized_with_weight_decay']} prior-regularized "
                "expert parameter(s) sit in a weight_decay > 0 group; decay pulls them "
                "toward zero while the prior pulls them toward the reference"
            )

    if problems:
        raise AuditFailure("pre-flight audit failed: " + "; ".join(problems))

    logger.info(
        "Freeze audit passed: coupling=%s experts=%s other=0",
        report["coupling_trainable"], report["experts_trainable"],
    )
    return True
