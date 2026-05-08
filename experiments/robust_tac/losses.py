import torch
import torch.nn.functional as F


def entropy(prob):
    mean_prob = prob.mean(dim=0).clamp_min(1e-9)
    return -(mean_prob * mean_prob.log()).sum()


def consistency_loss(anchor, neighbor):
    similarity = torch.sum(anchor * neighbor, dim=1).clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(similarity, torch.ones_like(similarity))


def kl_target_to_prediction(target_prob, pred_prob):
    return F.kl_div(pred_prob.clamp_min(1e-9).log(), target_prob.detach(), reduction="batchmean")


def per_sample_kl_target_to_prediction(target_prob, pred_prob):
    target = target_prob.detach().clamp_min(1e-9)
    pred = pred_prob.clamp_min(1e-9)
    return torch.sum(target * (target.log() - pred.log()), dim=1)


def tac_clean_loss(q_text, p_clean, balance_weight=5.0):
    loss_consist = consistency_loss(q_text, p_clean)
    loss_balance = entropy(q_text) + entropy(p_clean)
    return loss_consist - balance_weight * loss_balance, loss_consist, loss_balance


def per_sample_eg_kl_text(q_text, p_adv):
    return per_sample_kl_target_to_prediction(q_text, p_adv)


def loss_eg_text(q_text, p_adv):
    return per_sample_eg_kl_text(q_text, p_adv).mean()


def per_sample_eg_kl_text_symmetric(q_text, p_adv, eta_sym=0.1):
    teacher_to_adv = per_sample_kl_target_to_prediction(q_text, p_adv)
    adv_to_teacher = per_sample_kl_target_to_prediction(p_adv, q_text)
    return teacher_to_adv + eta_sym * adv_to_teacher


def loss_eg_text_symmetric(q_text, p_adv, eta_sym=0.1):
    return per_sample_eg_kl_text_symmetric(q_text, p_adv, eta_sym=eta_sym).mean()


def anchor_relation(features, noun_anchor_bank, tau_relation):
    if noun_anchor_bank is None:
        raise ValueError("anchor_kl requires noun_anchor_bank from tac_text guidance")
    logits = features @ noun_anchor_bank.t()
    return torch.softmax(logits / tau_relation, dim=1)


def per_sample_anchor_kl(v_clean, v_adv, noun_anchor_bank, tau_relation):
    r_clean = anchor_relation(v_clean, noun_anchor_bank, tau_relation)
    r_adv = anchor_relation(v_adv, noun_anchor_bank, tau_relation)
    return per_sample_kl_target_to_prediction(r_clean, r_adv), r_clean, r_adv


def cosine_factor(v_clean, v_adv):
    return F.cosine_similarity(v_clean, v_adv, dim=1) + 1.0
