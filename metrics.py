import torch
from utils import gather_log_probs, mask_hf_labels, masked_mean


def es_sentiment(pre_logits, post_logits, raw_targets, same_sent_mask, NULL_TOKEN=0):
    with torch.no_grad():
        mask, targ = mask_hf_labels(raw_targets)
        pos_mask = same_sent_mask.unsqueeze(-1) * mask
        neg_mask = (~same_sent_mask).unsqueeze(-1) * mask

        # Compute log likelihoods of pos/neg samples
        pre_edit_token_log_probs = gather_log_probs(pre_logits, targ)
        post_edit_token_log_probs = gather_log_probs(post_logits, targ)

        mean_pos_pre = masked_mean(pre_edit_token_log_probs, pos_mask)
        mean_pos_post = masked_mean(post_edit_token_log_probs, pos_mask)
        mean_neg_post = masked_mean(post_edit_token_log_probs, neg_mask)

        z_sent = (mean_pos_post - mean_neg_post).sigmoid()
        z_topic_raw = (mean_pos_post - mean_pos_pre).exp()
        z_topic = min(1, z_topic_raw)

        es_sent = z_sent * z_topic

        return {
            "acc_sent": es_sent,
            "z_sent": z_sent,
            "z_topic": z_topic,
            "z_topic_raw": z_topic_raw,
            "correct_probs": mean_pos_post,
            "wrong_probs": mean_neg_post,
        }


# For zsRE and F-NLI
def retain_rate(pre_logits, post_logits, mask=None):
    if pre_logits.shape[-1] == 1:
        pre_logits = pre_logits.squeeze(-1)
    if post_logits.shape[-1] == 1:
        post_logits = post_logits.squeeze(-1)

    assert pre_logits.shape == post_logits.shape
    assert pre_logits.shape[0] == mask.shape[0]

    if pre_logits.dim() == 1:
        # binary classification
        pre_preds = pre_logits > 0
        post_preds = post_logits > 0
        retain = (pre_preds == post_preds).float().mean()
    elif pre_logits.dim() == 3:
        # sequence modeling
        pre_preds = pre_logits.argmax(-1)
        post_preds = post_logits.argmax(-1)
        match = (pre_preds == post_preds) * mask
        retain = (match.sum(-1) == mask.sum(-1)).float().mean()
    else:
        raise NotImplementedError

    return retain.item()
