import torch
from torchmetrics import Metric


@torch.jit.script
def l2_loss_opt(pred: torch.Tensor, target: torch.Tensor):
    """
    Computes the normalized L2 loss between predictions and targets.

    :param pred: Predictions tensor.
    :param target: Target tensor.
    :return: Mean of the normalized L2 losses across all examples in the batch.
    """
    num_examples = pred.shape[0]

    target_reshaped = target.reshape(num_examples, -1)

    diff_norms = torch.norm(pred.reshape(num_examples, -1) - target_reshaped, 2, 1)
    y_norms = torch.norm(target_reshaped, 2, 1)

    return torch.mean(diff_norms / y_norms)


class NormalizedL2Loss(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        num_examples = preds.shape[0]
        target_reshaped = target.reshape(num_examples, -1)
        diff_norms = torch.norm(preds.reshape(num_examples, -1) - target_reshaped, 2, 1)
        y_norms = torch.norm(target_reshaped, 2, 1)

        self.total += torch.mean(diff_norms / y_norms)
        self.count += 1

    def compute(self):
        return self.total / self.count
