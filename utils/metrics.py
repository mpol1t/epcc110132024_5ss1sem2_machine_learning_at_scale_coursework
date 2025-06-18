import torch


@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    """
    Converts an index to a latitude value.

    :param j: Tensor of indices.
    :param num_lat: Total number of latitude points.
    :return: Corresponding latitude values in degrees.
    """
    return 90. - j * 180. / float(num_lat - 1)


@torch.jit.script
def latitude_weighting_factor(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    """
    Calculates latitude weighting factors for weighting errors in loss calculations.

    :param j: Tensor of latitude indices.
    :param num_lat: Total number of latitude points.
    :param s: Sum of cosines of latitudes, used for normalization.
    :return: Tensor of latitude weighting factors.
    """
    return num_lat * torch.cos(3.1416 / 180. * lat(j, num_lat)) / s


@torch.jit.script
def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the weighted root mean square error (RMSE) for each channel, considering the latitude.

    :param pred: Prediction tensor of shape [n, c, h, w].
    :param target: Target tensor of the same shape as prediction.
    :return: Tensor of weighted RMSE for each channel.
    """
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    # num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416 / 180. * lat(lat_t, num_lat)))

    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target) ** 2., dim=(-1, -2)))

    return result


@torch.jit.script
def weighted_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean weighted root mean square error (RMSE) across all channels.

    :param pred: Prediction tensor of shape [n, c, h, w].
    :param target: Target tensor of the same shape as prediction.
    :return: Scalar tensor of the mean weighted RMSE.
    """
    return torch.mean(weighted_rmse_channels(pred, target), dim=0)


@torch.jit.script
def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the weighted accuracy for each channel, considering the latitude.

    :param pred: Prediction tensor of shape [n, c, h, w].
    :param target: Target tensor of the same shape as prediction.
    :return: Tensor of weighted accuracy for each channel.
    """
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    # num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416 / 180. * lat(lat_t, num_lat)))

    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))

    return torch.sum(
        weight * pred * target,
        dim=(-1, -2)
    ) / torch.sqrt(
        torch.sum(
            weight * pred * pred,
            dim=(-1, -2)
        ) * torch.sum(
            weight * target * target,
            dim=(-1, -2)
        )
    )


@torch.jit.script
def weighted_acc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean weighted accuracy across all channels.

    :param pred: Prediction tensor of shape [n, c, h, w].
    :param target: Target tensor of the same shape as prediction.
    :return: Scalar tensor of the mean weighted accuracy.
    """
    return torch.mean(weighted_acc_channels(pred, target), dim=0)

#
# class WeightedRMSE(Metric):
#     def __init__(self):
#         super().__init__(dist_sync_on_step=False)
#         self.total_batches = None
#         self.cumulative_rmse = None
#
#         self.add_state("cumulative_rmse", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")
#
#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         """
#         Updates the states for cumulative RMSE and count with each batch.
#
#         :param preds: Prediction tensor of shape [n, c, h, w].
#         :param target: Target tensor of the same shape as prediction.
#         """
#         rmse_batch = weighted_rmse(preds, target)
#
#         self.cumulative_rmse += rmse_batch.sum()
#         self.total_batches += preds.shape[0]
#
#     def compute(self):
#         """
#         Computes the average weighted RMSE over all batches.
#
#         :return: The mean weighted RMSE computed over all batches.
#         """
#         return self.cumulative_rmse / self.total_batches
#
#     def reset(self):
#         """Resets the states at the end of an epoch."""
#         super().reset()
#         self.cumulative_rmse = torch.tensor(0.0)
#         self.total_batches = torch.tensor(0)
#
