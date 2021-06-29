from functools import lru_cache
from math import factorial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class BezierSimplexDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: str,
        label: str,
        header: int = 0,
        delimiter: str = " ",
        batch_size: Optional[int] = None,
        split_ratio: float = 0.5,
        normalize: str = "none",  # "max", "std", "quantile" or "none"
    ):
        # REQUIRED
        super().__init__()
        self.data = data
        self.label = label
        self.header = header
        self.delimiter = delimiter
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.normalize = normalize
        with open(self.data) as f:
            self.n_params = len(f.readline().split(self.delimiter))
        with open(self.label) as f:
            self.n_values = len(f.readline().split(self.delimiter))

    def setup(self, stage: Optional[str]=None):
        # OPTIONAL
        params = torch.from_numpy(
            np.loadtxt(self.data, delimiter=self.delimiter, skiprows=self.header)
        )
        values = torch.from_numpy(
            np.loadtxt(self.label, delimiter=self.delimiter, skiprows=self.header)
        )
        if self.normalize == "max":
            mins = values.amin(dim=0)
            maxs = values.amax(dim=0)
            mins[mins == maxs] -= 0.5  # Avoid division by zero
            maxs[mins == maxs] += 0.5  # Avoid division by zero
            values = (values - mins) / (maxs - mins)
        elif self.normalize == "std":
            stds, means = torch.std_mean(values, dim=0)
            stds[stds == 0.0] = 1.0  # Avoid division by zero
            values = (values - means) / stds
        elif self.normalize == "quantile":
            q = 0.05  # Ignore 5% outliers
            mins = values.quantile(q, dim=0)
            maxs = values.quantile(1 - q, dim=0)
            mins[mins == maxs] -= 0.5  # Avoid division by zero
            maxs[mins == maxs] += 0.5  # Avoid division by zero
            values = (values - mins) / (maxs - mins)
        xy = TensorDataset(params, values)
        size = len(xy)
        n_train = int(size * self.split_ratio)
        self.trainset, self.valset = random_split(xy, [n_train, size - n_train])

    def train_dataloader(self) -> DataLoader:
        # REQUIRED
        train_loader = DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size or len(self.trainset),
        )
        return train_loader
        
    def val_dataloader(self) -> DataLoader:
        # OPTIONAL
        val_loader = DataLoader(
            self.valset,
            batch_size=self.batch_size or len(self.valset),
        )
        return val_loader
        
    def test_dataloader(self) -> DataLoader:
        # OPTIONAL
        return self.val_dataloader()


Index = Tuple[int, ...]

def indices(dim: int, deg: int) -> Iterable[Index]:
    def iterate(c, r):
        if len(c) == dim - 1:
            yield c + (r, )
        else:
            for i in range(r, -1, -1):
                yield from iterate(c + (i, ), r - i)

    yield from iterate((), deg)


@lru_cache(1024)
def polynom(degree: int, index: Iterable[int]) -> float:
    r = factorial(degree)
    for i in index:
        r /= factorial(i)
    return r


def monomial(var: Iterable[float], deg: Iterable[int]) -> torch.Tensor:
    var = torch.as_tensor(var)
    deg = torch.as_tensor(deg, device=var.device)
    return (var ** deg).prod(dim=-1)


class BezierSimplex(pl.LightningModule):
    def __init__(
        self,
        n_params: int,
        n_values: int,
        degree: int,
    ):
        # REQUIRED
        super().__init__()
        self.n_params = n_params
        self.n_values = n_values
        self.degree = degree
        self.control_points = nn.ParameterDict({
            str(i): nn.Parameter(torch.randn(n_values)) for i in indices(n_params, degree)
        })

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # REQUIRED
        x = 0
        for i in indices(self.n_params, self.degree):
            x += polynom(self.degree, i) * torch.outer(monomial(t, i), self.control_points[str(i)])
        return x

    def training_step(self, batch, batch_idx) -> Dict[str, Any]:
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        self.log("train_mse", loss, sync_dist=True)
        return {'loss': loss, 'log': tensorboard_logs}
 
    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log("val_mse", mse, sync_dist=True)
        self.log("val_mae", mae, sync_dist=True)
        return {'val_loss': mse}
 
    def validation_end(self, outputs) -> Dict[str, Any]:
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log("val_avg_mse", avg_loss, sync_dist=True)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
 
    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_mae", mae, sync_dist=True)
        return {'test_loss': mse}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # REQUIRED
        optimizer = torch.optim.LBFGS(self.parameters())
        return optimizer
 
    def meshgrid(self, num: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Meshgrid

        Parameters
        ----------
        num
            number of grid points on each edge

        Returns
        ----------
        ts : torch.Tensor
            Array of mesh grid
        xs : torch.Tensor
            The concatenated array
        """
        ts = torch.Tensor(list(indices(dim=self.n_params, deg=num))) / num
        xs = self.forward(ts)
        return ts, xs


def fit(
    params: torch.Tensor,
    values: torch.Tensor,
    degree: int,
    batch_size: Optional[int]=None,
    max_epochs: int=1000,
    gpus: Union[str, int, List[int]]=-1,
    num_nodes: int=1,
    accelerator: str="ddp",
):
    data = TensorDataset(params, values)
    dl = DataLoader(data, batch_size=batch_size or len(data))
    bs = BezierSimplex(n_params=int(params.shape[1]), n_values=int(values.shape[1]), degree=degree)
    trainer = pl.Trainer(
        gpus=gpus,
        auto_select_gpus=True,
        accelerator=accelerator,
        num_nodes=num_nodes,
        max_epochs=max_epochs,
    )
    trainer.fit(bs, dl)
    return bs