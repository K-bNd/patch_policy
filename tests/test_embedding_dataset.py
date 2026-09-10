"""TrajectoryEmbeddingDataset.__getitem__ must return the same values as
before and must not copy: the training loader calls it once per window."""
import torch
from torch import nn

from datasets.core import TrajectoryDataset, TrajectoryEmbeddingDataset


class _Ragged(TrajectoryDataset):
    """Three episodes of unequal length; obs is already a token grid."""

    def __init__(self, lengths=(7, 5, 9)):
        g = torch.Generator().manual_seed(0)
        self.eps = [
            (
                torch.randn(t, 1, 4, 3, generator=g),  # T V P E
                torch.randn(t, 2, generator=g),  # T A
                torch.ones(t, dtype=torch.bool),  # T
            )
            for t in lengths
        ]

    def get_seq_length(self, idx):
        return self.eps[idx][0].shape[0]

    def get_frames(self, idx, frames):
        return [x[frames] for x in self.eps[idx]]

    def __getitem__(self, idx):
        return self.eps[idx]

    def __len__(self):
        return len(self.eps)


def test_getitem_is_an_unpadded_view():
    ds = TrajectoryEmbeddingDataset(nn.Identity(), _Ragged(), device="cpu")
    for i in range(len(ds)):
        t = ds.get_seq_length(i)
        for got, padded in zip(ds[i], ds.data):
            assert torch.equal(got, padded[i, range(t)])  # what it returned before
            assert got.shape[0] == t  # padding stripped
            # a view of the padded tensor, not a gather-copy of it
            assert got.untyped_storage().data_ptr() == padded.untyped_storage().data_ptr()
