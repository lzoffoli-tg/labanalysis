"""pytorch module testing"""

import sys
from os.path import abspath, dirname

import pytest
import torch

sys.path.append(dirname(dirname(abspath(__file__))))
import src.labanalysis as laban


@pytest.fixture
def sample_input():
    return {
        "x1": torch.tensor([1.0, 2.0, 3.0]),
        "x2": torch.tensor([4.0, 5.0, 6.0]),
    }


def test_features_generator(sample_input):
    fg = laban.FeaturesGenerator(
        order=2,
        apply_log_transform=True,
        apply_inverse_transform=True,
        include_interactions=True,
    )
    output = fg(sample_input)
    assert isinstance(output, dict)
    assert all(isinstance(v, torch.Tensor) for v in output.values())


def test_boxcox_transform():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bct = laban.BoxCoxTransform(n_features=2)
    y = bct(x)
    x_inv = bct.inverse(y)
    assert y.shape == x.shape
    assert x_inv.shape == x.shape


def test_sigmoid_transformer():
    x = torch.randn(5, 3)
    st = laban.SigmoidTransformer(input_dim=3, output_dim=2, transform_dim=1)
    y = st(x)
    assert y.shape == (5, 2)


def test_pca():
    x = torch.randn(10, 5)
    pca = laban.PCA(input_dim=5, output_dim=3)
    y = pca(x)
    loss = pca.orthogonality_loss()
    assert y.shape == (10, 3)
    assert isinstance(loss, torch.Tensor)


def test_lasso():
    x = torch.randn(8, 4)
    lasso = laban.Lasso(in_features=4, out_features=2)
    y = lasso(x)
    loss = lasso.lasso_loss()
    assert y.shape == (8, 2)
    assert isinstance(loss, torch.Tensor)
