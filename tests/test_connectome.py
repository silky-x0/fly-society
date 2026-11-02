import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from core.connectome import RAW, load_connectome

SHIU = Path(os.environ.get("SHIU_DIR", Path.home() / "Downloads" / "Drosophila_brain_model"))

pytestmark = pytest.mark.skipif(
    not (RAW / "proofread_connections_783.feather").exists(), reason="FlyWire data not downloaded"
)


@pytest.fixture(scope="module")
def connectome():
    return load_connectome()


def test_shape_and_dtypes(connectome):
    W, ids = connectome
    assert W.shape == (139_255, 139_255) == (len(ids),) * 2
    assert W.nnz == 15_091_983
    assert W.dtype == np.float32
    assert W.indices.dtype == np.int32
    assert np.abs(W.data).sum(dtype=np.float64) == 54_492_922  # float32 sum loses precision past 2**24


@pytest.mark.skipif(not (SHIU / "Connectivity_783.parquet").exists(), reason="Shiu repo not cloned")
def test_matches_shiu(connectome):
    W, ids = connectome
    s = pd.read_parquet(SHIU / "Connectivity_783.parquet")
    index = pd.Series(np.arange(len(ids)), index=ids)
    post = index.reindex(s.Postsynaptic_ID).to_numpy()
    pre = index.reindex(s.Presynaptic_ID).to_numpy()

    counts = sp.csr_matrix((s.Connectivity.to_numpy(np.float32), (post, pre)), shape=W.shape)
    assert (abs(W) != counts).nnz == 0  # identical edges and synapse counts

    signed = sp.csr_matrix((s["Excitatory x Connectivity"].to_numpy(np.float32), (post, pre)), shape=W.shape)
    agree = (W.multiply(signed) > 0).sum() / W.nnz
    assert agree > 0.95  # sign rules differ: Shiu use a per-neuron NT table; we use synapse-weighted argmax
