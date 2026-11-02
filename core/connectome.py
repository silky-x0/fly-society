"""Build the FlyWire v783 connectome as a signed float32 CSR matrix."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
NTS = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
INHIBITORY = ["gaba_avg", "glut_avg"]  # Shiu et al. 2024 convention


def load_connectome(raw_dir=RAW):
    """Return (W, root_ids).

    W[post, pre] = sign(pre) * synapse count, summed over neuropils.
    Rows are postsynaptic so synaptic input is W @ spikes.
    sign(pre) = -1 if the synapse-weighted top neurotransmitter is GABA or glutamate, else +1.
    root_ids[i] is the FlyWire root ID of row/column i.
    """
    raw_dir = Path(raw_dir)
    ids = np.load(raw_dir / "proofread_root_ids_783.npy").astype(np.int64)
    c = pd.read_feather(
        raw_dir / "proofread_connections_783.feather",
        columns=["pre_pt_root_id", "post_pt_root_id", "syn_count", *NTS],
    )
    c[NTS] = c[NTS].fillna(0)  # 1,732 rows carry no NT prediction

    nt = c[NTS].mul(c.syn_count, axis=0).groupby(c.pre_pt_root_id).sum()
    sign = pd.Series(np.where(nt.idxmax(axis=1).isin(INHIBITORY), -1, 1), index=nt.index)

    index = pd.Series(np.arange(len(ids), dtype=np.int32), index=ids)
    pre = index.reindex(c.pre_pt_root_id).to_numpy()
    post = index.reindex(c.post_pt_root_id).to_numpy()
    if np.isnan(pre).any() or np.isnan(post).any():
        raise ValueError("connections reference root IDs missing from proofread_root_ids_783.npy")

    w = (c.syn_count.to_numpy() * sign.reindex(c.pre_pt_root_id).to_numpy()).astype(np.float32)
    n = len(ids)
    # COO -> CSR sums the per-neuropil duplicate rows into one edge per neuron pair
    W = sp.coo_matrix((w, (post.astype(np.int32), pre.astype(np.int32))), shape=(n, n)).tocsr()
    return W, ids
