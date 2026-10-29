"""Download the authors' released TIGER weights and convert them to a plain .npz (needs `tensorflow` only to read the checkpoint index; nothing is executed).

    python -m guidefam.scripts.export_official_tiger OUT.npz

By downloading you accept NYGC's academic / non-commercial licence (LICENSE.txt in the Hugging Face space Knowles-Lab/tiger). Do not redistribute the .npz.
"""
import sys
import tempfile

import numpy as np


def main():
    out = sys.argv[1]
    from huggingface_hub import snapshot_download
    d = snapshot_download("Knowles-Lab/tiger", repo_type="space", allow_patterns=["model/*", "LICENSE.txt"], local_dir=tempfile.mkdtemp())
    import tensorflow as tf
    r = tf.train.load_checkpoint(d + "/model/variables/variables")
    g = lambda k: r.get_tensor(k + "/.ATTRIBUTES/VARIABLE_VALUE")
    w = {"conv1_kernel": g("variables/0"), "conv1_bias": g("variables/1"), "conv2_kernel": g("variables/2"), "conv2_bias": g("variables/3"),
         "dense1_kernel": g("layer_with_weights-1/kernel"), "dense1_bias": g("layer_with_weights-1/bias"),
         "dense2_kernel": g("layer_with_weights-2/kernel"), "dense2_bias": g("layer_with_weights-2/bias"),
         "dense3_kernel": g("layer_with_weights-3/kernel"), "dense3_bias": g("layer_with_weights-3/bias")}
    assert w["conv1_kernel"].shape == (4, 4, 2, 32) and w["dense1_kernel"].shape == (1664, 128), "unexpected architecture"
    np.savez(out, **w)
    print("wrote", out, {k: v.shape for k, v in w.items()})


if __name__ == "__main__":
    main()
