
import numpy as np
import pandas as pd

data = pd.read_csv('')

guide = data['guide_seq']
target_guide = data['target_at_guide']
context_after = data['target_after']
context_before = data['target_before']
guide_target_hamming_dist = data['guide_target_hamming_dist']
out_logk_measurement = data['out_logk_measurement']

onehot_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
onehot_order = ('A', 'C', 'G', 'T')


def onehot(b):
    assert b in onehot_idx.keys()
    v = [0, 0, 0, 0]
    v[onehot_idx[b]] = 1
    return v

def encode_context_before(context_before):
    input_feats_context_before = []
    for seq in context_before:
        # print(seq)
        encoded_seq = []
        for pos in range(len(seq)):
            v = onehot(seq[pos])
            v += [0, 0, 0, 0]
            encoded_seq.append(v)
        input_feats_context_before.append(encoded_seq)
    return input_feats_context_before

context_before_onehot = encode_context_before(context_before)

context_after_onehot = encode_context_before(context_after)

def encode_guide(target,guide):
    input_feats_guide = []
    for t_seq, g_seq in zip(target, guide):
        # print(t_seq,g_seq)
        encoded_seq = []
        for pos in range(len(t_seq)):
            v_target = onehot(t_seq[pos])
            v_guide = onehot(g_seq[pos])
            v = v_target + v_guide
            encoded_seq.append(v)
        input_feats_guide.append(encoded_seq)
    return input_feats_guide

context_guide_onehot = encode_guide(target_guide,guide)

input_feats = np.concatenate([context_before_onehot, context_guide_onehot, context_after_onehot], axis=1)

print("Shape of input_feats:", input_feats.shape)


print(input_feats[:1])



import pickle
import os

labels = data.iloc[:, 8].values

save_dir = ""

data_dict = {'embeddings': input_feats, 'labels': labels}

save_path = os.path.join(save_dir, '')

with open(save_path, 'wb') as f:
    pickle.dump(data_dict, f)

print("Data saved to:", save_path)