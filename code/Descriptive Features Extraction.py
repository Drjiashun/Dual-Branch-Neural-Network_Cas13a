import numpy as np
import pandas as pd

data = pd.read_csv('')

guide = data['guide_seq']
target_guide=data['target_at_guide']
context_after = data['target_after']
context_before = data['target_before']
guide_target_hamming_dist=data['guide_target_hamming_dist']
out_logk_measurement=data['out_logk_measurement']


input_feats = []
input_feats.append(out_logk_measurement.values)
input_feats_array = np.array(input_feats).T

input_feats.append(guide_target_hamming_dist.values)
mismatch_counts_4_27 = []
mismatch_a = []
mismatch_g = []
mismatch_c = []
mismatch_t = []

for g, tg in zip(guide, target_guide):
    g = g[4:27]
    tg = tg[4:27]

    mismatches_4_27 = sum(1 for a, b in zip(g, tg) if a != b)
    mismatch_counts_4_27.append(mismatches_4_27)

    mismatch_a_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'A':
            mismatch_a_count += 1
    mismatch_a.append(mismatch_a_count)

    mismatch_g_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'G':
            mismatch_g_count += 1
    mismatch_g.append(mismatch_g_count)

    mismatch_c_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'C':
            mismatch_c_count += 1
    mismatch_c.append(mismatch_c_count)

    mismatch_t_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'T':
            mismatch_t_count += 1
    mismatch_t.append(mismatch_t_count)

input_feats.append(mismatch_counts_4_27)
input_feats.append(mismatch_a)
input_feats.append(mismatch_g)
input_feats.append(mismatch_c)
input_feats.append(mismatch_t)

mismatch_counts_9_23 = []
mismatch_a = []
mismatch_g = []
mismatch_c = []
mismatch_t = []

for g, tg in zip(guide, target_guide):
    g = g[9:23]
    tg = tg[9:23]
    mismatches_9_23 = sum(1 for a, b in zip(g, tg) if a != b)
    mismatch_counts_9_23.append(mismatches_9_23)

    mismatch_a_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'A':
            mismatch_a_count += 1
    mismatch_a.append(mismatch_a_count)

    mismatch_g_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'G':
            mismatch_g_count += 1
    mismatch_g.append(mismatch_g_count)

    mismatch_c_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'C':
            mismatch_c_count += 1
    mismatch_c.append(mismatch_c_count)

    mismatch_t_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'T':
            mismatch_t_count += 1
    mismatch_t.append(mismatch_t_count)

input_feats.append(mismatch_counts_9_23)
# input_feats.append(mismatch_a)
input_feats.append(mismatch_g)
input_feats.append(mismatch_c)
input_feats.append(mismatch_t)

mismatch_counts_14_front = []
mismatch_a = []
mismatch_g = []
mismatch_c = []
mismatch_t = []

for g, tg in zip(guide, target_guide):
    g = g[:15]
    tg = tg[:15]
    mismatches_14_front = sum(1 for a, b in zip(g, tg) if a != b)
    mismatch_counts_14_front.append(mismatches_14_front)

    mismatch_a_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'A':
            mismatch_a_count += 1
    mismatch_a.append(mismatch_a_count)

    mismatch_g_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'G':
            mismatch_g_count += 1
    mismatch_g.append(mismatch_g_count)

    mismatch_c_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'C':
            mismatch_c_count += 1
    mismatch_c.append(mismatch_c_count)

    mismatch_t_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'T':
            mismatch_t_count += 1
    mismatch_t.append(mismatch_t_count)

input_feats.append(mismatch_counts_14_front)
input_feats.append(mismatch_a)

mismatch_counts_14_back = []
mismatch_a = []
mismatch_g = []
mismatch_c = []
mismatch_t = []

for g, tg in zip(guide, target_guide):

    g = g[-14:]
    tg = tg[-14:]

    mismatches_14_back = sum(1 for a, b in zip(g, tg) if a != b)
    mismatch_counts_14_back.append(mismatches_14_back)

    mismatch_a_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'A':
            mismatch_a_count += 1
    mismatch_a.append(mismatch_a_count)

    mismatch_g_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'G':
            mismatch_g_count += 1
    mismatch_g.append(mismatch_g_count)

    mismatch_c_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'C':
            mismatch_c_count += 1
    mismatch_c.append(mismatch_c_count)

    mismatch_t_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'T':
            mismatch_t_count += 1
    mismatch_t.append(mismatch_t_count)

input_feats.append(mismatch_counts_14_back)
input_feats.append(mismatch_a)
input_feats.append(mismatch_g)

mismatch_counts_4_front = []
mismatch_a = []
mismatch_g = []
mismatch_c = []
mismatch_t = []

for g, tg in zip(guide, target_guide):
    g = g[:5]
    tg = tg[:5]

    mismatches_4_front = sum(1 for a, b in zip(g, tg) if a != b)
    mismatch_counts_4_front.append(mismatches_4_front)

    mismatch_a_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'A':
            mismatch_a_count += 1
    mismatch_a.append(mismatch_a_count)

    mismatch_g_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'G':
            mismatch_g_count += 1
    mismatch_g.append(mismatch_g_count)

    mismatch_c_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'C':
            mismatch_c_count += 1
    mismatch_c.append(mismatch_c_count)

    mismatch_t_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'T':
            mismatch_t_count += 1
    mismatch_t.append(mismatch_t_count)

input_feats.append(mismatch_counts_4_front)
input_feats.append(mismatch_a)
input_feats.append(mismatch_g)
input_feats.append(mismatch_c)
input_feats.append(mismatch_t)

mismatch_counts_4_back = []
mismatch_a = []
mismatch_g = []
mismatch_c = []
mismatch_t = []

for g, tg in zip(guide, target_guide):
    g = g[-4:]
    tg = tg[-4:]
    mismatches_4_back = sum(1 for a, b in zip(g, tg) if a != b)
    mismatch_counts_4_back.append(mismatches_4_back)

    mismatch_a_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'A':
            mismatch_a_count += 1
    mismatch_a.append(mismatch_a_count)

    mismatch_g_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'G':
            mismatch_g_count += 1
    mismatch_g.append(mismatch_g_count)

    mismatch_c_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'C':
            mismatch_c_count += 1
    mismatch_c.append(mismatch_c_count)

    mismatch_t_count = 0
    for a, b in zip(g, tg):
        if a != b and a in 'T':
            mismatch_t_count += 1
    mismatch_t.append(mismatch_t_count)

input_feats.append(mismatch_counts_4_back)
input_feats.append(mismatch_a)
input_feats.append(mismatch_g)

mismatch_counts_1_front = []
mismatch_counts_1_back = []
for g, tg in zip(guide, target_guide):
    g_1_front = g[:1]
    tg_1_front = tg[:1]
    mismatches_1_front = sum(1 for a, b in zip(g_1_front, tg_1_front) if a != b)
    mismatch_counts_1_front.append(mismatches_1_front)

    g_1_back = g[-1:]
    tg_1_back = tg[-1:]

    mismatches_1_back = sum(1 for a, b in zip(g_1_back, tg_1_back) if a != b)
    mismatch_counts_1_back.append(mismatches_1_back)

input_feats.append(mismatch_counts_1_front)


exist_mismatch_17_23 = []
for g, tg in zip(guide, target_guide):

    g_17_23 = g[17:23]
    tg_17_23 = tg[17:23]

    mismatches_17_23 = sum(1 for a, b in zip(g_17_23, tg_17_23) if a != b)

    if mismatches_17_23 > 0:
        exist_mismatch_17_23.append(1)
    else:
        exist_mismatch_17_23.append(0)

input_feats.append(exist_mismatch_17_23)


mismatch_counts_14_front = []
mismatch_counts_14_back = []
exist_mismatch_14_front = []
exist_mismatch_14_back = []
for g, tg in zip(guide, target_guide):

    g_14_front = g[:14]
    tg_14_front = tg[:14]

    mismatches_14_front = sum(1 for a, b in zip(g_14_front, tg_14_front) if a != b)
    mismatch_counts_14_front.append(mismatches_14_front)

    g_14_back = g[-14:]
    tg_14_back = tg[-14:]

    mismatches_14_back = sum(1 for a, b in zip(g_14_back, tg_14_back) if a != b)
    mismatch_counts_14_back.append(mismatches_14_back)

    if mismatches_14_front > 0:
        exist_mismatch_14_front.append(1)
    else:
        exist_mismatch_14_front.append(0)

    if mismatches_14_back > 0:
        exist_mismatch_14_back.append(1)
    else:
        exist_mismatch_14_back.append(0)

# input_feats.append(mismatch_counts_14_front)#13
# input_feats.append(mismatch_counts_14_back)
input_feats.append(exist_mismatch_14_front)
input_feats.append(exist_mismatch_14_back)


exist_mismatch_4_27 = []
for g, tg in zip(guide, target_guide):

    g_4_27 = g[4:27]
    tg_4_27 = tg[4:27]

    mismatches_4_27 = sum(1 for a, b in zip(g_4_27, tg_4_27) if a != b)

    if mismatches_4_27 > 3:
        exist_mismatch_4_27.append(1)
    else:
        exist_mismatch_4_27.append(0)

exist_mismatch_3_28 = []
for g, tg in zip(guide, target_guide):

    g_3_28 = g[3:28]
    tg_3_28 = tg[3:28]

    mismatches_3_28 = sum(1 for a, b in zip(g_3_28, tg_3_28) if a != b)


    if mismatches_3_28 > 4:
        exist_mismatch_3_28.append(1)
    else:
        exist_mismatch_3_28.append(0)

input_feats.append(exist_mismatch_4_27)
input_feats.append(exist_mismatch_3_28)


mismatch_a = []
mismatch_g = []
mismatch_c = []
mismatch_t = []

for g, tg in zip(guide, target_guide):

    g_28 = g[:29]
    tg_28 = tg[:29]

    mismatch_a_count = 0
    mismatch_g_count = 0
    mismatch_c_count = 0
    mismatch_t_count = 0


    for a, b in zip(g_28, tg_28):
        if a != b:
            if a == 'A':
                mismatch_a_count += 1
            if a == 'G':
                mismatch_g_count += 1
            if a == 'C':
                mismatch_c_count += 1
            if a == 'T':
                mismatch_t_count += 1

    mismatch_a.append(mismatch_a_count)
    mismatch_g.append(mismatch_g_count)
    mismatch_c.append(mismatch_c_count)
    mismatch_t.append(mismatch_t_count)

input_feats.append(mismatch_a)
input_feats.append(mismatch_g)
input_feats.append(mismatch_c)
input_feats.append(mismatch_t)


mismatch_vectors = []
continuous_match_lengths = []
continuous_mismatch_lengths = []
first_mismatch_positions = []
last_mismatch_positions = []
average_mismatch_distances = []
for g_seq, t_seq in zip(guide, target_guide):
    mismatch_vector = [0] * len(g_seq)
    mismatch_positions = []
    current_match_length = 0
    current_mismatch_length = 0
    match_lengths = []
    mismatch_lengths = []

    for i in range(len(g_seq)):
        if g_seq[i] != t_seq[i]:
            mismatch_vector[i] = 1
            mismatch_positions.append(i)
            if current_match_length > 0:
                match_lengths.append(current_match_length)
                current_match_length = 0
            current_mismatch_length += 1
        else:
            mismatch_vector[i] = 0
            if current_mismatch_length > 0:
                mismatch_lengths.append(current_mismatch_length)
                current_mismatch_length = 0
            current_match_length += 1

    # Append the last segment lengths
    if current_match_length > 0:
        match_lengths.append(current_match_length)
    if current_mismatch_length > 0:
        mismatch_lengths.append(current_mismatch_length)

    mismatch_vectors.append(mismatch_vector)
    continuous_match_lengths.append(match_lengths)
    continuous_mismatch_lengths.append(mismatch_lengths)

    if mismatch_positions:
        first_mismatch_positions.append(mismatch_positions[0])
        last_mismatch_positions.append(mismatch_positions[-1])
        if len(mismatch_positions) > 1:
            avg_distance = np.mean(np.diff(mismatch_positions))
        else:
            avg_distance = 0
    else:
        first_mismatch_positions.append(-1)
        last_mismatch_positions.append(-1)
        avg_distance = 0

    average_mismatch_distances.append(avg_distance)

mismatch_df = pd.DataFrame(mismatch_vectors)

for col in mismatch_df.columns:
    input_feats.append(mismatch_df[col].values)

max_length = max(max(len(x) for x in continuous_match_lengths), max(len(x) for x in continuous_mismatch_lengths))


for lengths in continuous_match_lengths:
    lengths.extend([0] * (max_length - len(lengths)))
for lengths in continuous_mismatch_lengths:
    lengths.extend([0] * (max_length - len(lengths)))

avg_continuous_match_lengths = [np.mean(l) if l else 0 for l in continuous_match_lengths]
avg_continuous_mismatch_lengths = [np.mean(l) if l else 0 for l in continuous_mismatch_lengths]

input_feats.append(avg_continuous_mismatch_lengths)

input_feats.append(first_mismatch_positions)
input_feats.append(last_mismatch_positions)


bases = ('A', 'C', 'G', 'T')

for b in bases:
    input_feats.append(guide.str.count(b))
    input_feats.append(target_guide.str.count(b))
    input_feats.append(context_before.str.count(b))
    input_feats.append(context_after.str.count(b))
    input_feats.append(context_before.str.count(b) + context_after.str.count(b) + target_guide.str.count(b))

input_feats.append(guide.str.count('A' + 'A'))
input_feats.append(guide.str.count('C' + 'C'))
input_feats.append(target_guide.str.count('A' + 'A'))

pfs=context_after.str.slice(0, 1)
# print(pfs)
input_feats.append(pfs.str.count('A'))
input_feats.append(pfs.str.count('G'))
input_feats.append(pfs.str.count('C'))
input_feats.append(pfs.str.count('T'))

pfs1=context_after.str.slice(0, 2)
input_feats.append(pfs1.str.count('A'))
input_feats.append(pfs1.str.count('G'))
input_feats.append(pfs1.str.count('C'))
input_feats.append(pfs1.str.count('T'))

pfs2=context_after.str.slice(1, 2)
input_feats.append(pfs2.str.count('A'))
input_feats.append(pfs2.str.count('G'))
input_feats.append(pfs2.str.count('C'))
input_feats.append(pfs2.str.count('T'))


input_feats_array = np.array(input_feats).T

input_feats_df = pd.DataFrame(input_feats_array)

train_data=input_feats_df

train_data.to_csv('',index=False)

print("Train data shape:", train_data.shape)








