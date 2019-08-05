import numpy as np
import torch

ignore_token = -100

def _topological_loop(theta, batch_sizes):
    new = theta.new

    # batch_sizes is from packed_sequence, i.e. each element is the 'stride' within
    # the sequence data till the next element of the same batch
    B = batch_sizes[0].item()
    # So this, T, is the number of chunks in the packed sequence
    T = len(batch_sizes)
    # Theta itself is the packed sequence, so we have L as the flattened sequence length
    # S is the number of alignable states, note that this will be padded with infs, as the number of target states
    # varies across a batch
    L, S = theta.size()

    # Q stores history of gradients across sequence/batch
    Q = new(L, S+1).fill_(np.float('inf'))
    # V stores the history of potentials across the sequence
    # V[i, j] gives the cost of being in state j at time (i % batch_size) (i.e. this is a packed sequence). We pad with
    # an additional B elements to give an initial state of zero
    # Initialise with inf as some transitions at the start aren't valid - we begin in the upper left corner and not
    # all nodes are reachable from this parent node
    V = new(L + B, S+1).fill_(np.float('inf'))
    # i.e. we force position 0 for each batch (so a slice of elements of size B at the start of B) to be zero
    V[:B, 0] = 0

    left = B
    prev_length = B

    # For each step along sequence
    for t in range(1, T+1):
        # Handling the end case, which is just to put final result into Vt and Qt
        # if t == T:
        #     cur_length = 0
        # else:
        # cur_length means step to the next element in batch/size of chunk
        cur_length = batch_sizes[t-1]
        right = left + cur_length
        prev_left = left - prev_length
        # We cut at prev_right + cur_length - prev_length, to cut off any trailing batches
        # which we are no longer considering.
        prev_cut = right - prev_length
        len_term = prev_length - cur_length

        if cur_length != 0:
            for j in range(1, V.shape[1]):
                # This is the dynamic part, the softmin of the total cost to get to each parent node
                v_prev, Q[left-B:right-B, j] = \
                    torch.cat((V[prev_left:prev_cut, j-1, None],
                               V[prev_left:prev_cut, j, None],
                               V[left:right, j-1, None],
                               ), dim=-1).min(dim=-1)
                # Remember theta is a packed sequence, so this is a cut across batches at a sequence index
                # So the slice of theta is the potential of transitioning from si-1 to si
                V[left:right, j] = (theta[left-B:right-B, j-1] + v_prev)

        left = right
        prev_length = cur_length

    return V, Q


def backtrack(trellis_batch, heights, widths):
    paths = []
    for b in range(trellis_batch.shape[0]):
        trellis = trellis_batch[b]

        path = []
        i = heights[b]-1
        j = widths[b]-1
        while True:
            path = [(i, j)] + path

            if i == 0 and j == 0:
                break

            if j == 0:
                i -= 1
            elif i == 0:
                j -= 1
            else:
                parents = torch.cat((trellis[i-1, max(j-1, 0):j+1], trellis[i, max(j-1, 0)][None]))
                parent_idx = torch.argmin(parents)
                j -= 1 if parent_idx == 0 or parent_idx == 2 else 0
                i -= 1 if parent_idx == 0 or parent_idx == 1 else 0
                i = max(i, 0)
                j = max(j, 0)

        paths.append(path)

    return paths


def get_next_path_step(trellis, i, j):
    if j == 0:
        i -= 1
    elif i == 0:
        j -= 1
    else:
        parents = torch.cat((trellis[i - 1, max(j - 1, 0):j + 1], trellis[i, max(j - 1, 0)][None]))
        parent_idx = torch.argmin(parents)
        j -= 1 if parent_idx == 0 or parent_idx == 2 else 0
        i -= 1 if parent_idx == 0 or parent_idx == 1 else 0
        i = max(i, 0)
        j = max(j, 0)

    return i, j


def teacher_path_difference(trellis_batch, gt_trellis_batch, heights, widths):
    diffs = []
    for b in range(trellis_batch.shape[0]):
        trellis = trellis_batch[b]
        gt_trellis = gt_trellis_batch[b]

        gt_i = i = heights[b]-1
        gt_j = j = widths[b]-1

        error = 0
        path_length = 0

        while True:
            path_length += 1

            if i == 0 and j == 0:
                break

            i, j = get_next_path_step(trellis, i, j)
            gt_i, gt_j = get_next_path_step(gt_trellis, gt_i, gt_j)

            if i != gt_i or j != gt_j:
                error += 1

            i = gt_i
            j = gt_j

        diffs.append(float(error) / path_length)

    return diffs


def backtrack_Q(Q_batch, heights, widths):
    paths = []
    for b in range(Q_batch.shape[0]):
        Q = Q_batch[b]

        path = []
        i = heights[b]-1
        j = widths[b]-1
        while True:
            if i < 0 or j < 0:
                break

            path = [(i, j)] + path

            if j == 0:
                i -= 1
            elif i == 0:
                j -= 1
            else:
                parent_idx = Q[i, j]
                j -= 1 if parent_idx == 0 or parent_idx == 2 else 0
                i -= 1 if parent_idx == 0 or parent_idx == 1 else 0

        paths.append(path)

    return paths

