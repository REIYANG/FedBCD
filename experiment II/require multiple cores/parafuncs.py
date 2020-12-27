import copy

''' Various computational functions for NN parameter sets '''

def avg_dict(para_set):
    para_copy = copy.deepcopy(para_set)
    N = float(len(para_copy))
    return { k : sum(t[k] for t in para_copy)/N for k in para_copy[0] }

def weighted_dict(para_set, weight):
    para_copy = copy.deepcopy(para_set)
    for k in range(len(para_copy)):
        para_copy[k].update((x, y*weight[k]) for x, y in para_copy[k].items())
    return { k : sum(t[k] for t in para_copy) for k in para_copy[0] }

def sub_dict(primal_set1, primal_set2):
    primal_set1_copy = copy.deepcopy(primal_set1)
    primal_set2_copy = copy.deepcopy(primal_set2)
    return { k: primal_set1_copy[k] - primal_set2_copy.get(k, 0) for k in primal_set1_copy }

def mul_dict(primal_set, ratio):
    primal_set_copy = copy.deepcopy(primal_set)
    return { k: primal_set_copy[k] * ratio for k in primal_set_copy }

def l2_reg_para(primal_set):
    primal_set_copy = copy.deepcopy(primal_set)
    return torch.sum(torch.stack([torch.norm(x)**2 for x in primal_set_copy]))

def sub_para(primal_set1, primal_set2):
    primal_set1_copy = copy.deepcopy(primal_set1)
    primal_set2_copy = copy.deepcopy(primal_set2)
    return [i - j for i, j in zip(primal_set1_copy, primal_set2_copy)]
