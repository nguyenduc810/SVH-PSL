
# This file computes dominance relation between two evaluates solutions
# ind1 dominates ind2, output 1
# ind2 dominates ind1, output 2
# otherwise, ouput 0


# judge the Pareto dominance relation between two evaluated solutions
def pareto_dominance(ind1, ind2):
    r = get_pareto_dom_rel(ind1, ind2)
    return r




def scalar_dominance(ind1, ind2):
    if ind1.fitness.valid and ind2.fitness.valid:
        if ind1.cluster_id != ind2.cluster_id:
            return 0
        else:
            if ind1.scalar_dist < ind2.scalar_dist:
                return 1
            else:
                return 2
    else:
        raise TypeError("Scalar dominance comparison cannot be done "
                        "when either of two individuals has not been evaluated")


def get_pareto_dom_rel(values1, values2):
    n1, n2 = 0, 0
    for v1, v2 in zip(values1, values2):
        if v1 < v2:
            n1 += 1
        elif v2 < v1:
            n2 += 1

        if n1 > 0 and n2 > 0:
            return 0

    if n2 == 0 and n1 > 0:
        return 2
    elif n1 == 0 and n2 > 0:
        return 0
    else:
        return 1


def get_inverted_dom_rel(r):
    return r if r == 0 else 3 - r


def access_dom_rel(i, j, x, y, rel_map, dom):
    if rel_map[i, j] != -1:
        return rel_map[i, j]
    else:
        r = dom(y[i,:], y[j,:])
        rel_map[i, j] = r
        rel_map[j, i] = get_inverted_dom_rel(r)
        return r
