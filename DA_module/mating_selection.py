import random
import math
import numpy as np
from deap import base
from deap import tools
# This file implements three alternative variation methods to produce offsprings

toolbox = base.Toolbox()
# use genetic operations
def random_genetic_variation(population, n, low, up,n_dim,cxpb=1.0, mutpb=1.0 ):
    offsprings = []
    pop_size = len(population)
    k = math.ceil(n / 2)

    for _ in range(k):
        i = random.randrange(pop_size)
        j = get_another_randint(i, pop_size)
        ind1 = toolbox.clone(population[i])
        ind2 = toolbox.clone(population[j])
        if random.random() < cxpb:
            ind1, ind2 = tools.cxSimulatedBinaryBounded(ind1, ind2,low=low,up=up,eta=30.0)


        if random.random() < mutpb:
            ind1, = tools.mutPolynomialBounded(ind1,low=low,up=up,eta=20.0,indpb=1.0 / n_dim)


        if random.random() < mutpb:
            ind2, = tools.mutPolynomialBounded(ind1,low=low,up=up,eta=20.0,indpb=1.0 / n_dim)


        offsprings.append(ind1)
        offsprings.append(ind2)

    return offsprings[0:n]


# use differential evolution (DE) operations
def random_de_variation(population, n, low, up, mf=0.5, cr=1.0):
    pop_size = len(population)
    offsprings = []

    for i in range(n):
        j = random.randrange(pop_size)

        xr0 = toolbox.clone(population[j])
        del xr0.fitness.values

        indexes = [idx for idx in range(pop_size) if idx != j]
        i1, i2 = np.random.choice(indexes, 2, replace=False)
        xr1 = population[i1]
        xr2 = population[i2]

        dim = len(xr0)
        ird = random.randrange(dim)

        for k in range(dim):
            if k == ird or random.random() < cr:
                xr0[k] = xr0[k] + mf * (xr1[k] - xr2[k])
                if xr0[k] < low[k] or xr0[k] > up[k]:
                    xr0[k] = random.uniform(low[k], up[k])

        xr0, = tools.mutPolynomialBounded(xr0)
        offsprings.append(xr0)

    return offsprings


# use both genetic and DE operations, each produces n / 2
def random_hybrid_variation(population, n, low, up, cxpb=1.0, mutpb=1.0, mf=0.5, cr=1.0):
    n_ = n // 2
    g_offsprings = random_genetic_variation(population, n_, cxpb, mutpb)
    d_offsprings = random_de_variation(population, n - n_,  low, up, mf, cr)

    return g_offsprings + d_offsprings


def get_another_randint(i, size):
    j = random.randrange(size)
    while j == i:
        j = random.randrange(size)
    return j



