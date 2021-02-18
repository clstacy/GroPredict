#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:35:42 2021

@author: carson
"""

# environment initialization
import os
import pickle
import matplotlib.pyplot as plt
import phenom
from functools import reduce
from phenom.dataset import DataSet
from phenom.phenotype import Phenotype
from phenom.design import Formula



# process raw growth data
# make data

# Dataset loading


batch1 = DataSet.fromDirectory("../data/raw/hsal/20150517_PQ_3/", datafile= "PQ_batch3.csv", metafile= "PQ_batch3_key.csv")
batch2 = DataSet.fromDirectory("../data/raw/hsal/20150607_PQ_4/", datafile= "PQ4.csv", metafile= "PQ_batch4_key.csv")
batch3 = DataSet.fromDirectory("../data/raw/hsal/20150702_PQ_6/", datafile= "PQ_batch6.csv", metafile= "PQ_batch6_key.csv")





#example structure of dataset
batch1.data.head()

batch1.meta.head()

lines = plt.plot(batch1.data.loc[:, batch1.meta['mM PQ'] == 0], c="C0");
lines2 = plt.plot(batch1.data.loc[:, batch1.meta['mM PQ'] != 0], c="C1");
plt.legend(lines[:1]+lines2[:1], ["control", "treatment"])
plt.xlabel("time (h)")
plt.ylabel("OD")
plt.show()

lines = plt.plot(batch2.data.loc[:, batch1.meta['mM PQ'] == 0], c="C0");
lines2 = plt.plot(batch2.data.loc[:, batch1.meta['mM PQ'] != 0], c="C1");
plt.legend(lines[:1]+lines2[:1], ["control", "treatment"])
plt.xlabel("time (h)")
plt.ylabel("OD")
plt.show()

lines = plt.plot(batch3.data.loc[:, batch1.meta['mM PQ'] == 0], c="C0");
lines2 = plt.plot(batch3.data.loc[:, batch1.meta['mM PQ'] != 0], c="C1");
plt.legend(lines[:1]+lines2[:1], ["control", "treatment"])
plt.xlabel("time (h)")
plt.ylabel("OD")
plt.show()

# combine all batches into a single dataset
ds = batch1.concat(batch2).concat(batch3)

ds.log() # log the data

# remove spaces for formula specification
ds.meta['mMPQ'] = ds.meta['mM PQ']



# skip the early timepoints,
# which usually have higher error,
# and subsample the observations
ds.data = ds.data.iloc[5::10, :]


# plot all data after processing
plt.figure(figsize=(12, 4))
for i, batch in enumerate(ds.meta.plate.unique()):
    plt.subplot(1, 3, i+1)

    lines = plt.plot(ds.data.loc[:, (ds.meta.plate==batch) & (ds.meta['mM PQ'] == 0)], c="C0");
    lines2 = plt.plot(ds.data.loc[:, (ds.meta.plate==batch) & (ds.meta['mM PQ'] != 0)], c="C1");

    plt.xlabel("time (h)")
    plt.title(batch)

    if i == 0:
        plt.ylabel("log(OD)")
        plt.legend(lines[:1]+lines2[:1], ["control", "treatment"])

plt.tight_layout()


#describe Fixed Effects
# formula = """1 + C(mMAcid, Sum, levels=[5, 10, 20, 0])
#                + C(pH, Sum, levels=[6.5, 6.0, 5.5, 5, 7])
#                + C(mMAcid, Sum, levels=[5, 10, 20, 0]):C(pH, Sum, levels=[6.5, 6.0, 5.5, 5, 7])"""
fixed_effects = Formula(ds.meta, "C(mMPQ, Sum)")
# fixed_effects = Formula(ds.meta, "mMPQ")
fixed_effects.frame.head()


# ## specify the mixed effects model incorporating batch effects

# # adding formulas concatenates them
# # we add a constant term with a plate specific effect to represent the fixed and random effects in the data
hierarchy = Formula(ds.meta, "1") + Formula(ds.meta, "C(plate) + 0")

mixed_effects = fixed_effects * hierarchy
mixed_effects.frame.head()


fixed_effects.L, mixed_effects.L


mixed_effects.priors





mnull = Phenotype(ds.data, fixed_effects)

mbatch = Phenotype(ds.data, mixed_effects,
                    alpha_priors=[[6, 1], [6, 1], [6, 1], [6, 1]],
                    lengthscale_priors=[[10.0, 10.0], [10.0, 10.0], [7.0, 10.0], [7.0, 10.0]],
                    minExpectedCross=0.1, # min/max expected cross constrain the lengthscale to lie within a certain range
                    maxExpectedCross=3,
                    sigma_prior=[0.02, 20],)

mfull = Phenotype(ds.data, mixed_effects,
                  model="phenom_marginal.stan",
                  alpha_priors=[[6, 1], [6, 1], [6, 1], [6, 1]],
                  lengthscale_priors=[[10.0, 10.0], [10.0, 10.0], [7.0, 10.0], [7.0, 10.0]],
                  minExpectedCross=0.1, # min/max expected cross constrain the lengthscale to lie within a certain range
                  maxExpectedCross=3,
                  sigma_prior=[0.02, 20],)



null_samples = mnull.samples(iter=2000, warmup=1000, control=dict(adapt_delta=.985, max_treedepth=20))

mnull.save("example/mnull")

batch_samples = mbatch.samples(iter=2000, warmup=1000, control=dict(adapt_delta=.985, max_treedepth=20))

mbatch.save("example/mbatch")

full_samples = mfull.samples(iter=1200, warmup=1000, control=dict(adapt_delta=.97, max_treedepth=20))

mfull.save("example/mfull")


null_samples = pickle.load(open("example/mnull/samples/posterior_0.pkl", "rb"))
batch_samples = pickle.load(open("example/mbatch/samples/posterior_0.pkl", "rb"))
full_samples = pickle.load(open("example/mfull/samples/posterior_0.pkl", "rb"))



plt.figure(figsize=(9, 3))

plt.subplot(131)
phenom.plot.function.interval(ds.data.index, -2*null_samples['f-native'][:, 1, :])
plt.axhline(0, c="k")
plt.ylabel("log (OD)")
plt.xlabel("time (h)")

# the treatment fixed effect is at index 4 for these models
plt.subplot(132)
phenom.plot.function.interval(ds.data.index, -2*batch_samples['f-native'][:, 4, :], color="C1")
plt.axhline(0, c="k")
plt.xlabel("time (h)")

plt.subplot(133)
phenom.plot.function.interval(ds.data.index, -2*full_samples['f-native'][:, 4, :], color="C2")
plt.axhline(0, c="k")
plt.xlabel("time (h)")

plt.tight_layout()











# treatment = Formula(ds.meta, "C(Strain) + C(mMPQ) + C(Strain):C(mMPQ)")

# base = Formula(ds.meta, '1')

# batch = Formula(ds.meta, 'C(plate) + 0')

# hierarchy = base + batch

# design = treatment * hierarchy

# mnull = Phenotype(ds.data, treatment, model = "phenom_deriv.stan")

# mbatch = Phenotype(ds.data, design, model = "phenom_deriv.stan",
#                    alpha_priors=[[6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1]],
#                    lengthscale_priors=[[10.0, 10.0], [10.0, 10.0], [7.0, 10.0], [7.0, 10.0],
#                                        [10.0, 10.0], [10.0, 10.0], [7.0, 10.0], [7.0, 10.0]],
#                    minExpectedCross=0.1,
#                    maxExpectedCross=30,
#                    sigma_prior=[0.02, 20],)

#                    # min/max expected cross constrain the lengthscale to lie within a certain range


# mfull = Phenotype(ds.data, design,
#                   model="phenom_marginal.stan",
#                   alpha_priors=[[6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1]],
#                   lengthscale_priors=[[10.0, 10.0], [10.0, 10.0], [7.0, 10.0], [7.0, 10.0],
#                                       [10.0, 10.0], [10.0, 10.0], [7.0, 10.0], [7.0, 10.0]],
#                   minExpectedCross=0.1, # min/max expected cross constrain the lengthscale to lie within a certain range
#                   maxExpectedCross=30,
#                   sigma_prior=[0.02, 20],)



# # null_samples = mnull.samples(iter=1200, warmup=1000, control=dict(adapt_delta=.985, max_treedepth=20))

# # mnull.save("example/mnull")

# batch_samples = mbatch.samples(iter=1200, warmup=1000, control=dict(adapt_delta=.985, max_treedepth=20))

# mbatch.save("example/mbatch")

# full_samples = mfull.samples(iter=1200, warmup=1000, control=dict(adapt_delta=.985, max_treedepth=20))

# mfull.save("example/mfull")


