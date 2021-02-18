#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:53:11 2021

@author: carson
"""

# environment initialization
import os
import pickle
import matplotlib.pyplot as plt
import phenom
import pandas as pd
import numpy as np
from functools import reduce
from phenom.dataset import DataSet
from phenom.phenotype import Phenotype
from phenom.design import Formula


# Dataset loading
batch1 = DataSet.fromDirectory("/home/carson/Documents", datafile= "data_FASTPHAGEcleaned.csv", metafile= "meta_FASTPHAGEcleaned.csv")


batch1.data.head()

batch1.meta.head()



lines = plt.plot(batch1.data.loc[:, batch1.meta['virus_num'] == 0], c="C0");
lines2 = plt.plot(batch1.data.loc[:, batch1.meta['virus_num'] != 0], c="C1");
plt.legend(lines[:1]+lines2[:1], ["control", "treatment"])
plt.xlabel("time (h)")
plt.ylabel("OD")
plt.show()


# see log transformed data
lines = plt.plot(np.log(batch1.data.loc[:, batch1.meta['virus_num'] == 0]), c="C0");
lines2 = plt.plot(np.log(batch1.data.loc[:, batch1.meta['virus_num'] != 0]), c="C1");
plt.legend(lines[:1]+lines2[:1], ["control", "treatment"])
plt.xlabel("time (h)")
plt.ylabel("OD")
plt.show()

#combine batches if existing
ds = batch1

#log transform OD data
ds.log()

np.unique(ds.meta["host_num"])

#describe Fixed Effects
formula = """1 + C(host_num, Sum, levels=[100000, 500000, 1000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000])
               + C(virus_num, Sum, levels=[0, 50, 500, 5000, 50000, 500000, 5000000, 50000000, 500000000])
               + C(host_num, Sum, levels=[100000, 500000, 1000000, 5000000, 10000000,  20000000, 50000000, 100000000, 200000000, 500000000]):C(virus_num, Sum, levels=[0, 50, 500, 5000, 50000, 500000, 5000000, 50000000, 500000000])"""
fixed_effects = Formula(ds.meta, formula)
# fixed_effects = Formula(ds.meta, "mMPQ")
fixed_effects.frame.head()


mnull = Phenotype(ds.data, fixed_effects)
null_samples = mnull.samples(iter=1200, warmup=1000, control=dict(adapt_delta=.97, max_treedepth=20))
mnull.save("example/Rajmnull")


null_samples = pickle.load(open("example/Rajmnull/samples/posterior_0.pkl", "rb"))

phenom.plot.function.interval(ds.data.index, -2*null_samples['f-native'][:, 1, :])
plt.axhline(0, c="k")
plt.ylabel("log (OD)")
plt.xlabel("time (h)")
plt.show()




#do i want to try a for loop per host cell concentration (from other example on github)

#currently stuck figuring out what the output I got below means. It isn't one per virus


for hostconc in ['H1E5', 'H1E6', 'H1E7', 'H1E8', 'H2E7', 'H2E8', 'H5E5', 'H5E6',
        'H5E7', 'H5E8']:
# for hostconc in ['H1E5']:
    ds = phenom.dataset.DataSet.fromDirectory("../data/Rajnovic/{}/".format(hostconc))
    meta = ds.meta
    print("*"*40)
    print(hostconc)
    fixed_effects = Formula(meta, "C(virus_num, Sum) + 1")
    fixed_effects.frame.head()
    phen = Phenotype(ds.data, fixed_effects, model="phenom_deriv.stan")
    samples = phen.samples()
    print(samples)
    phen.save("../samples/Rajnovic/{}".format(hostconc))
    print()



null_samples_1e5 = pickle.load(open("../samples/Rajnovic/H1E5/samples/posterior_0.pkl", "rb"))
null_samples_5e5 = pickle.load(open("../samples/Rajnovic/H5E5/samples/posterior_0.pkl", "rb"))
null_samples_1e6 = pickle.load(open("../samples/Rajnovic/H1E6/samples/posterior_0.pkl", "rb"))
null_samples_5e6 = pickle.load(open("../samples/Rajnovic/H5E6/samples/posterior_0.pkl", "rb"))
null_samples_1e7 = pickle.load(open("../samples/Rajnovic/H1E7/samples/posterior_0.pkl", "rb"))
null_samples_2e7 = pickle.load(open("../samples/Rajnovic/H2E7/samples/posterior_0.pkl", "rb"))
null_samples_5e7 = pickle.load(open("../samples/Rajnovic/H5E7/samples/posterior_0.pkl", "rb"))
null_samples_1e8 = pickle.load(open("../samples/Rajnovic/H1E8/samples/posterior_0.pkl", "rb"))
null_samples_2e8 = pickle.load(open("../samples/Rajnovic/H2E8/samples/posterior_0.pkl", "rb"))
null_samples_5e8 = pickle.load(open("../samples/Rajnovic/H5E8/samples/posterior_0.pkl", "rb"))


# testasd = [0, 50, 500, 5000, 50000, 500000, 5000000, 50000000, 500000000]
data_index_1e5=pd.read_csv("../samples/Rajnovic/H1E5/data.csv")["time"].tolist()
data_index_5e5=pd.read_csv("../samples/Rajnovic/H5E5/data.csv")["time"].tolist()
data_index_1e6=pd.read_csv("../samples/Rajnovic/H1E6/data.csv")["time"].tolist()
data_index_5e6=pd.read_csv("../samples/Rajnovic/H5E6/data.csv")["time"].tolist()
data_index_1e7=pd.read_csv("../samples/Rajnovic/H1E7/data.csv")["time"].tolist()
data_index_2e7=pd.read_csv("../samples/Rajnovic/H2E7/data.csv")["time"].tolist()
data_index_5e7=pd.read_csv("../samples/Rajnovic/H5E7/data.csv")["time"].tolist()
data_index_1e8=pd.read_csv("../samples/Rajnovic/H1E8/data.csv")["time"].tolist()

phenom.plot.function.interval(data_index_1e5, -(-2*null_samples_1e5['f-native'][:, 1, :]), color="darkred")
phenom.plot.function.interval(data_index_5e5, -(-2*null_samples_5e5['f-native'][:, 1, :]), color="red")
phenom.plot.function.interval(data_index_1e6, -(-2*null_samples_1e6['f-native'][:, 1, :]), color="orange")
phenom.plot.function.interval(data_index_5e6, -(-2*null_samples_5e6['f-native'][:, 1, :]), color="yellow")
phenom.plot.function.interval(data_index_1e7, -(-2*null_samples_1e7['f-native'][:, 1, :]), color="green")
phenom.plot.function.interval(data_index_2e7, -(-2*null_samples_2e7['f-native'][:, 1, :]), color="lightblue")
phenom.plot.function.interval(data_index_5e7, -(-2*null_samples_5e7['f-native'][:, 1, :]), color="blue")
phenom.plot.function.interval(data_index_1e8, -(-2*null_samples_1e8['f-native'][:, 1, :]), color="darkblue")
phenom.plot.function.interval(ds.data.index, -(-2*null_samples_2e8['f-native'][:, 1, :]), color="indigo")
phenom.plot.function.interval(ds.data.index, -(-2*null_samples_5e8['f-native'][:, 1, :]), color="black")
plt.axhline(0, c="k")
plt.ylabel("log (OD)")
plt.xlabel("time (h)")
plt.show()
