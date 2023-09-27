# -*- coding: utf-8 -*-
"""
Sample code to,
- initialise the order planning model from "model" class
- initialise and run the solver from the "solve" class
- validate solution from solver by running a simulation from the "model" class using a different random seed
@author: cstan
"""

import numpy as np
from pymoo.indicators.hv import HV

import model as mop
import solve as planner

#setting up problem parameters
nFact, nCust, nPrdt = 3, 3, 2
tLT = np.ones((nFact, nCust))
tLT[0, 1], tLT[1, 0], tLT[2, 1], tLT[1, 2] = 2, 2, 2, 2
tLT[0, 2], tLT[2, 0] = 3, 3

#maxHr: avialable hours per day for each factory
#pRate: production rate (qty/ hr) for each product per factory (nPrdt x nFact)
#aveD: average demand for each product per customer (nPrdt x nCust)
param = {"nF": nFact, "nC": nCust, "nP": nPrdt,
         "tLT": tLT, "maxHr":np.array([10, 10, 10]),
         "pRate": np.array([[15.6, 5.2, 15.6], [10.4, 10.4, 5.2]]),
         #"aveD": np.array([[60, 20, 30], [40, 40, 10]]),
         "aveD": np.array([[200, 20, 30], [40, 40, 10]]),
         #"devD": np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])}
         "devD": np.array([[0.3, 0.2, 0.2], [0.1, 0.1, 0.1]])}

R = 20 #num of replications
T = 20 #planning horizon length
problem = mop.AllocProblem(param, R)
#x1, y1 = planner.runNSGAII(problem, 50, 20)
#x2, y2 = planner.runOpt(problem, 50, 20)

x3, y3 = planner.runTransferOpt(problem, 50, 20)
#y3Ext = np.empty((x3.shape[0], 2*param["nF"]))
'''
#get the factory level perf for each sol in PS
for i in range(x3.shape[0]):
    alloc, minPHr = problem.decode(x3[i])
    #for j in range():
        #for k in range()
    # run for "r" replications
    aveUnUtilHr, aveLT = {}, {}
    for f in range(param["nF"]): aveUnUtilHr[f], aveLT[f] = 0, 0
    for r in range(R):
        np.random.seed(r)
        for f in problem.factory: f.reset()
        projDemand = problem.simDemand(T)  # sample demand from distribution
        # print()
        completedOrder = problem.simPlan(projDemand, alloc, minPHr, T)  # simulate planning scenarios
        # compute perf
        unUtilHr, lt, _, _, _, _ = problem.computePrefFact(completedOrder)

        for f in range(param["nF"]):
            aveLT[f] += lt[f]
            aveUnUtilHr[f] += unUtilHr[f]

    for f in range(param["nF"]):
        y3Ext[i, f*2] = aveLT[f]/R
        y3Ext[i, (f*2)+1] = aveUnUtilHr[f]/R
'''

'''
#simulate actual scenario using first sol with a different random seed num, i.e. 20
x = x3[-1]
alloc, minPHr = problem.decode(x)
aveUnUtilHr, aveLT = {}, {}
for f in range(param["nF"]): aveUnUtilHr[f], aveLT[f] = 0, 0
#print()
np.random.seed(20)
for f in problem.factory: f.reset()
projDemandPlan = problem.simDemand(50, seed=20)  # sample demand from distribution
#print()
completedOrderPlan = problem.simPlan(projDemandPlan, alloc, minPHr, 50, seed=20)  # simulate actual scenarios
# compute perf
_, _, dailyUnUtilHr, dailyOrderAlloc, dailyOrderFilled, dalilyOrderFillTime = problem.computePrefFact(completedOrderPlan)
'''


import matplotlib.pyplot as plt
#plt.scatter(y1[:,0], y1[:,1])
#plt.scatter(y2[:,0], y2[:,1])
plt.scatter(y3[:,0], y3[:,1])
#plt.legend(["no trasfer-Pymoo", "no trasfer", "with transfer"])
plt.show()


'''
#find ref point
rPt = np.array([max(np.amax(y1[:,0]), np.amax(y2[:,0]), np.amax(y3[:,0])),
                max(np.amax(y1[:,1]), np.amax(y2[:,1]), np.amax(y3[:,1]))])
#compute hv
hv = HV(ref_point=rPt)
noTrf_hv1 = hv(y1)
noTrf_hv2 = hv(y2)
withTrf_hv = hv(y3)
'''
