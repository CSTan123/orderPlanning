# -*- coding: utf-8 -*-
"""
Functions for the order planning system to call
input:
1) param - problem parameters, refer below for its format
optional input:
1) R - number of replications;
2) T: planning horizon;
3) factPrefReq: dictates if factory level performace is required
4) allocRange: if true converts the solution to a min and max range

output:
1a) sol, 20 by (num of fact * num of cust + number of fact)
1b) sol, 20 by (2*num of fact * num of cust + number of fact)
2) scPerf, 20 by 2 (average fulfillment Time, average UnUtilised Hr) - SC lvl performance
3) factPerf, 20 by (2 * num fact) - fact lvl performance
4) unUtilCapPref, 20 by 1 - preference value from 0 to 1 on the unutilised capacity
5) perfCat, 20 by 1 - categorises the solution into the following 4 categories
- 0, Short Order Fulfilment Time with Higher Unutilized Production Capacity
- 1, Mid Order Fulfilment Time with Slightly Higher Unutilized Production Capacity
- 2, Mid Unutilized Production Capacity with Slightly Longer Order Fulfilment Time
- 3, Low Unutilized Production Capacity with Longer Order Fulfilment Time
"""

import numpy as np
import model as mop
import solve as planner
#
def plan(param, R=20, T=20, factPerfReq=True, allocRange=True):
    #initialise the oreder problem
    problem = mop.AllocProblem(param, R)
    x, scPerf = planner.runTransferOpt(problem, 50, 20) #gen, pop
    #convert the allocation percentage
    sol = x.copy()
    for i in range(x.shape[0]):
        for c in range(param["nC"]):
            sol[i, param["nF"]*c : param["nF"]*c + param["nF"]] = \
                x[i, param["nF"]*c : param["nF"]*c + param["nF"]]\
                / x[i, param["nF"]*c : param["nF"]*c + param["nF"]].sum()

    factPerf = np.empty((x.shape[0], 2*param["nF"]))
    if (factPerfReq):
        for i in range(x.shape[0]):
            alloc, minPHr = problem.decode(x[i])
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
                unUtilHr, lt, _, _, _, _ = problem.computePrefFact(completedOrder, T)

                for f in range(param["nF"]):
                    aveLT[f] += lt[f]
                    aveUnUtilHr[f] += unUtilHr[f]

            for f in range(param["nF"]):
                factPerf[i, f * 2] = aveLT[f] / R
                factPerf[i, (f * 2) + 1] = aveUnUtilHr[f] / R
    if (allocRange):
        sol_range = np.empty((x.shape[0], 2*param["nF"]*param["nC"]+param["nF"]))
        for c in range(param["nC"]):
            for f in range(param["nF"]):
                idx_range = 2 * (c*param["nF"] + f)
                idx_sol = c*param["nF"] + f
                #sol_range[:, idx] - lower bound
                #sol_range[:, idx+1] - upper bound
                if f == 0:
                    sol_range[:, idx_range] = 0
                    sol_range[:, idx_range+1] = sol[:, idx_sol]
                else:
                    sol_range[:, idx_range] = sol_range[:, idx_range-1]
                    sol_range[:, idx_range + 1] = sol[:, idx_sol] + sol_range[:, idx_range-1]
        sol_range[:, 2*param["nF"]*param["nC"]: 2*param["nF"]*param["nC"]+param["nF"]] = sol[:, param["nF"]*param["nC"]:]
        sol = sol_range.copy()

        unutilCapPref, perfCat = np.linspace(0, 1, num=x.shape[0]), np.empty(x.shape[0])
        sortIdx = np.argsort(np.argsort(-scPerf[:,1]))
        unutilCapPref = unutilCapPref[sortIdx]

        binSize = 4/ (x.shape[0]-1) #4 categories
        for i in range(x.shape[0]):
            if unutilCapPref[i] <= binSize: perfCat[i] = 0 # Short Order Fulfilment Time with Higher Unutilized Production Capacity
            elif unutilCapPref[i] <= 2*binSize: perfCat[i] = 1 # Mid Order Fulfilment Time with Slightly Higher Unutilized Production Capacity
            elif unutilCapPref[i] <= 3*binSize: perfCat[i] = 2 # Mid Unutilized Production Capacity with Slightly Longer Order Fulfilment Time
            else: perfCat[i] = 3 # Low Unutilized Production Capacity with Longer Order Fulfilment Time

    return sol, scPerf, factPerf, unutilCapPref, perfCat
'''
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
         "aveD": np.array([[60, 20, 30], [40, 40, 10]]),
         #"devD": np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])}
         "devD": np.array([[0.3, 0.2, 0.2], [0.1, 0.1, 0.1]])}

sol, scPerf, factPerf, unutilCapPref, perfCat = plan(param)
'''