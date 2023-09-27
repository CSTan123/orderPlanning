# -*- coding: utf-8 -*-
"""
Order planning optimization problem and defines
- its problem parameters & decision variables
- objective functions
- experiment, for performing multiple simulations to compute the performance/ objective function
@author: cstan
"""
#external lib
import numpy as np
from pymoo.core.problem import Problem
#internal lib
import dataObject as obj

class AllocProblem(Problem):
    def __init__(self, param, R):
        self.R = R  # number of replications for the projected allocation/ demand
        self.p = param
        # allocation for all orders and min. production qty for all timestep
        self.nVar = self.p["nF"]*self.p["nC"] + self.p["nF"]
        self.currDemand = None  # nC by 1

        self.factory = None  # factory list
        self.initiFactory()

        super().__init__(n_var=self.nVar, n_obj=2, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        aveLT, aveUnUtil = np.zeros(x.shape[0]), np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            aveLT[i], aveUnUtil[i] = self.experiment(x[i])

        out["F"] = np.column_stack([aveLT, aveUnUtil])

    def initiFactory(self):
        self.factory = []
        for f in range(self.p["nF"]):
            self.factory.append(obj.Factory(
                f, self.p["maxHr"][f], self.p["tLT"][f]
            ))

    # last term of currX & projX is the minPQty
    def experiment(self, x, T=20):
        # decision variables/ allocation & production plan
        alloc, minPHr = self.decode(x)
        #print("Min production hr", minPHr)
        # run for "r" replications
        aveLT, aveUnUtilHr = 0, 0

        for r in range(self.R):
            # np.random.seed(r) #fixed random seed based on replication number
            np.random.seed(r)
            #print("Replication", r)
            # reset problem settings
            for f in self.factory: f.reset()
            projDemand = self.simDemand(T, seed=r) #sample demand from distribution
            #print()
            completedOrder = self.simPlan(projDemand, alloc, minPHr, T, seed=r) #simulate planning scenarios
            #compute perf
            lt, unUtilHr = self.computePref(completedOrder, T)
            aveLT += lt
            aveUnUtilHr += unUtilHr

        aveLT /= self.R
        aveUnUtilHr /= self.R

        return aveLT, aveUnUtilHr

    def decode(self, x):
        alloc = np.zeros((self.p["nF"], self.p["nC"]))  # values, nF by nC
        # determine demand allocation percentage
        for c in range(self.p["nC"]):
            # curr demand allocation
            alloc[:, c] = x[c * self.p["nF"]: (c + 1) * self.p["nF"]]
            alloc[:, c] = alloc[:, c] / alloc[:, c].sum()  # normalise it
            # compute cummulative prob
            for f in range(1, self.p["nF"]): alloc[f, c] += alloc[f - 1, c]
            # print("Allocation for C", c, ":", alloc[:, c])
        minPHr = x[self.p["nF"] * self.p["nC"]:] * self.p["maxHr"]

        return alloc, minPHr

    def simDemand(self, T, seed=20):
        np.random.seed(seed)
        # generate demand projection trajectory
        projDemand = np.zeros((self.p["nP"], self.p["nC"], T))
        for p in range(self.p["nP"]):
            for c in range(self.p["nC"]):
                mu, sigma = self.p["aveD"][p][c], self.p["aveD"][p][c] * self.p["devD"][p][c]
                projDemand[p][c] = np.random.normal(mu, sigma, T)
        projDemand[projDemand < 0] = 0  # convert negative value to zero
        projDemand = (np.rint(projDemand)).astype(int)  # round demand to nearest integer
        # print("Demand Generated", projDemand)

        return projDemand

    def simPlan(self, projDemand, alloc, minPHr, T, seed=20):
        np.random.seed(seed)
        completedOrder = []
        for f in self.factory: f.reset()
        # print("======Simulation for proj demand======")
        # allocation of proj demand
        for t in range(0, T):
            #print("Time", t)
            for f in range(self.p["nF"]): self.factory[f].dailyOrderAlloc.append(0)
            for c in range(self.p["nC"]):
                rand = np.random.rand()
                for f in range(self.p["nF"]):
                    if alloc[f, c] >= rand:
                        self.factory[f].dailyOrderAlloc[t] += 1 #count num of orders allocated to each fact
                        # convert order to production hours requried
                        reqHr = (projDemand[:, c, t] / self.p["pRate"][:, f]).sum()
                        # allocate order
                        self.factory[f].activeOrder.append(obj.Order(c, t, reqHr, f))
                        #print("Rand Num", round(rand, 2), "; C", c, "with demand:", projDemand[:, c, t],
                        #"allocated to F", f, "requiring", round(reqHr,2), "production hours")
                        break
            # fulfilment of current demand
            #print()
            for f in range(self.p["nF"]):
                completedOrder.extend(self.factory[f].produce(t, minPHr[f]))
                # print()
            # print(len(completedOrder), "order have been completed")
            # print()
        return completedOrder

    def computePref(self, completedOrder, T):
        aveLT, unUtilHr = 0, 0
        # obj 1: average fulfilment lead time
        aveLT = T
        if len(completedOrder)> 0:
            aveLT = sum(o.fulfilmentTime for o in completedOrder)/ len(completedOrder)
        # obj 2: average unutilized cap %
        unUtilHr = 0
        if sum(f.totAvailHr for f in self.factory) > 0:
            unUtilHr = sum(f.unUtilHr for f in self.factory)/ sum(f.totAvailHr for f in self.factory)

        return aveLT, unUtilHr

    def computePrefFact(self, completedOrder, T):
        unUtilHr, aveLT = {}, {}
        dailyUnUtilHr, dailyOrderAlloc, dailyOrderFilled, dailyOrderFillTime = {}, {}, {}, {}
        completedOrderFact = {}
        for f in range(self.p["nF"]):
            unUtilHr[f] = self.factory[f].unUtilHr/ self.factory[f].totAvailHr

            completedOrderFact[f] = []
            for o in completedOrder:
                if o.fact == f: completedOrderFact[f].append(o)

            aveLT[f] = T
            if len(completedOrderFact[f]) > 0:
                aveLT[f] = sum(o.fulfilmentTime for o in completedOrderFact[f])/ len(completedOrderFact[f])

            dailyUnUtilHr[f], dailyOrderAlloc[f], dailyOrderFilled[f], dailyOrderFillTime[f] = [], [], [], []

            dailyUnUtilHr[f].extend(self.factory[f].dailyUnUtilHr)
            dailyOrderAlloc[f].extend(self.factory[f].dailyOrderAlloc)
            dailyOrderFilled[f].extend(self.factory[f].dailyOrderFilled)
            dailyOrderFillTime[f].extend(self.factory[f].dailyFillTime)

        return unUtilHr, aveLT, dailyUnUtilHr, dailyOrderAlloc, dailyOrderFilled, dailyOrderFillTime