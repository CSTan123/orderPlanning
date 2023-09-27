# -*- coding: utf-8 -*-
"""
Helper Functions to get additional info. to display in order planning app
def genOrder - generate demand for T time units with mock-up customer names, can be used to generate actual demand
input:
1) aveD - average demand for each product per customer np(nPrdt x nCust)
2) devD - standard deviation in % of aveD for each product per customer np(nPrdt x nCust)
3) T - number of time units for demand projection
output:
1) projDemand - customer demand for each product per customer over a period T np(nPrdt x nCust x T)
2) orderRecords - customer names for each customer site over a period T dict{custID: [name1, name2, ...], ...}

def simActualPerf - evaluate the acutal performance of a selected solution computed by the orderPlanner class
input:
1) param - problem parameters
2) sol - a solution/ plan computed by the orderPlanner class, same format
3) T - number of time units for demand projection
output:
1) dailyUnUtilHr - untilized capacity for each factory per day dict{fac 0: [0.89, ...],...}
2) dailyOrderAlloc - number of orders allocated to each factory per day dict{fac 0: [0, 1, ...],...}
3) dailyOrderFilled - number of orders filled by each fact per day dict{fac 0: [0, 2, ...],...}
4) dalilyOrderFillTime - order fulfillment time for each fact per day (non-cumulative) dict{fac 0: [0, 2, ...],...}

def triggerReplan - compare the average demand of past and new demand to determine if replanning is required
input:
1) pastAveDemand - past average demand for each product per customer np(nPrdt x nCust)
2) newAveDemand - new average demand for each product per customer np(nPrdt x nCust)
output:
1) replan - true if replanning is required, otherwise false

def reAllocateOrder - reallocate past orders that have been allocated but yet to produce based on new plan
input:
1) orders - past orders that have been allocated but yet to produce dict{orderId: custID}
2) plan - newly selected solution/ plan computed by the orderPlanner class, same format
3) nF - number of factory
4) nC - number of customer
output:
1) allocOrders - factory that the orders have been reallocated to dict{orderID: factID}

"""

import numpy as np
import model as mop

def genOrder(aveD, devD, T, seed=20):
    np.random.seed(seed)
    nPrdt, nCust = aveD.shape[0], aveD.shape[1]
    # generate demand projection trajectory
    projDemand = np.zeros((nPrdt, nCust, T))
    for p in range(nPrdt):
        for c in range(nCust):
            mu, sigma = aveD[p][c], aveD[p][c] * devD[p][c]
            projDemand[p][c] = np.random.normal(mu, sigma, T)
    projDemand[projDemand < 0] = 0  # convert negative value to zero
    projDemand = (np.rint(projDemand)).astype(int)  # round demand to nearest integer

    #creates cust name in following dict format: {custID, [cust name]} where list [cust name] is time-indexed
    orderRecords = {}
    custName = ["Benjamin Tan", "Amanda Lim", "Desmond Wong", "Chloe Ng", "Joshua Lee", "Jasmine Teo", "Isaac Goh", "Sophia Lim",
                "Alexander Tan", "Emily Choo", "Lucas Koh", "Isabella Ng", "Ethan Ng", "Samantha Tan", "Nicholas Lee", "Grace Lim",
                "Aaron Tan", "Victoria Tan", "Evan Lim", "Alyssa Wong", "Caleb Ong", "Natalie Lim", "Nathan Lim", "Ava Chong",
                "Ethan Seah", "Hannah Yeo", "Gabriel Tan", "Sarah Ng", "Ryan Tan", "Audrey Lim", "Daniel Ng", "Kayla Tan", "Benjamin Goh",
                "Ashley Tan", "Matthew Lim", "Isabelle Tan", "Jonathan Lee", "Elyse Lim", "Dylan Tan", "Hannah Goh", "Alexander Goh",
                "Chloe Lee", "Joshua Lim", "Emma Tan", "Lucas Tan", "Jessica Lim", "Samuel Tan", "Amelia Ng", "Ethan Lim", "Lauren Tan",
                "Nathan Goh", "Mia Lim", "Ryan Goh", "Grace Tan", "Aaron Lim", "Rachel Ng", "Isaac Tan", "Sophia Goh", "Adam Lim", "Olivia Tan",
                "Caleb Ng", "Emily Tan", "Lucas Goh", "Charlotte Lim", "Jayden Tan", "Zoey Lim", "Ethan Ong", "Audrey Goh", "Daniel Tan", "Hannah Lim",
                "Nicholas Tan", "Isabella Tan", "Dylan Goh", "Alyssa Lim", "Benjamin Ong", "Victoria Goh", "Evan Tan", "Natalie Tan", "Nathan Ong",
                "Ava Lim", "Ryan Lee", "Amelia Tan", "Ethan Lee", "Chloe Ong", "Joshua Tan", "Emma Goh", "Samuel Lim", "Sarah Tan", "Gabriel Goh",
                "Ashley Goh", "Matthew Tan", "Isabelle Lim", "Jonathan Goh", "Elyse Goh", "Dylan Lee", "Hannah Ong", "Alexander Lee",
                "Kayla Goh", "Benjamin Lee", "Olivia Goh", "Caleb Tan", "Emily Goh", "Lucas Lee", "Jessica Goh", "Samuel Goh", "Amelia Lee",
                "Ethan Goh", "Lauren Goh", "Nathan Lee", "Mia Goh", "Ryan Lee", "Grace Goh", "Aaron Lee", "Rachel Goh", "Isaac Lee", "Sophia Lee",
                "Adam Goh", "Charlotte Goh", "Jayden Goh", "Zoey Goh", "Ethan Lee", "Audrey Lee", "Daniel Goh", "Hannah Lee", "Nicholas Goh",
                "Isabella Lee", "Dylan Lee", "Alyssa Lee", "Benjamin Lee", "Victoria Lee", "Evan Lee", "Natalie Goh", "Nathan Lee", "Ava Lee",
                "Ryan Lee", "Amelia Lee", "Ethan Lee", "Chloe Lee", "Joshua Lee", "Emma Lee", "Samuel Lee", "Sarah Lee", "Gabriel Lee", "Ashley Lee",
                "Matthew Lee", "Isabelle Lee", "Jonathan Lee", "Elyse Lee", "Dylan Lee", "Hannah Lee"]

    nameIndex = np.random.randint(0, high=len(custName), size=nCust*T, dtype=int)
    for c in range(nCust):
        nameList = []
        for t in range(T):
            idx = nameIndex[(c*T) + t]
            nameList.append(custName[idx])
        orderRecords[c] = nameList

    return projDemand, orderRecords

def simActualPerf(param, sol, T, seed=20):
    problem = mop.AllocProblem(param, 20)
    nF, nC = param["nF"], param["nC"]
    minPHr = sol[sol.size - nF: sol.size] * param["maxHr"]
    alloc = np.empty((nF, nC))
    for c in range(nC):
        for f in range(nF):
            alloc[f, c] = sol[((c*nC)+f+1)*2 -1]

    projDemand, _ = genOrder(param["aveD"], param["devD"], T, seed=seed)  # sample demand from distribution
    # print()
    completedOrderPlan = problem.simPlan(projDemand, alloc, minPHr, T, seed=seed)  # simulate actual scenarios
    # compute perf
    _, _, dailyUnUtilHr, dailyOrderAlloc, dailyOrderFilled, dalilyOrderFillTime = problem.computePrefFact(completedOrderPlan, T)

    return dailyUnUtilHr, dailyOrderAlloc, dailyOrderFilled, dalilyOrderFillTime

def triggerReplan(pastAveDemand, newAveDemand, thres=0.1):
    replan = False
    nPrdt, nCust = pastAveDemand.shape[0], pastAveDemand.shape[1]
    for p in range(nPrdt):
        totchange = 0
        for c in range(nCust):
            totchange += abs(pastAveDemand[p,c] - newAveDemand[p, c])
        if totchange/ pastAveDemand[p].sum() > thres:
            replan = True
            break

    return replan

def reAllocateOrder(orders, plan, nF, nC):
    allocOrders = {}
    alloc = np.empty((nF, nC))
    for c in range(nC):
        for f in range(nF):
            alloc[f, c] = plan[((c * nC) + f + 1) * 2 - 1]
    for o in orders:
        rand = np.random.rand()
        c = orders[o]
        for f in range(nF):
            if alloc[f, c] >= rand: allocOrders[o] = f
    return allocOrders

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

#sol, scPerf, factPerf, unutilCapPref, perfCat = plan(param)
#sol1 = np.array([0, 0.955056272, 0.955056272, 1, 1, 1, 0, 0.042543865, 0.042543865, 1, 1, 1, 0, 0.034535346, 0.034535346, 0.034535346, 0.034535346, 1, 0.742271913, 0.605076259,	0.694233332])
sol2 = np.array([0,	0.938800396, 0.938800396, 0.957529236, 0.957529236, 1, 0, 0.226743328, 0.226743328, 0.862384365, 0.862384365, 1, 0, 0.020903211, 0.020903211, 0.181724192, 0.181724192, 1, 0.81782308, 0.918792121, 0.700181916])
problem, projDemand, completedOrderPlan, dailyUnUtilHr, dailyOrderAlloc, dailyOrderFilled, dalilyOrderFillTime = simActualPerf(param, sol2, 50, seed=20)
'''

'''
aveD = np.array([[60, 20, 30], [40, 40, 10]])
devD = np.array([[0.3, 0.2, 0.2], [0.1, 0.1, 0.1]])
projDemand, orderRecords = genOrder(aveD, devD, 50, seed=20)
'''