# -*- coding: utf-8 -*-
"""
Algorithm for transfer optimization based on NSGA2 and, -
1) Lim, R., Gupta, A., Ong, Y.S., Feng, L. and Zhang, A.N., 2021. Non-linear domain adaptation in transfer evolutionary optimization. Cognitive Computation, 13, pp.290-307.
2) Da, B., Gupta, A. and Ong, Y.S., 2018. Curbing negative influences online for seamless transfer evolutionary optimization. IEEE transactions on cybernetics, 49(12), pp.4365-4378.

@author: cstan
"""

import numpy as np
import random
from MOEA_operators import SBX_crossover, polynomial_mutation, binary_tournament, crowding_distance, \
    sort_distance, fast_non_dominated_sort, check_bounds

class trNSGA2():
    def __init__(self, problem, max_gen, pop_size, nVar, mixture_model=None, tr_int=2, seed=1):
        random.seed(seed)
        self.problem = problem
        
        self.gen_no = 0
        self.sol = []
        self.obj1, self.obj2 = [], []
    
        self.mixture_model = mixture_model
        self.pop_mean = []
        self.pop_var = []
        
        self.run(max_gen, pop_size, nVar, tr_int)
        
    def run(self, max_gen, pop_size, nVar, tr_int):
        self.gen_no = 0
        #initial random solution
        self.sol = [[random.random() for _ in range(nVar)] for _ in range(0, pop_size)]
        #initial random solution with 1 optimal sol here
        #self.sol = [[random.random() for _ in range(nVar)] for _ in range(0, pop_size-1)]
        #opt1 = [0.955514, 0, 0, 0, 0.80494, 0, 0, 0, 0.865341, 0.921572, 0.837982, 0.727248]        
        #self.sol.append(opt1)
        self.obj1, self.obj2 = [], []
        for i in range(pop_size):
            obj = self.problem.evaluate(np.array(self.sol[i]))
            self.obj1.append(obj[0]) 
            self.obj2.append(obj[1])
        
        curr_sol = np.array(self.sol)
        self.pop_mean.append(np.mean(curr_sol, axis=0))
        self.pop_var.append(np.diag(np.cov(curr_sol.T)))
        while (self.gen_no < max_gen):
            non_dominated_sorted_solution = fast_non_dominated_sort(self.obj1[:], self.obj2[:])
            #print("AMTEA Output for Generation ", self.gen_no, " :")
            parent_front_f11 = []
            parent_front_f22 = []
            non_dominated_sorted_solution[0].sort()
            for index in non_dominated_sorted_solution[0]:
                parent_front_f11.append(self.obj1[index])
                parent_front_f22.append(self.obj2[index])
                
            # Generating offsprings
            solution2 = self.sol[:]
            while (len(solution2) < 2 * pop_size):
                # offspring generated by sampling the target probabilistic mixture model at specified transfer intervals
                if (tr_int is not None) and (self.gen_no + 1) % tr_int == 0:
                    self.mixture_model.update(np.array(self.sol))
                    offspring_A = self.mixture_model.sample(pop_size)
                    for i in range(0, len(offspring_A)):
                        offspring_B = check_bounds(offspring_A[i].tolist())
                        solution2.append(offspring_B)
                # Offspring generated via standard reproduction during non-transfer intervals
                else: 
                    a1 = random.randint(0, pop_size - 1)
                    a2 = random.randint(0, pop_size - 1)
                    a = binary_tournament(a1, a2, self.obj1[:], self.obj2[:])
                    b1 = random.randint(0, pop_size - 1)
                    b2 = random.randint(0, pop_size - 1)
                    b = binary_tournament(b1, b2, self.obj1[:], self.obj2[:])
                    c1, c2 = SBX_crossover(self.sol[a], self.sol[b])
                    c1_mutated, c2_mutated = polynomial_mutation(c1, c2)
                    solution2.append(c1_mutated)
                    solution2.append(c2_mutated)
            function1_values2 = self.obj1[:]
            function2_values2 = self.obj2[:]
            #evaluate offspring obj
            for i in range(pop_size, 2 * pop_size):    
                obj = self.problem.evaluate(np.array(solution2[i]))
                function1_values2.append(obj[0])
                function2_values2.append(obj[1])
                
            non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
            crowding_distance_values2 = []
            for i in range(0, len(non_dominated_sorted_solution2)):
                crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
    
            # Environmental selection
            new_solution = []
            self.obj1, self.obj2 = [], []
            for i in range(0, len(non_dominated_sorted_solution2)):
                non_dominated_sorted_solution2[i].sort()
                front = sort_distance(non_dominated_sorted_solution2[i], crowding_distance_values2[i])
                front.reverse()
                for index in front:
                    new_solution.append(solution2[index])
                    self.obj1.append(function1_values2[index])
                    self.obj2.append(function2_values2[index])
                    if (len(new_solution) == pop_size):
                        break
                if (len(new_solution) == pop_size):
                    break
    
            self.sol = new_solution[:]
            self.gen_no += 1
            
            curr_sol = np.array(self.sol)
            self.pop_mean.append(np.mean(curr_sol, axis=0))
            self.pop_var.append(np.diag(np.cov(curr_sol.T)))