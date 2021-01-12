import time
from source import Utils, SolutionsUtils
from pysat.solvers import Minicard
import itertools
import numpy as np


def sat_solver(x, A, k, sat_instances, sat_clauses, sat_solutions):
    """
    solve the SAT problem associated with x and the input parameters A, k and sat_clauses.

    :param x: vector of binary values: It represents a node of the tree used to represent a subset of A.
    :param A: list of vectors of floats:  set of alternatives.
    :param k: integer: cardinality of SMMR.
    :param sat_instances: vector of |A| Minicard models: the ith position represent the SAT model associated with the
    internal node representing the substring x(0,i).
    If the ith position null, then there are no model for the corresponding internal node
    :param sat_clauses: vector of binary vectors: SAT clauses for the SAT problem.
    :param sat_solutions: vector of |A| Minicard solutions: the ith position represent the solution of the SAT model associated with the
    internal node representing the substring x(0,i).
    If the ith position null, then there are no model for the corresponding internal node

    """

    start_time = time.time()
    time_to_add_constraints = 0

    index_constraints = -1
    if len(x)>1 and  sat_instances[len(x)-2] != None: #sat of parent node
        # print()
        # print(sat_solutions[len(x)-2][len(x)-1])
        # print(x[len(x)-1])
        if np.sign(sat_solutions[len(x)-2][len(x)-1]) == np.sign(x[len(x)-1]-0.5):
            return True, 0, 0, 0
        g=sat_instances[len(x)-2]
        #set X_j=0 or X_j=1

        i = len(x)-1
        if x[i] == 1:
            g.add_clause([i + 1])
        else:
            g.add_clause([-(i + 1)])
    else:
        g = Minicard()
        for c in sat_clauses:
            g.add_clause(c)

        # The cardinality constraint that |By| = K (i.e., that y is a complete string) isexpressed as sum_αi∈A Xi = K.
        v = []
        for i in range(len(A)):
            v.append(i + 1)
        ad = time.time()
        g.add_atmost(v, k)
        time_to_add_constraints += time.time() - ad

        # The constraint that y extends x is expressed as: For all i ∈ {1, Len(x)},
        for i in range(len(x)):
            # if x(i) = 1 then Xi = 1 (where x(i) is the ith position of Boolean string x);
            if x[i] == 1:
                g.add_clause([i + 1])
            else:
                g.add_clause([-(i + 1)])




    constraints_time = time.time() - start_time

    start_time = time.time()
    feasible = g.solve()
    if feasible:
        sat_solutions[len(x) - 1] = g.get_model()
        sat_instances[len(x) - 1] = g
    satsolver_time =  time.time() - start_time

    # print(feasible)
    # print(constraints_time)
    # print(time_to_add_constraints)
    # print(satsolver_time)


    return feasible, constraints_time, time_to_add_constraints, satsolver_time


def get_sat_clauses(Wp_valA, A, r, up_Wp=float('inf')):
    """
    List of SAT clauses, one for each w ∈ W0 containing index of α ∈ A such taht  ValA(w) − fw(α) < r

    :param Wp_valA: dictionary string, integer: Val A for each point of W'
    :param A: list of vectors of floats:  set of alternatives.
    :param r: float: current upper bound of SMMR.
    :param up_Wp: integer: upper bound on the cardinality of W'.
    :return:
    """
    clauses = []
    count=1
    for key in Wp_valA:
        if count > up_Wp:
            break
        count+=1
        ep = [float(j) for j in key.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(',', '').split()]
        l = []
        for i in range(len(A)):
            valA = Wp_valA[key]
            dot = Utils.dot(ep, A[i])
            if valA - dot < r:
                l.append(i+1)
        clauses.append(l)
    return clauses


def  nextBT(x, k, n):
    """
    Generate next string with backtracking case.

    :param x: vector of binary values: it represents a node of the tree used to represent a subset of A.
    :param k: integer: cardinality of SMMR.
    :param n: integer: number of alternatives (|A|)
    :return:
    """



    #If Ifxcontains no zero (i.e., it only contains ones) orxstarts with K ones (i.e.x=1[k]0[N−K]) then we finish:
    #  since we have then completely covered the space
    if x.count(0) == 0 or x[:k].count(1) == k:
        return [-1]

    #Otherwise, we can write x as x=yz where:

    # z=01[i] if x ends with a one where y is a (possibly empty) string and i >0
    if x[len(x) - 1]  == 1:
        last_zero_index = len(x) - 1 - x[::-1].index(0)
        y = x[:last_zero_index]

    # z= 0 if x ends with a zero and Ones(x)< K
    if x.count(1) < k and x[len(x) - 1] == 0:
        y = x[:len(x) - 1]

    # z= 01[i]0[j] if x ends with a zero and Ones(x)=K where i >0 and j >0
    if x.count(1) == k and x[len(x) - 1] == 0:
        last_one_index = len(x) - 1 - x[::-1].index(1)
        x1 = x[:last_one_index]
        last_zero_index = len(x1) - 1 - x1[::-1].index(0)
        y = x1[:last_zero_index]

    # If Ones(y) = K − 1 we define NextBT(x) to be the complete string y10 · · · 0. (The
    # reason for filling with zeros is that we already have the maximum number of ones.)
    if y.count(1) == k - 1:
        return y + [1] + [0] * (n - len(y) - 1)
    # If Ones(y) < K −1, we define NextBT(x) to be the string y1.
    else:
        return y + [1]

def next(x, k, n):
    """
    Generate next string with non-backtracking case, and so x is a partial string.

    :param x: vector of binary values: current node.
    :param k: integer: cardinality of SMMR.
    :param n: integer: number of alternatives (|A|)
    :return:
    """


    #We start with x being the empty string, and we move next to the string 0.
    if x[:k].count(1) == k:
        return [-1]

    zeros_x = x.count(0)
    # If Zeros(x)< N−K−1 and Ones(x)= K then we set Next(x) to be the string x0...0
    if zeros_x < n - k - 1 and x.count(1) == k:
        return x + [0] * (n - len(x))
    #If Zeros(x)< N−K−1 and Ones(x)< K then we set Next(x) to be the string x0...0
    elif zeros_x < n - k - 1:
        return x + [0]
    #Otherwise, we haveZeros(x) =N−K−1 and we set Next(x)to be the complete string of the form x01···1
    # (since x0 already has the maximum number of zeros, so all the later values must be 1).
    else:
        return x + [0] + [1] * (n - len(x) - 1)


def getBFromString(x, A):
    """

    :param x: string of binary values: it represents a node of the tree used to represent a subset B of A.
    :param A: list of vectors of floats: set of alternatives.
    :return: list of vectors: set of alternatives corresponding to the set B represented by x.
    """
    B = []
    indexes_B = []
    for i in range(len(A)):
        if x[i] == 1:
            B.append(A[i])
            indexes_B.append(i)
    return B, indexes_B


def SMMR_BF_LP(k, A, polytope):
    """
    Computation of SMMR of A with cardinality k and polytope W. Brute force algorithm using linear programming.

    :param A: list of vectors of floats:  set of alternatives.
    :param k: integer: cardinality of SMMR.
    :param polytope: Instance of class Polytope: user preference state space
    :return:
    """
    cominations = itertools.combinations(A, k)
    r = float('inf')
    B_star = []
    tot_it=0
    for B in cominations:
        tot_it+=1
        SMR, PW = SolutionsUtils.SMR_LP(B, A, polytope)
        if SMR < r:
            r = SMR
            B_star = B
    return r, B_star, tot_it


def SMMR_BF_EPI(k, A, polytope):
    """
    Computation of SMMR of A with cardinality k and polytope W. Brute force algorithm using epigraph of the value function.

    :param A: list of vectors:  set of alternatives.
    :param k: integer: cardinality of SMMR.
    :param polytope: Instance of class Polytope: user preference state space
    :return:
    """

    cominations = itertools.combinations(A, k)
    r = float('inf')
    B_star = []
    tot_it=0
    for B in cominations:
        tot_it+=1
        SMR, w = SolutionsUtils.SMR_epi(B, A, polytope)
        if SMR < r:
            r = SMR
            B_star = B
    return r, B_star, tot_it


def SMMR_SAT_LP(k, A, polytope, up_Wp = float('inf'), upper_bound=float('inf')):
    """
    Computation of SMMR of A with cardinality k and polytope W. SAT based algorithm using linear programming.

    :param A: list of vectors:  set of alternatives.
    :param k: integer: cardinality of SMMR.
    :param polytope: Instance of class Polytope: user preference state space
    :param up_Wp: integer: upper bound on the cardinality of W'.
    :param upper_bound: float: initial upper bpund of SMMR
    :return:
    """
    n=len(A)
    r= upper_bound
    it=0
    A_ordered = A
    best_single_MR= [float('inf')] * len(A)

    SMR_tot_time = 0
    tot_SMR_calls = 0
    SAT_tot_time = 0

    if k==1:
        r, sol, best_single_MR = SolutionsUtils.mMR(A, polytope)
        A_ordered = [x for _, x in sorted(zip(best_single_MR, A), key=lambda pair: pair[0])]
        A_ordered = list(reversed(A_ordered))
        best_single_MR = list(reversed(sorted(best_single_MR)))
        return r, sol, 0, best_single_MR, A_ordered, 0

    x = [0] * (n - k) + [1] * k

    sat_instances = [None]*n
    sat_solutions = [None]*n
    sat_clauses = None

    B_star =[]
    tot_it = it

    # W' and corresponding Val_A
    Wp_valA = {}

    while x != [-1]:
        tot_it+=1
        if len(x) ==n:
            #print(x)

            B, indexes_B = getBFromString(x, A_ordered)
            #
            worst_max_regret = True
            #check SMR in extreme points
            for ww in Wp_valA:
                w = [float(i) for i in ww.strip('][').split(', ')]
                worst_max_regret = True
                for alpha in B:
                    dot = Utils.dot(alpha, w)
                    if Wp_valA[ww] - dot <= r:
                        worst_max_regret = False
                        break
                if worst_max_regret:
                    break

            if worst_max_regret and len(Wp_valA)>0:
                x = nextBT(x, k, n)
                continue

            tot_SMR_calls+=1


            start_time = time.time()

            # start_time = time.time()
            SMR, Wp= SolutionsUtils.SMR_LP(B, A, polytope)
            SMR_time = time.time() - start_time
            SMR_tot_time += SMR_time
            # print(SMR_time)
            # print(SMR)

            if SMR < r:
                for w in Wp:
                    w = [float(i) for i in w]
                    if str(w) not in Wp_valA:
                        valA = - float('inf')
                        for i in range(len(A)):
                            alpha = A[i]
                            dot = Utils.dot(alpha, w)
                            if dot > valA:
                                valA = dot
                                indexA = i
                        Wp_valA[str(w)] = valA

                r = SMR

                B_star = B
                sat_instances = [None] * n
                sat_solutions = [None] * n

                start_time = time.time()
                sat_clauses = get_sat_clauses(Wp_valA, A_ordered, r, up_Wp)
                SAT_time = time.time() - start_time
                SAT_tot_time += SAT_time

            x = nextBT(x,k,n)
        else:
            #print(x)

            start_time = time.time()
            if sat_clauses == None:
                sat_clauses = get_sat_clauses(Wp_valA, A_ordered, r, up_Wp)
            feasible, constraints_time, time_to_add_constraints, satsolver_time = sat_solver(x, A_ordered, k, sat_instances, sat_clauses, sat_solutions)
            SAT_time = time.time() - start_time
            SAT_tot_time += SAT_time

            if feasible:
                x = next(x, k, n)
            else:
                x = nextBT(x, k, n)

    #reorder A
    A_ordered = [x for _, x in sorted(zip(best_single_MR, A_ordered), key=lambda pair: pair[0])]
    A_ordered = list(reversed(A_ordered))
    best_single_MR = list(reversed(sorted(best_single_MR)))
    return r, B_star, tot_it, best_single_MR, A_ordered, len(sat_clauses),SMR_tot_time,tot_SMR_calls, SAT_tot_time


def SMMR_SAT_EPI(k, A, polytope, up_Wp = float('inf'), upper_bound=float('inf')):
    """
    Computation of SMMR of A with cardinality k and polytope W. SAT based algorithm using epigraph of the value function.

    :param A: list of vectors:  set of alternatives.
    :param k: integer: cardinality of SMMR.
    :param polytope: Instance of class Polytope: user preference state space
    :param up_Wp: integer: upper bound on the cardinality of W'.
    :param upper_bound: float: initial upper bpund of SMMR
    :return:
    """

    n=len(A)
    it=0
    A_ordered = A
    best_single_MR= [float('inf')] * len(A)

    SMR_tot_time = 0
    tot_SMR_calls = 0
    SAT_tot_time = 0

    if k==1:
        r, sol, best_single_MR = SolutionsUtils.mMR(A, polytope)
        A_ordered = [x for _, x in sorted(zip(best_single_MR, A), key=lambda pair: pair[0])]
        A_ordered = list(reversed(A_ordered))
        best_single_MR = list(reversed(sorted(best_single_MR)))
        return r, sol, 0, best_single_MR, A_ordered, 0

    x = [0] * (n - k) + [1] * k

    sat_instances = [None]*n
    sat_solutions = [None]*n
    sat_clauses = None

    B_star =[]
    tot_it = it

    # W' and corresponding Val_A
    Wp_valA = {}

    r=upper_bound

    while x != [-1]:
        tot_it+=1
        if len(x) ==n:
            B, indexes_B = getBFromString(x, A_ordered)

            worst_max_regret = True
            #check SMR in extreme points
            for ww in Wp_valA:
                w = [float(i) for i in ww.strip('][').split(', ')]
                worst_max_regret = True
                for alpha in B:
                    dot = Utils.dot(alpha, w)
                    if Wp_valA[ww] - dot <= r:
                        worst_max_regret = False
                        break
                if worst_max_regret:
                    break

            if worst_max_regret and len(Wp_valA)>0:
                x = nextBT(x, k, n)
                continue

            tot_SMR_calls+=1


            start_time = time.time()
            SMR, Wp = SolutionsUtils.SMR_epi(B, A_ordered, polytope)
            SMR_time = time.time() - start_time
            SMR_tot_time += SMR_time

            if SMR < r:
                for w in Wp:
                    w = [float(i) for i in w]
                    if str(w) not in Wp_valA:
                        valA = - float('inf')
                        for i in range(len(A)):
                            alpha = A[i]
                            dot = Utils.dot(alpha, w)
                            if dot > valA:
                                valA = dot
                                indexA = i
                        Wp_valA[str(w)] = valA
                r = SMR
                B_star = B
                sat_instances = [None] * n
                sat_solutions = [None] * n

                start_time = time.time()
                sat_clauses = get_sat_clauses(Wp_valA, A_ordered, r, up_Wp)
                SAT_time = time.time() - start_time
                SAT_tot_time += SAT_time

                #print(B_star)
            x = nextBT(x,k,n)
        else:

            start_time = time.time()
            if sat_clauses == None:
                sat_clauses = get_sat_clauses(Wp_valA, A_ordered, r, up_Wp)
            feasible, constraints_time, time_to_add_constraints, satsolver_time = sat_solver(x, A_ordered, k, sat_instances, sat_clauses, sat_solutions)
            SAT_time = time.time() - start_time
            SAT_tot_time += SAT_time

            if feasible:
                x = next(x, k, n)
            else:
                x = nextBT(x, k, n)

    #reorder A
    A_ordered = [x for _, x in sorted(zip(best_single_MR, A_ordered), key=lambda pair: pair[0])]
    A_ordered = list(reversed(A_ordered))
    best_single_MR = list(reversed(sorted(best_single_MR)))
    # print(r)
    # print('SMR tot time: ' + str(round(SMR_tot_time,1)))
    # print('tot SMR calls: ' + str(round(tot_SMR_calls,1)))
    return r, B_star, tot_it, best_single_MR, A_ordered, len(sat_clauses), SMR_tot_time, tot_SMR_calls, SAT_tot_time


