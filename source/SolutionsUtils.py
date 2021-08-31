"""
\n
Set of methods to be used for:\n
\t1 - Evaluating alternatives (pairwise_dominance, PO_test, PSO_test, UD_test, MR, SMR, mMR, mSMR_size2, MMD_size2).\n
\t2 - Filtering set of alternatives (PO, PSO, UD). \n
\t3 - Query computation (mSMR_size2, MMD_size2).\n
\t4 - Generic operations (replace_dominated_alternatives).\n
\n
Notation of comments: \n
\t1 - Objects alternative are indicated with s and t. \n
\t2 - Sets of objects alternative are indicated with A and B. \n
\t3 - Objects polytope (i.e. set of user preferences) are indicated with W. \n
\t4 - Points of polytopes are indicated with w. \n
"""

import cplex
import time
from source import Utils, Polytope
import copy


def pairwise_dominance(s, t, W, strict=True):
    """ Testing if s dominates t.
    Recall that a alternative s strictly dominates alternative t (w.r.t. W) if and only if
    for all consistent weights vectors w, we have :math:`w\cdot (y_s - y_t)\geq 0`, and for some consistent weights
    vector w we have :math:`w\cdot (y_s - y_t)> 0`. Aalternative s
    dominates alternative t (w.r.t. W) if and only if
    for all consistent weights vectors w, we have :math:`w\cdot (y_s - y_t)\geq 0`.
    Since :math:`w\cdot (y_s - y_t)` is a linear
    function and W is a convex polytope, we can check if s strictly dominates alternative
    t evaluating
    only the extreme points :math:`Ext(W)` of W.


    :param s: vector of float: alternative.
    :param t: vector of float: alternative.
    :param W: object of class Polytope.
    :param strict: boolean (default value=False).
        If true the function check if s strictly dominates t.
    :return: boolean.
        True if s dominates t.
    """

    dominance = False
    for w in W.extreme_points:
        scalar_utility_s = Utils.dot(w, s)
        scalar_utility_t = Utils.dot(w, t)

        val = scalar_utility_s - scalar_utility_t

        if strict:
            if val < 0:
                return False

            elif val > 0:
                dominance = True
        else:
            epsilon=-0.00001
            if val < epsilon:
                return False
            else:
                dominance = True

    return dominance

def UD_test(s, A, W):
    """ Testing if there exists an element in A that (strictly) dominates s
    Recall that that s is undominated in A (w.r.t. W) if there does not exist any
    element t in A that dominates s.

    :param s: vector of float: alternative.
    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: boolean.
        True if s is undominated in A.
    """


    for t in A:
        if pairwise_dominance(t, s, W, False):
            return False
    return True

def UD(A, W):
    """ Computing the set of undominated elements of A.
    Recall that UD(A) is the set of undominated alternatives in A (w.r.t. W).
    To compute UD(A), we remove all the strictly dominated alternatives in A.
    we keep only one element for each subset of A of equivalent alternatives.

    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: list of alternatives.
        List of undominated elements of A.
    """

    A_UD = copy.deepcopy(A)
    i=0
    while i < len(A_UD):
        s = A_UD[i]
        j=i+1
        while j < len(A_UD):
            t = A_UD[j]
            if pairwise_dominance(s,t,W, False):
                A_UD.remove(t)
            elif pairwise_dominance(t,s,W, False):
                A_UD.remove(s)
                i -= 1
                break
            else:
                j +=1


        i+=1
    return A_UD

def PO_test(s, A, W):
    """ Testing if alternative s is possibly optimal in A.
    (based only on the utility vector of objects alternative, i.e. weights vectors
    of objects alternative are ignored).
    Recall that alternative s is possibly optimal in A (w.r.t. W), i.e.,
    s is in PO(A),
    if and only if s is in A and s is optimal given
    some consistent w. Therefore, to test if a
    alternative s is in PO(A), we can test if there exists at least a consistent w such that
    :math:`w\cdot (y_s - y_t)\geq 0` for all :math:`t\in A\setminus \{s\}`
    by checking the feasibility of a linear programming problem.

    :param s: vector of float: alternative.
    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: boolean.
        True if s is possibly optimal in A.
    """



    A_copy = copy.deepcopy(A)
    if s in A_copy:
        A_copy.remove(s)


    obj_function = ''
    for i in range(len(W.vars)):
        obj_function += '0 %s' % W.vars[i]
        if i != len(W.vars) -1:
            obj_function  += ' + '

    formatted_constraints = copy.deepcopy(W.formatted_constraints)
    for t in A_copy:
        new_constraint = ''
        for i in range(len(s)):
            if i == len(s) - 1:
                new_constraint += str(s[i] - t[i]) + ' w' + str(i) + ' >= 0'
            else:
                new_constraint += str(s[i] - t[i]) + ' w' + str(i) + ' + '
        formatted_constraints.append(new_constraint)

    var_names, obj_function_coefficients, coefficients_matrix, constant_terms, constraints_sense = \
        Utils.get_cplex_constraints(obj_function, formatted_constraints)

    start_time = time.time()
    my_prob = cplex.Cplex()
    # avoid printing results
    my_prob.set_log_stream(None)
    my_prob.set_error_stream(None)
    my_prob.set_warning_stream(None)
    my_prob.set_results_stream(None)

    my_prob.objective.set_sense(my_prob.objective.sense.maximize)
    lower_bounds = [0]  * (len(W.vars))
    upper_bounds = [1]  * (len(W.vars))
    my_prob.variables.add(obj=obj_function_coefficients, names=var_names, lb=lower_bounds, ub=upper_bounds)
    my_prob.linear_constraints.add(lin_expr=coefficients_matrix, rhs=constant_terms, senses=constraints_sense)
    my_prob.solve()
    # print(my_prob.linear_constraints.get_rows())
    cplex_time = time.time() - start_time

    status = my_prob.solution.get_status()
    my_prob.end()
    del my_prob

    if status == 3:  # unfeasible set
        return False
    else:
        return True

def PO(A, W):
    """Generating the set of possibly optimal alternatives in A.
    (based only on the utility vector of objects alternative, i.e. weights
    vectors of objects alternative are ignored).
    We compute PO(A)
    by testing if each s is possibly optimal in A (w.r.t. W),
    and removing s from A
    if it is not possibly optimal.

    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: list of alternatives.
         List of possibly optimal elements of A.
    """


    A_PO = copy.deepcopy(A)
    for s in A:
        if not PO_test(s, A_PO, W):
            A_PO.remove(s)
    return A_PO

def PSO_test(s, A, W):
    """ Testing if alternative s is possibly strictly optimal in A.
    (based only on the utility vector of objects alternative, i.e.
    weights vectors of objects alternative are ignored).
    Recall that alternative s is possibly strictly optimal in A
    (w.r.t. the W), i.e., s is in PSO(A),
    if and only if s is in A and s
    is strictly optimal given some consistent w. Therefore, to test if a
    alternative s is in PO(A), we can test if there exists :math:`x>0` such that
    :math:`w\cdot (y_s - y_t)\geq x` for all :math:`t \in A \setminus \{A\}` and :math:`w\in W`
    using a linear programing solver.

    :param s: object of class Solution.
    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: boolean.
        True if s is possibly strictly optimal in A.
    """


    A_copy = copy.deepcopy(A)
    if s in A_copy:
        A_copy.remove(s)


    s_is_PSO = False

    #x is to avoid stric inequalities
    obj_function = '1 x + '
    for i in range(len(W.vars)):
        obj_function += '0 %s' % W.vars[i]
        if i != len(W.vars) -1:
            obj_function  += ' + '

    formatted_constraints = copy.deepcopy(W.formatted_constraints)
    for t in A_copy:
        new_constraint = ''
        for i in range(len(s)):
            if i == len(s) - 1:
                new_constraint += str(s[i] - t[i]) + ' w' + str(i) + ' + -1 x >= 0'
            else:
                new_constraint += str(s[i] - t[i]) + ' w' + str(i) + ' + '
        formatted_constraints.append(new_constraint)
    var_names, obj_function_coefficients, coefficients_matrix, constant_terms, constraints_sense = \
        Utils.get_cplex_constraints(obj_function, formatted_constraints)


    start_time = time.time()
    my_prob = cplex.Cplex()
    # avoid printing results
    my_prob.set_log_stream(None)
    my_prob.set_error_stream(None)
    my_prob.set_warning_stream(None)
    my_prob.set_results_stream(None)

    my_prob.objective.set_sense(my_prob.objective.sense.maximize)
    lower_bounds = [-float('inf')]  * (len(W.vars) + 1)
    my_prob.variables.add(obj=obj_function_coefficients, names=var_names, lb=lower_bounds)
    my_prob.linear_constraints.add(lin_expr=coefficients_matrix, rhs=constant_terms, senses=constraints_sense)
    my_prob.solve()
    # print(my_prob.linear_constraints.get_rows())
    cplex_time = time.time() - start_time

    if my_prob.solution.get_status() != 3:
        if my_prob.solution.get_status() == 4:
            alpha = float('inf')
        else:
            alpha = my_prob.solution.get_objective_value()
            # print(alpha)
        #solution_point = my_prob.solution.get_values()

        if alpha <= 0.00000001:
            s_is_PSO = False
        else:
            s_is_PSO = True
    else:
        s_is_PSO = False

    my_prob.end()
    del my_prob

    return s_is_PSO

def PSO(A, W):
    """ Generating the set of possibly strictly optimal alternatives in A.
    (based only on the utility vector of objects alternative,
    i.e. weights vectors of objects alternative are ignored).
    We compute PSO(A)
    by testing if each s is possibly optimal in A (w.r.t. W),
    and removing s from A
    if it is not strictly possibly optimal.
    Note that PSO make also the input set equivalence free. If there are two equivalent alternative in A,
    PSO(A) will contain only the first appearing on the list A.

    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: list of alternatives.
         List of possibly striclty optimal elements of A.
    """


    A_PSO = copy.deepcopy(A)
    for s in A:
        if not PSO_test(s, A_PSO, W):
            A_PSO.remove(s)
    return A_PSO

def NO_test(s, A, W):
    """ Testing if s is necessarily optimal A.
    (based only on the utility vector of objects alternative, i.e.
    weights vectors of objects alternative are ignored).
    Recall that element s of A is necessarily optimal in A
    (w.r.t. W), i.e., s is in NO(A),
    if and only if s is optimal given every consistent w,
    Therefore, s is in NO(A) if and only if :math:`w\cdot (y_s - y_t)\geq 0`
    for any :math:`w \in W` and for any :math:`t \in A`.
    Since :math:`w\cdot (y_s - y_t)` is a linear function and W is a convex polytope,
    we can check if s is in NO(A) evaluating
    only the extreme points :math:`Ext(W)` of W,
    i.e. s is in NO(A) if and only if
    :math:`w\cdot (y_s - y_t)< 0` is false for any :math:`w \in Ext(W)` and for any :math:`t \in A`.

    :param s: object of class Solution.
    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: boolean.
        True if s is necessarily optimal in A.
    """


    for w in W.extreme_points:
        scalar_utility_s = Utils.dot(w, s)
        for t in A:
            scalar_utility_t = Utils.dot(w, t)
            if scalar_utility_s - scalar_utility_t < 0:
                return False
    return True

def SMR_LP(B, A, W):
    """ Computation of worst case loss of B with respect to A and W
    using the linear programming solver

    :param B: list of alternatives
    :param A: list of alternatives.
    :param W: object of class Polytope.
    :return: float, list of vectors of floats.
        SMR of B with respect to A, list of points w in W where SMR_W(B,{a}) is maximised for each ai in A
    """


    #x is to avoid stric inequalities
    obj_function = '1 x + '
    for i in range(len(W.vars)):
        obj_function += '0 %s' % W.vars[i]
        if i != len(W.vars) -1:
            obj_function  += ' + '

    SMR = -float('Inf')
    PW = []
    w = 0

    tot_constraint_time = 0
    tot_cplex_time =0
    formatted_constraints =None
    #for each beta solve a linear programming solver
    for t in A:
        #print(beta)
        constraint_time  = time.time()
        formatted_constraints =  copy.deepcopy(W.formatted_constraints)
        for s in B:
            new_constraint = ''
            for i in range(len(s)):
                if i == len(s) - 1:
                    new_constraint += str(t[i] - s[i] ) + ' w' + str(i) + ' + -1 x >= 0'
                else:
                    new_constraint += str(t[i] - s[i]) + ' w' + str(i) + ' + '
            formatted_constraints.append(new_constraint)

        tot_constraint_time+= time.time() - constraint_time

        # for c in formatted_constraints:
        #     print(c)


        var_names, obj_function_coefficients, coefficients_matrix, constant_terms, constraints_sense = \
            Utils.get_cplex_constraints(obj_function, formatted_constraints)

        cplex_time = time.time()
        my_prob = cplex.Cplex()
        #avoid printing results
        my_prob.set_log_stream(None)
        my_prob.set_error_stream(None)
        my_prob.set_warning_stream(None)
        my_prob.set_results_stream(None)

        my_prob.objective.set_sense(my_prob.objective.sense.maximize)

        lower_bounds = [-float('inf')] * (len(W.vars) + 1)
        my_prob.variables.add(obj=obj_function_coefficients, lb=lower_bounds, names=var_names)
        my_prob.linear_constraints.add(lin_expr=coefficients_matrix, rhs=constant_terms, senses=constraints_sense)
        my_prob.solve()
        #print(my_prob.linear_constraints.get_rows())
        tot_cplex_time += time.time() - cplex_time

        #print(my_prob.solution.get_status())
        if my_prob.solution.get_status() != 3:
            alpha = my_prob.solution.get_objective_value()
            solution_point = my_prob.solution.get_values()
            w=solution_point[1:]
            if w not in PW:
                PW.append(w)

            #
            if alpha > SMR:
                w=solution_point[1:]
                # print(w)

            SMR = max(SMR, alpha)


        else:
            raise Exception('Unfeasible liner programming problem computing MR')

        my_prob.end()
        del my_prob
    # print(tot_constraint_time)
    # print(tot_cplex_time)
    return SMR, PW

def SMR_epi(B, A, W):
    """ Computation of worst case loss of B with respect to A and W using the
    epigraph of the value function.

    :param B: list of alternatives.
    :param A: list of alternatives.
    :param W: object of class Polytope.
    :return: float, discrete subset of W.
        SMR of B with respect to A, projection in W of extreme points of epigraph of B
    """

    vars2 = W.vars.copy()
    vars2 = ['x'] + vars2
    formatted_constraints_epigraph = W.formatted_constraints.copy()
    for alpha in B:
        constraint = ''
        for i in range(len(alpha)):
            if i < len(alpha) - 1:
                constraint += str(-alpha[i]) + ' w' + str(i) + ' + '
            else:
                constraint += str(-alpha[i]) + ' w' + str(i) + ' + 1 x >= 0'

        formatted_constraints_epigraph.append(constraint)

    epigraph_B = Polytope.Polytope(vars2, frac=W.frac, epi_var='x')
    epigraph_B.add_formatted_constraints(formatted_constraints_epigraph)


    # computation of SMR(B,A) and update Wp := Wp ∪ WB; where WB is the projection in W of the extreme points of the
    # epigraph of B
    eps = []
    MR_W_A_B = -float('inf')
    for p in epigraph_B.extreme_points:
        #projection ext pints epigraph in W
        #W0_ep = p[1:]
        W0_ep = list(p[1:])
        eps.append(W0_ep)
        if W.frac:
            for j in range(len(W0_ep)):
                W0_ep[j]=float(W0_ep[j])

        #update Wp_valA and evaluating i-th ext point of epi_B fot the computation of SMR(B,A)
        valA = - float('inf')
        for i in range(len(A)):
            alpha = A[i]
            dot = Utils.dot(alpha, W0_ep)
            if dot > valA:
                valA = dot


        if  valA - p[0] > MR_W_A_B:
            MR_W_A_B = valA - p[0]
            # print(W0_ep)


    return MR_W_A_B, eps

def mSMR_size2(A, W):
    """ Computing setwise minimax regret of subset of A with cardinality 2 (w.r.t. W).
    (based only on the utility vector of objects alternative,
    i.e. weights vectors of objects alternative are ignored).

    We compute SMR(B, A, W) for each subset of
    A of cardinality 2 and take the minimum.

    :param A: list of alternatives.
    :param W: object of class Polytope.
    :return: float, list of alternatives.
        setwise minimax regret of subset of A with cardinality 2,
        and corresponding  subset of cardinality 2
        that minimize the setwise regret.
    """

    mSMR = float('inf')
    s = None
    i=0
    while i < len(A):
        j=i+1
        while j < len(A):
            SMRij = SMR([A[i], A[j]], A, W)
            if SMRij < mSMR:
                mSMR = SMRij
                s = [A[i], A[j]]
            j += 1
        i += 1

    return mSMR, copy.deepcopy(s)

def MR(s, A, W):
    """ Computing max regret of s in A (w.r.t. W) evaluating extreme points of W.
    (based only on the utility vector of objects alternative, i.e. weights
    vectors of objects alternative are ignored).
    The MR of s in A is maximized in one of the extreme points of W.

    :param s: vector of float: alternative.
    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: float, vector of floats.
        Max regret of s with respect to A, and point w in W in which regret of s is maximised
    """


    MRs = -float('inf')
    wstar = None

    for w in W.extreme_points:
        for t in A:
            val = Utils.dot(Utils.vector_sub(t, s), w)
            if val> MRs:
                MRs= val
                wstar=w

    return MRs, wstar

def PMR_LP(s, t, W):
    """ Computing max regret of s with respect to t (w.r.t. W) using linear programming.

    :param s: vector of float: alternative.
    :param t: vector of float: alternative.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: float, vector of floats.
        Max regret of s with respect to t, and point w in W in which regret of s is maximised
    """


    #coefficient linear objective function for for var_names
    obj_function_lin = ''
    for i in range(len(s)):
        obj_function_lin += str(t[i] - s[i]) + ' ' + W.vars[i]
        if i != len(s) -1:
            obj_function_lin  += ' + '


    # # adding formatted constraints to get normalized preference vecotrs
    # p = Polytope.Polytope(perf_vars)
    # formatted_constraints = formatted_constraints + p.formatted_constraints

    formatted_constraints = copy.deepcopy(W.formatted_constraints)
    var_names, obj_function_coefficients, coefficients_matrix, constant_terms, constraints_sense = \
        Utils.get_cplex_constraints(obj_function_lin, formatted_constraints)

    start_time = time.time()
    my_prob = cplex.Cplex()
    # avoid printing results
    my_prob.set_log_stream(None)
    my_prob.set_error_stream(None)
    my_prob.set_warning_stream(None)
    my_prob.set_results_stream(None)

    my_prob.objective.set_sense(my_prob.objective.sense.maximize)


    my_prob.variables.add(obj=obj_function_coefficients, names=var_names)


    my_prob.linear_constraints.add(lin_expr=coefficients_matrix, rhs=constant_terms, senses=constraints_sense)
    my_prob.solve()
    # print(my_prob.linear_constraints.get_rows())
    tot_time = time.time() - start_time

    # print(my_prob.solution.get_status())
    if my_prob.solution.get_status() != 3:
        alpha = my_prob.solution.get_objective_value()
        solution_point = my_prob.solution.get_values()



    else:
        raise Exception('Unfeasible liner programming problem computing MR')

    my_prob.end()
    del my_prob
    return alpha, solution_point

def MR_LP(s, A, W):
    """ Computing max regret of s in A (w.r.t. W) using the linear programming solver.
    (based only on the utility vector of objects alternative, i.e. weights
    vectors of objects alternative are ignored).

    :param s: vector of float: alternative.
    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: float, vector of floats.
        Max regret of s with respect to A, and point w in W in which regret of s is maximised.
    """


    MRs = -float('inf')
    wstar = None

    for t in A:
        val, w = PMR_LP(s, t, W)
        if val> MRs:
            MRs= val
            wstar=w

    return MRs, wstar

def mMR(A, W):
    """ Computing minimax regret of A evaluating extreme point of W. It can be used to select a binary query.
    (based only on the utility vector of objects alternative,
    i.e. weights vectors of objects alternative are ignored).

    We compute SMR(s, A) for each alternative :math:`s\in A`, and take the minimum.

    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: float, alternative, list of floats.
        Minimax regret of A w.r.t. W, and corresponding
        alternative that minimize the max regret in A, and list of max regret of every alternative of A.
    """

    single_MR = []
    mMR = float('inf')
    s = None
    for t in A:
        MRt, wstar = MR(t, A, W)
        single_MR.append(MRt)
        if MRt < mMR:
            mMR = MRt
            s = t

    return mMR, copy.deepcopy(s), single_MR

def mMR_LP(A, W):
    """ Computing minimax regret of A using the linear programming solver. It can be used to select a binary query.
    (based only on the utility vector of objects alternative,
    i.e. weights vectors of objects alternative are ignored).

    We compute SMR(s, A) for each alternative :math:`s\in A`, and take the minimum.

    :param A: list of alternatives.
    :param W: object of class Polytope.
        It represents the user preferences.
    :return: float, alternative, list of floats.
        Minimax regret of A w.r.t. W, and corresponding
        alternative that minimize the max regret in A, and list of max regret of every alternative of A.
    """

    single_MR = []
    mMR = float('inf')
    s = None
    for t in A:
        MRt, wstar = MR_LP(t, A, W)
        single_MR.append(MRt)
        if MRt < mMR:
            mMR = MRt
            s = t

    return mMR, copy.deepcopy(s), single_MR

