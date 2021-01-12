

import random
from source import Utils, SolutionsUtils, Polytope
from random import randint

def get_random_constrants_T(d, T):
    """
    Random generation of T user preferences with d criteria.

    :param n: integer: number of criteria.
    :param T: integer: number of user preferences.
    """
    formatted_constraints = list([])

    random_w = []
    sum = 0
    for j in range(d):
        rnd = random.uniform(0, 1)
        random_w.append(rnd)
        sum += rnd
    random_w = [x / sum for x in random_w]

    for i in range(T):
        # randomly select e_i
        e_i = randint(0, d - 1)
        e_j = e_i
        e_k = e_i
        # randomly select e_j
        while e_j == e_i:
            e_j = randint(0, d - 1)
        # randomly select e_k
        while e_k == e_i or e_k == e_j:
            e_k = randint(0, d - 1)

        coefficients = [0]*d
        a = random.uniform(0, 1)
        b = -random.uniform(0, 1)
        c = -random.uniform(0, 1)
        coefficients[e_i] = a
        coefficients[e_j] = b
        coefficients[e_k] = c
        if Utils.dot(random_w, coefficients) >= 0:
            sign = '>='
        else:
            sign = '<='
        formatted_constraints.append('%f w%i + %f w%i + %f w%i %s 0' % (a, e_i, b, e_j, c, e_k, sign))

    return formatted_constraints


def random_problem(vars, n, J, T, mu=10, theta=50, delta=60, int_values=False):
    """
    Random generation of J sets of n alternatives with T random user preferences.

    :param vars: list of string: variables name of each criteria.
    :param n: integer: number of elements for each set of alternative.
    :param J: integer: number of sets of alternatives.
    :param T: integer: number of user preferences.
    :param mu: integer: statistical parameter for random generator:.
    :param theta: integer: statistical parameter for random generator.
    :param delta: integer: statistical parameter for random generator.
    :param int_values: boolean: if true the alternatives are vectors of integer, otherwise vectors of floats.
    """

    d=len(vars)
    formatted_constraints = get_random_constrants_T(d, T)

    W = Polytope.Polytope(vars, frac=True)

    A = []
    for j in range(J):
        mu_j = random.random() * 2 * mu  - mu
        theta_j = random.random() * 2 * theta
        delta_j = random.random() * 2 * delta

        mu_ji = [0] * d
        delta_ji = [0] * d
        for i in range(d):
            mu_ji[i] = mu_j + random.random() * 2 * theta_j - theta_j
            delta_ji[i] = random.random() * 2 * delta_j
        Aj = []
        for k in range(n):
            v = [0] * d
            for i in range(d):
                if int_values:
                    v[i] = int(mu_ji[i] + random.random() * 2 * delta_ji[i] - delta_ji[i])
                else:
                    v[i] = mu_ji[i] + random.random() * 2 * delta_ji[i] - delta_ji[i]
            Aj.append(v)

        A.append(Aj)

    return A, formatted_constraints, W

def random_problem_UD(vars, n, J, T, mu=10, theta=50, delta=60, int_values=False):
    """
    Random generation of J sets of n undominated alternatives with T random user preferences.

    :param vars: list of string: variables name of each criteria.
    :param n: integer: number of elements for each set of alternative.
    :param J: integer: number of sets of alternatives.
    :param T: integer: number of user preferences.
    :param mu: integer: statistical parameter for random generator:.
    :param theta: integer: statistical parameter for random generator.
    :param delta: integer: statistical parameter for random generator.
    :param int_values: boolean: if true the alternatives are vectors of integer, otherwise vectors of floats.
    """

    d=len(vars)

    formatted_constraints = get_random_constrants_T(d, T)

    W = Polytope.Polytope(vars, frac=True)

    A = []
    for j in range(J):
        mu_j = random.random() * 2 * mu  - mu
        theta_j = random.random() * 2 * theta
        delta_j = random.random() * 2 * delta

        mu_ji = [0] * d
        delta_ji = [0] * d
        for i in range(d):
            mu_ji[i] = mu_j + random.random() * 2 * theta_j - theta_j
            delta_ji[i] = random.random() * 2 * delta_j
        Aj = []
        for k in range(n*10000):
            v = [0] * d
            for i in range(d):
                if int_values:
                    v[i] = int(mu_ji[i] + random.random() * 2 * delta_ji[i] - delta_ji[i])
                else:
                    v[i] = mu_ji[i] + random.random() * 2 * delta_ji[i] - delta_ji[i]
            #Aj.append(v)
            #ud = SolutionsUtils.UD(Aj, W)

            correct = True
            for s in Aj:
                if SolutionsUtils.pairwise_dominance(s, v, W, False) or SolutionsUtils.pairwise_dominance(v, s, W, False):
                    correct = False
                    break
            if correct:
                Aj.append(v)
                # print(k)
                # print(len(Aj))
            if len(Aj)==n:
                break

        if T!= 0:
            formatted_constraints = get_random_constrants_T(d, T)
            W.add_formatted_constraints(formatted_constraints)
            A.append(SolutionsUtils.UD(Aj, W))
        else:
            A.append(Aj)


    return A, formatted_constraints, W

def random_problem_PO(vars, n, J, T, mu=10, theta=50, delta=60, int_values=False):
    """
    Random generation of J sets of n possibly optimal alternatives with T random user preferences.

    :param vars: list of string: variables name of each criteria.
    :param n: integer: number of elements for each set of alternative.
    :param J: integer: number of sets of alternatives.
    :param T: integer: number of user preferences.
    :param mu: integer: statistical parameter for random generator:.
    :param theta: integer: statistical parameter for random generator.
    :param delta: integer: statistical parameter for random generator.
    :param int_values: boolean: if true the alternatives are vectors of integer, otherwise vectors of floats.
    """

    d=len(vars)

    formatted_constraints = get_random_constrants_T(d, T)

    W = Polytope.Polytope(vars, frac=True)

    A = []
    for j in range(J):
        mu_j = random.random() * 2 * mu  - mu
        theta_j = random.random() * 2 * theta
        delta_j = random.random() * 2 * delta

        mu_ji = [0] * d
        delta_ji = [0] * d
        for i in range(d):
            mu_ji[i] = mu_j + random.random() * 2 * theta_j - theta_j
            delta_ji[i] = random.random() * 2 * delta_j
        Aj = []
        for k in range(n*1000):
            v = [0] * d
            for i in range(d):
                if int_values:
                    v[i] = int(mu_ji[i] + random.random() * 2 * delta_ji[i] - delta_ji[i])
                else:
                    v[i] = mu_ji[i] + random.random() * 2 * delta_ji[i] - delta_ji[i]

            correct = True
            if SolutionsUtils.PO_test(v, Aj, W):
                Aj.append(v)
                if(SolutionsUtils.PO(Aj, W) != Aj):
                    Aj.remove(v)
                else:
                    print(k)
                    print(len(Aj))
            if len(Aj)==n:
                break

        if T!= 0:
            formatted_constraints = get_random_constrants_T(d, T)
            W.add_formatted_constraints(formatted_constraints)
            A.append(SolutionsUtils.PO(Aj, W))
        else:
            A.append(Aj)

    return A, formatted_constraints, W

def random_problem_PSO(vars, n, J, T, mu=10, theta=50, delta=60, int_values=False):
    """
    Random generation of J sets of n strictly optimal alternatives with T random user preferences.

    :param vars: list of string: variables name of each criteria.
    :param n: integer: number of elements for each set of alternative.
    :param J: integer: number of sets of alternatives.
    :param T: integer: number of user preferences.
    :param mu: integer: statistical parameter for random generator:.
    :param theta: integer: statistical parameter for random generator.
    :param delta: integer: statistical parameter for random generator.
    :param int_values: boolean: if true the alternatives are vectors of integer, otherwise vectors of floats.
    """

    d=len(vars)

    formatted_constraints = get_random_constrants_T(d, T)

    W = Polytope.Polytope(vars, frac=True)

    A = []
    for j in range(J):
        mu_j = random.random() * 2 * mu  - mu
        theta_j = random.random() * 2 * theta
        delta_j = random.random() * 2 * delta

        mu_ji = [0] * d
        delta_ji = [0] * d
        for i in range(d):
            mu_ji[i] = mu_j + random.random() * 2 * theta_j - theta_j
            delta_ji[i] = random.random() * 2 * delta_j
        Aj = []
        for k in range(n*1000):
            v = [0] * d
            for i in range(d):
                if int_values:
                    v[i] = int(mu_ji[i] + random.random() * 2 * delta_ji[i] - delta_ji[i])
                else:
                    v[i] = mu_ji[i] + random.random() * 2 * delta_ji[i] - delta_ji[i]

            correct = True
            if SolutionsUtils.PSO_test(v, Aj, W):
                add = True
                for s in Aj:
                    if not SolutionsUtils.PSO_test(s, Aj + [v], W):
                        add=False
                        break
                if(add):
                    Aj.append(v)
                    print(k)
                    print(len(Aj))
            if len(Aj)==n:
                break

        if T!= 0:
            formatted_constraints = get_random_constrants_T(d, T)
            W.add_formatted_constraints(formatted_constraints)
            A.append(SolutionsUtils.PO(Aj, W))
        else:
            A.append(Aj)

    return A, formatted_constraints, W
