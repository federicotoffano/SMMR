from source import SMMR, SolutionsUtils, RandomProblem
import time
import datetime
import statistics

#Number of repetitions for the experiments
repetitions = 2

dateTimeObj = datetime.datetime.now()
#filename to store summary of the experiments
filename = 'experiments/random_' + str(dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S"))
#filename to store details of the experiments
filename_data = 'experiments/random_' + str(dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S")) + "_data"



#Experiment header
current_test = 'Table2'
file1 = open(filename, "a+")
file1.write('\n\n' + current_test)
file1.close()
file1 = open(filename_data, "a+")
file1.write('\n\n' + current_test)
file1.close()

for c in [0,1,2,3,4,5,6,7,8,9]:

    #Cardinality of set of alternatives
    n_elements = 500
    #Size of subsets to evaluate
    k_size = 2
    #Number of criteria
    n_criteria = 4

    # Id of criteria (vector w)
    vars = list([])
    for i in range(n_criteria):
        vars.append('w' + str(i))

    #Parameters for random instance generator. See documentation for details
    #Number of random constraints (input user preferences)
    n_constraints = c
    #Number of sets of alternatives to generate
    nSets = 1


    #Total execution time per repetition
    total_time_list__SAT_EPI = []
    #Lenght of W' per iteration
    len_wp_list__SAT_EPI = []
    #Time spent to compute all SMR per repetition
    SMR_time_list__SAT_EPI = []
    #Number of SMR calls per repetition
    SMR_calls_list__SAT_EPI = []
    #Time spent to solve SAT problems per repetition
    SAT_time_list__SAT_EPI = []
    #Time spent to solve SAT problems per repetition
    UD_size_list__SAT_EPI = []

    for i in range(repetitions):
        print(current_test)

        # generate random set
        ranodm_sets, formatted_constraints, polytope = \
            RandomProblem.random_problem_UD(vars, n_elements, nSets, n_constraints, int_values=False)

        A = ranodm_sets[0]
        A = SolutionsUtils.UD(A, polytope)


        start_time = time.time()
        SMMR__SAT_EPI, B_star__SAT_EPI, it__SAT_EPI, best_single_MR__SAT_EPI, \
        A_ordered__SAT_EPI, len_wp__SAT_EPI, SMR_tot_time__SAT_EPI, tot_SMR_calls__SAT_EPI, SAT_tot_time__SAT_EPI = \
            SMMR.SMMR_SAT_EPI(k_size, A, polytope)
        time__SAT_EPI = time.time() - start_time

        total_time_list__SAT_EPI.append(time__SAT_EPI)
        SMR_time_list__SAT_EPI.append(SMR_tot_time__SAT_EPI)
        SMR_calls_list__SAT_EPI.append(float(tot_SMR_calls__SAT_EPI))
        SAT_time_list__SAT_EPI.append(SAT_tot_time__SAT_EPI)
        len_wp_list__SAT_EPI.append(float(len_wp__SAT_EPI))
        UD_size_list__SAT_EPI.append(float(len(A)))

        # store results
        data = 'Iteration ' + str(i) + "\n"
        data += 'Database: ' + str(A) + '\n'
        data += '|A|: ' + str(n_elements) + '\n'
        data += 'k: ' + str(k_size) + '\n'
        data += 'p: ' + str(n_criteria) + '\n'
        data += 'Number of random constraints: ' + str(n_constraints) + '\n'
        data += 'Timestamp: ' + str(time.time()) + '\n'
        data += '\n'
        data += "time_SAT__SAT_EPI: " + str(time__SAT_EPI) + '\n'
        data += "SMMR__SAT_EPI: " + str(SMMR__SAT_EPI) + '\n'
        data += "len_wp__SAT_EPI: " + str(len_wp__SAT_EPI) + '\n'
        data += "SAT_tot_time__SAT_EPI: " + str(SAT_tot_time__SAT_EPI) + '\n'
        data += "SMR_tot_time__SAT_EPI: " + str(SMR_tot_time__SAT_EPI) + '\n'
        data += "tot_SMR_calls__SAT_EPI: " + str(tot_SMR_calls__SAT_EPI) + '\n'
        data += "|UD_W(A)|: " + str(len(A)) + '\n'
        print(data)


        file1 = open(filename_data, "a+")
        file1.write('\n' + data)
        file1.close()


    #string formatted for latex
    res = str(i) + ' & ' + str(n_elements) + ' & ' + str(k_size) + ' & ' + \
          str(n_criteria) + ' & ' + str(n_constraints) + ' & ' + \
                str(round(statistics.mean(total_time_list__SAT_EPI),3))  + ' & ' + \
                str(round(statistics.mean(len_wp_list__SAT_EPI),3)) + ' & ' + \
                str(round(statistics.mean(SMR_time_list__SAT_EPI),3)) + ' & ' + \
                str(round(statistics.mean(SMR_calls_list__SAT_EPI),3)) + ' & ' + \
                str(round(statistics.mean(SAT_time_list__SAT_EPI),3)) + ' & ' + \
                str(round(statistics.mean(UD_size_list__SAT_EPI),3))  +  \
              '\\\\ \n \\hline'

    file1 = open(filename, "a+")
    file1.write('\n' + res)
    file1.close()



#Experiment header
current_test = 'Table3'
file1 = open(filename, "a+")
file1.write('\n\n' + current_test)
file1.close()
file1 = open(filename_data, "a+")
file1.write('\n\n' + current_test)
file1.close()

for n in [100,200,300,400,500,600]:

    #Cardinality of set of alternatives
    n_elements = n
    #Size of subsets to evaluate
    k_size = 2
    #Number of criteria
    n_criteria = 4

    # Id of criteria (vector w)
    vars = list([])
    for i in range(n_criteria):
        vars.append('w' + str(i))

    #Parameters for random instance generator. See documentation for details
    #Number of random constraints (input user preferences)
    n_constraints = 0
    #Number of sets of alternatives to generate
    nSets = 1


    #Total execution time per repetition
    total_time_list__SAT_EPI = []
    #Lenght of W' per iteration
    len_wp_list__SAT_EPI = []
    #Time spent to compute all SMR per repetition
    SMR_time_list__SAT_EPI = []
    #Number of SMR calls per repetition
    SMR_calls_list__SAT_EPI = []
    #Time spent to solve SAT problems per repetition
    SAT_time_list__SAT_EPI = []

    for i in range(repetitions):
        print(current_test)

        # generate random set
        A = []
        ranodm_sets, formatted_constraints, polytope = \
            RandomProblem.random_problem_UD(vars, n_elements, nSets, n_constraints, int_values=False)
        A = ranodm_sets[0]

        start_time = time.time()
        SMMR__SAT_EPI, B_star__SAT_EPI, it__SAT_EPI, best_single_MR__SAT_EPI, \
        A_ordered__SAT_EPI, len_wp__SAT_EPI, SMR_tot_time__SAT_EPI, tot_SMR_calls__SAT_EPI, SAT_tot_time__SAT_EPI = \
            SMMR.SMMR_SAT_EPI(k_size, A, polytope)
        time__SAT_EPI = time.time() - start_time

        total_time_list__SAT_EPI.append(time__SAT_EPI)
        SMR_time_list__SAT_EPI.append(SMR_tot_time__SAT_EPI)
        SMR_calls_list__SAT_EPI.append(float(tot_SMR_calls__SAT_EPI))
        SAT_time_list__SAT_EPI.append(SAT_tot_time__SAT_EPI)
        len_wp_list__SAT_EPI.append(len_wp__SAT_EPI)

        # store results
        data = 'Iteration ' + str(i) + "\n"
        data += 'Database: ' + str(A) + '\n'
        data += '|A|: ' + str(n_elements) + '\n'
        data += 'k: ' + str(k_size) + '\n'
        data += 'p: ' + str(n_criteria) + '\n'
        data += 'Number of random constraints: ' + str(n_constraints) + '\n'
        data += 'Timestamp: ' + str(time.time()) + '\n'
        data += '\n'
        print(data)
        data += "time_SAT__SAT_EPI: " + str(time__SAT_EPI) + '\n'
        data += "SMMR__SAT_EPI: " + str(SMMR__SAT_EPI) + '\n'
        data += "len_wp__SAT_EPI: " + str(len_wp__SAT_EPI) + '\n'
        data += "SAT_tot_time__SAT_EPI: " + str(SAT_tot_time__SAT_EPI) + '\n'
        data += "SMR_tot_time__SAT_EPI: " + str(SMR_tot_time__SAT_EPI) + '\n'
        data += "tot_SMR_calls__SAT_EPI: " + str(tot_SMR_calls__SAT_EPI) + '\n'


        file1 = open(filename_data, "a+")
        file1.write('\n' + data)
        file1.close()


    #string formatted for latex
    res = str(i) + ' & ' + str(n_elements) + ' & ' + str(k_size) + ' & ' + \
          str(n_criteria) + ' & ' + str(n_constraints) + ' & ' + \
                str(round(statistics.mean(total_time_list__SAT_EPI),3)) + ' & ' + \
                str(round(statistics.mean(len_wp_list__SAT_EPI),3)) + ' & ' + \
                str(round(statistics.mean(SMR_time_list__SAT_EPI),3)) + ' & ' + \
                str(round(statistics.mean(SMR_calls_list__SAT_EPI),3)) + ' & ' + \
                str(round(statistics.mean(SAT_time_list__SAT_EPI),3)) +  \
              '\\\\ \n \\hline'

    file1 = open(filename, "a+")
    file1.write('\n' + res)
    file1.close()



#Experiment header
current_test = 'Figure3 & Table 4'
file1 = open(filename, "a+")
file1.write('\n\n' + current_test)
file1.close()
file1 = open(filename_data, "a+")
file1.write('\n\n' + current_test)
file1.close()

for p in [2,3,4,5,6]:
    for k in [2,3,4,5,6]:

        #Cardinality of set of alternatives
        n_elements = 100
        #Size of subsets to evaluate
        k_size = k
        #Number of criteria
        n_criteria = p

        # Id of criteria (vector w)
        vars = list([])
        for i in range(n_criteria):
            vars.append('w' + str(i))

        #Parameters for random instance generator. See documentation for details
        #Number of random constraints (input user preferences)
        n_constraints = 0
        #Number of sets of alternatives to generate
        nSets = 1


        #Total execution time per repetition
        total_time_list__SAT_EPI = []
        #Lenght of W' per iteration
        len_wp_list__SAT_EPI = []
        #Time spent to compute all SMR per repetition
        SMR_time_list__SAT_EPI = []
        #Number of SMR calls per repetition
        SMR_calls_list__SAT_EPI = []
        #Time spent to solve SAT problems per repetition
        SAT_time_list__SAT_EPI = []

        for i in range(repetitions):
            print(current_test)

            # generate random set
            A = []
            ranodm_sets, formatted_constraints, polytope = \
                RandomProblem.random_problem_UD(vars, n_elements, nSets, n_constraints, int_values=False)
            A = ranodm_sets[0]

            start_time = time.time()
            SMMR__SAT_EPI, B_star__SAT_EPI, it__SAT_EPI, best_single_MR__SAT_EPI, \
            A_ordered__SAT_EPI, len_wp__SAT_EPI, SMR_tot_time__SAT_EPI, tot_SMR_calls__SAT_EPI, SAT_tot_time__SAT_EPI = \
                SMMR.SMMR_SAT_EPI(k_size, A, polytope)
            time__SAT_EPI = time.time() - start_time

            total_time_list__SAT_EPI.append(time__SAT_EPI)
            SMR_time_list__SAT_EPI.append(SMR_tot_time__SAT_EPI)
            SMR_calls_list__SAT_EPI.append(float(tot_SMR_calls__SAT_EPI))
            SAT_time_list__SAT_EPI.append(SAT_tot_time__SAT_EPI)
            len_wp_list__SAT_EPI.append(len_wp__SAT_EPI)

            # store results
            data = 'Iteration ' + str(i) + "\n"
            data += 'Database: ' + str(A) + '\n'
            data += '|A|: ' + str(n_elements) + '\n'
            data += 'k: ' + str(k_size) + '\n'
            data += 'p: ' + str(n_criteria) + '\n'
            data += 'Number of random constraints: ' + str(n_constraints) + '\n'
            data += 'Timestamp: ' + str(time.time()) + '\n'
            data += '\n'
            print(data)
            data += "time_SAT__SAT_EPI: " + str(time__SAT_EPI) + '\n'
            data += "SMMR__SAT_EPI: " + str(SMMR__SAT_EPI) + '\n'
            data += "len_wp__SAT_EPI: " + str(len_wp__SAT_EPI) + '\n'
            data += "SAT_tot_time__SAT_EPI: " + str(SAT_tot_time__SAT_EPI) + '\n'
            data += "SMR_tot_time__SAT_EPI: " + str(SMR_tot_time__SAT_EPI) + '\n'
            data += "tot_SMR_calls__SAT_EPI: " + str(tot_SMR_calls__SAT_EPI) + '\n'


            file1 = open(filename_data, "a+")
            file1.write('\n' + data)
            file1.close()


        #string formatted for latex
        res = str(i) + ' & ' + str(n_elements) + ' & ' + str(k_size) + ' & ' + \
              str(n_criteria) + ' & ' + str(n_constraints) + ' & ' + \
                    str(round(statistics.mean(total_time_list__SAT_EPI),3)) + ' & ' + \
                    str(round(statistics.mean(len_wp_list__SAT_EPI),3)) + ' & ' + \
                    str(round(statistics.mean(SMR_time_list__SAT_EPI),3)) + ' & ' + \
                    str(round(statistics.mean(SMR_calls_list__SAT_EPI),3)) + ' & ' + \
                    str(round(statistics.mean(SAT_time_list__SAT_EPI),3)) +  \
                  '\\\\ \n \\hline'

        file1 = open(filename, "a+")
        file1.write('\n' + res)
        file1.close()





#Experiment header
current_test = 'Table1'
file1 = open(filename, "a+")
file1.write('\n\n' + current_test)
file1.close()
file1 = open(filename_data, "a+")
file1.write('\n\n' + current_test)
file1.close()

for k in [2,3]:
    for p in [3,4]:

        #Cardinality of set of alternatives
        n_elements = 50
        #Size of subsets to evaluate
        k_size = k
        #Number of criteria
        n_criteria = p

        # Id of criteria (vector w)
        vars = list([])
        for i in range(n_criteria):
            vars.append('w' + str(i))

        #Parameters for random instance generator. See documentation for details
        #Number of random constraints (input user preferences)
        n_constraints = 0
        #Number of sets of alternatives to generate
        nSets = 1


        #Total execution time per repetition
        total_time_list__SAT_EPI = []
        total_time_list__SAT_LP = []
        total_time_list__BF_EPI = []
        total_time_list__BF_LP = []
        #Lenght of W' per iteration
        len_wp_list__SAT_EPI = []
        len_wp_list__SAT_LP = []
        #Time spent to compute all SMR per repetition
        SMR_time_list__SAT_EPI = []
        SMR_time_list__SAT_LP = []
        #Number of SMR calls per repetition
        SMR_calls_list__SAT_EPI = []
        SMR_calls_list__SAT_LP = []
        #Time spent to solve SAT problems per repetition
        SAT_time_list__SAT_EPI = []
        SAT_time_list__SAT_LP = []

        for i in range(repetitions):
            print(current_test)

            # generate random set
            A = []
            ranodm_sets, formatted_constraints, polytope = \
                RandomProblem.random_problem_UD(vars, n_elements, nSets, n_constraints, int_values=False)
            A = ranodm_sets[0]


            print('SAT_EPI')
            start_time = time.time()
            SMMR__SAT_EPI, B_star__SAT_EPI, it__SAT_EPI, best_single_MR__SAT_EPI, \
            A_ordered__SAT_EPI, len_wp__SAT_EPI, SMR_tot_time__SAT_EPI, tot_SMR_calls__SAT_EPI, SAT_tot_time__SAT_EPI = \
                SMMR.SMMR_SAT_EPI(k_size, A, polytope)
            time__SAT_EPI = time.time() - start_time

            total_time_list__SAT_EPI.append(time__SAT_EPI)
            SMR_time_list__SAT_EPI.append(SMR_tot_time__SAT_EPI)
            SMR_calls_list__SAT_EPI.append(float(tot_SMR_calls__SAT_EPI))
            SAT_time_list__SAT_EPI.append(SAT_tot_time__SAT_EPI)
            len_wp_list__SAT_EPI.append(len_wp__SAT_EPI)


            print('SAT_LP')
            start_time = time.time()
            SMMR__SAT_LP, B_star__SAT_LP, it__SAT_LP, best_single_MR__SAT_LP, \
            A_ordered__SAT_LP, len_wp__SAT_LP, SMR_tot_time__SAT_LP, tot_SMR_calls__SAT_LP, SAT_tot_time__SAT_LP = \
                SMMR.SMMR_SAT_LP(k_size, A, polytope)
            time__SAT_LP = time.time() - start_time

            total_time_list__SAT_LP.append(time__SAT_LP)
            SMR_time_list__SAT_LP.append(SMR_tot_time__SAT_LP)
            SMR_calls_list__SAT_LP.append(float(tot_SMR_calls__SAT_LP))
            SAT_time_list__SAT_LP.append(SAT_tot_time__SAT_LP)
            len_wp_list__SAT_LP.append(len_wp__SAT_LP)


            print('BF_EPI')
            start_time = time.time()
            SMMR__BF_EPI, B_star__BF_EPI, it__BF_EPI = SMMR.SMMR_BF_EPI(k_size, A, polytope)
            time__BF_EPI = time.time() - start_time
            total_time_list__BF_EPI.append(time__BF_EPI)


            print('BF_LP')
            start_time = time.time()
            SMMR__BF_LP, B_star__BF_LP, it__BF_LP = SMMR.SMMR_BF_LP(k_size, A, polytope)
            time__BF_LP = time.time() - start_time
            total_time_list__BF_LP.append(time__BF_LP)

            # store results
            data = 'Iteration ' + str(i) + "\n"
            data += 'Database: ' + str(A) + '\n'
            data += '|A|: ' + str(n_elements) + '\n'
            data += 'k: ' + str(k_size) + '\n'
            data += 'p: ' + str(n_criteria) + '\n'
            data += 'Number of random constraints: ' + str(n_constraints) + '\n'
            data += 'Timestamp: ' + str(time.time()) + '\n'
            data += '\n'
            print(data)
            data += "time_SAT__SAT_EPI: " + str(time__SAT_EPI) + '\n'
            data += "SMMR__SAT_EPI: " + str(SMMR__SAT_EPI) + '\n'
            data += "len_wp__SAT_EPI: " + str(len_wp__SAT_EPI) + '\n'
            data += "SAT_tot_time__SAT_EPI: " + str(SAT_tot_time__SAT_EPI) + '\n'
            data += "SMR_tot_time__SAT_EPI: " + str(SMR_tot_time__SAT_EPI) + '\n'
            data += "tot_SMR_calls__SAT_EPI: " + str(tot_SMR_calls__SAT_EPI) + '\n'
            data += '\n'
            data += "time_SAT__SAT_LP: " + str(time__SAT_LP) + '\n'
            data += "SMMR__SAT_LP: " + str(SMMR__SAT_LP) + '\n'
            data += "len_wp__SAT_LP: " + str(len_wp__SAT_LP) + '\n'
            data += "SAT_tot_time__SAT_LP: " + str(SAT_tot_time__SAT_LP) + '\n'
            data += "SMR_tot_time__SAT_LP: " + str(SMR_tot_time__SAT_LP) + '\n'
            data += "tot_SMR_calls__SAT_LP: " + str(tot_SMR_calls__SAT_LP) + '\n'
            data += '\n'
            data += "time_SAT__BF_EPI: " + str(time__BF_EPI) + '\n'
            data += "SMMR__BF_EPI: " + str(SMMR__BF_EPI) + '\n'
            data += '\n'
            data += "time_SAT__BF_LP: " + str(time__BF_LP) + '\n'
            data += "SMMR__BF_LP: " + str(SMMR__BF_LP) + '\n'


            file1 = open(filename_data, "a+")
            file1.write('\n' + data)
            file1.close()


        #string formatted for latex
        res = str(i) + ' & ' + str(n_elements) + ' & ' + str(k_size) + ' & ' + \
              str(n_criteria) + ' & ' + str(n_constraints) + ' & ' + \
                    str(round(statistics.mean(total_time_list__SAT_EPI),3)) + ' & ' + \
                    str(round(statistics.mean(total_time_list__SAT_LP),3)) + ' & ' + \
                    str(round(statistics.mean(total_time_list__BF_EPI),3)) + ' & ' + \
                    str(round(statistics.mean(total_time_list__BF_LP),3)) + \
                  '\\\\ \n \\hline'

        file1 = open(filename, "a+")
        file1.write('\n' + res)
        file1.close()
