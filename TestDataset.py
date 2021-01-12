from source import SMMR, SolutionsUtils, Polytope
import time
import datetime
import csv


datasets=[]

dateTimeObj = datetime.datetime.now()

filename = 'dataset/' + str('synthetic.dat')
datasets.append(filename)

filename_out = 'experiments/dataset_synthetic_' + str(dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S"))

for k_size in [2, 3, 4, 5]:

    print('------')
    A = []
    size = float('inf')

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        for row in csv_reader:
            # print(row)
            A.append(row)
            line_count += 1
            if line_count > size:
                break
        print(f'Processed {line_count} lines.')

    d = len(A[0])
    vars = []
    for i in range(d):
        vars.append('w' + str(i))
    W = Polytope.Polytope(vars, frac=True)

    start_time = time.time()
    A = SolutionsUtils.UD(A, W)
    UD_time = time.time() - start_time

    UD_size = len(A)
    print('UD size:' + str(len(A)))
    print(A)

    start_time = time.time()
    SMMR__SAT_EPI, B_star__SAT_EPI, it__SAT_EPI, best_single_MR__SAT_EPI, \
    A_ordered__SAT_EPI, len_wp__SAT_EPI, SMR_tot_time__SAT_EPI, tot_SMR_calls__SAT_EPI, \
    SAT_solver_time__SAT_EPI = SMMR.SMMR_SAT_EPI(k_size, A, W)
    time__SAT_EPI = time.time() - start_time

    data = '--------' + '\n'
    data += 'Dataset: ' + str(filename) + '\n'
    data += 'k: ' + str(k_size) + '\n'
    data += 'SMMR tot time EPI: ' + str(round(time__SAT_EPI, 3)) + '\n'
    data += 'SMMR EPI: ' + str(round(SMMR__SAT_EPI, 3)) + '\n'

    data += 'SAT solver tot time EPI: ' + str(round(SAT_solver_time__SAT_EPI, 3)) + '\n'

    data += 'SMR tot time EPI: ' + str(round(SMR_tot_time__SAT_EPI, 3)) + '\n'
    data += 'tot SMR calls EPI: ' + str(round(tot_SMR_calls__SAT_EPI, 3)) + '\n'

    data += 'Len W\' EPI: ' + str(round(len_wp__SAT_EPI, 3)) + '\n'

    data += 'UD size: ' + str(round(UD_size, 3)) + '\n'
    data += 'UD time: ' + str(round(UD_time, 3)) + '\n'
    data += '\n'

    print(data)


    file1 = open(filename_out, "a+")
    file1.write('\n' + data)
    file1.close()



filename = 'dataset/' + str('apt187_dataset.dat')
datasets.append(filename)

filename_out = 'experiments/dataset_apt187_' + str(dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S"))

for k_size in [2,3,4,5]:
    A=[]
    size =float('inf')

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        for row in csv_reader:
            # print(row)
            A.append(row)
            line_count+=1
            if line_count > size:
                break
        print(f'Processed {line_count} lines.')

    d=len(A[0])
    vars=[]
    for i in range(d):
        vars.append('w' + str(i))
    W = Polytope.Polytope(vars, frac=True)

    start_time = time.time()
    A = SolutionsUtils.UD(A, W)
    UD_time= time.time() - start_time

    UD_size = len(A)
    print('UD size:' + str(len(A)))
    print(A)


    start_time = time.time()
    SMMR__SAT_EPI, B_star__SAT_EPI, it__SAT_EPI, best_single_MR__SAT_EPI, \
    A_ordered__SAT_EPI, len_wp__SAT_EPI, SMR_tot_time__SAT_EPI, tot_SMR_calls__SAT_EPI, \
        SAT_solver_time__SAT_EPI = SMMR.SMMR_SAT_EPI(k_size, A, W)
    time__SAT_EPI = time.time() - start_time

    data = '--------' + '\n'
    data += 'Dataset: ' + str(filename)  + '\n'
    data += 'k: ' + str(k_size) + '\n'
    data += 'SMMR tot time EPI: ' + str(round(time__SAT_EPI,3)) + '\n'
    data += 'SMMR EPI: ' + str(round(SMMR__SAT_EPI,3)) + '\n'

    data += 'SAT solver tot time EPI: ' + str(round(SAT_solver_time__SAT_EPI,3)) + '\n'

    data += 'SMR tot time EPI: ' + str(round(SMR_tot_time__SAT_EPI,3)) + '\n'
    data += 'tot SMR calls EPI: ' + str(round(tot_SMR_calls__SAT_EPI,3)) + '\n'

    data += 'Len W\' EPI: ' + str(round(len_wp__SAT_EPI,3)) + '\n'

    data += 'UD size: ' + str(round(UD_size,3)) + '\n'
    data += 'UD time: ' + str(round(UD_time,3)) + '\n'
    data += '\n'

    print(data)

    file1 = open(filename_out, "a+")
    file1.write('\n' + data)
    file1.close()


filename = 'dataset/' + str('bostonHousing.dat')

filename_out = 'experiments/dataset_boston_' + str(dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S"))

for k_size in [2,3]:

    A=[]
    size =float('inf')

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        for row in csv_reader:
            # print(row)
            A.append(row)
            line_count+=1
            if line_count > size:
                break
        print(f'Processed {line_count} lines.')

    d=len(A[0])
    vars=[]
    for i in range(d):
        vars.append('w' + str(i))
    W = Polytope.Polytope(vars, frac=True)


    start_time = time.time()
    A = SolutionsUtils.UD(A, W)
    UD_time= time.time() - start_time

    UD_size = len(A)
    print('UD size:' + str(len(A)))
    print(A)


    start_time = time.time()
    SMMR__SAT_EPI, B_star__SAT_EPI, it__SAT_EPI, best_single_MR__SAT_EPI, \
    A_ordered__SAT_EPI, len_wp__SAT_EPI, SMR_tot_time__SAT_EPI, tot_SMR_calls__SAT_EPI, \
        SAT_solver_time__SAT_EPI = SMMR.SMMR_SAT_EPI(k_size, A, W)
    time__SAT_EPI = time.time() - start_time


    data = '--------'  + '\n'
    data += 'Dataset: ' + str(filename) + '\n'
    data += 'k: ' + str(k_size) + '\n'
    data += 'SMMR tot time EPI: ' + str(round(time__SAT_EPI,3)) + '\n'
    data += 'SMMR EPI: ' + str(round(SMMR__SAT_EPI,3)) + '\n'

    data += 'SAT solver tot time EPI: ' + str(round(SAT_solver_time__SAT_EPI,3)) + '\n'

    data += 'SMR tot time EPI: ' + str(round(SMR_tot_time__SAT_EPI,3)) + '\n'
    data += 'tot SMR calls EPI: ' + str(round(tot_SMR_calls__SAT_EPI,3)) + '\n'

    data += 'Len W\' EPI: ' + str(round(len_wp__SAT_EPI,3)) + '\n'

    data += 'UD size: ' + str(round(UD_size,3)) + '\n'
    data += 'UD time: ' + str(round(UD_time,3)) + '\n'
    data += '\n'

    print(data)

    file1 = open(filename_out, "a+")
    file1.write('\n' + data)
    file1.close()
