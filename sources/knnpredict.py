import numpy as np
import numpy.linalg as la
def run(training, testing, training_result, param):

    default = np.zeros((param, 2))
    test_rows = len(testing)
    train_rows = len(training)
    answer_array = np.zeros(test_rows)

    for i in range(test_rows):
        sorter = False
        for f in range(param):
            default[f,0] = la.norm(testing[i] - training[f])
            default[f,1] = training_result[f]
        sorted_array = default[default[:, 0].argsort()[::-1]]
        for g in range(train_rows):
            value = la.norm(testing[i] - training[g])
            if value < sorted_array[0,0]:
                sorter = True
                sorted_array[0,0] = value
                sorted_array[0,1] = training_result[g]
            if sorter:
                sorted_array = sorted_array[sorted_array[:, 0].argsort()[::-1]]
        zero_val = 0
        one_val = 0
        for k in range(param):
            if sorted_array[k,1] == 0:
                zero_val = zero_val + 1
            else:
                one_val = one_val + 1
        if one_val > zero_val:
            answer_array[i] = 1


    return answer_array