import numpy as np
import knnpredict

def run(data, training_data, validation_data, training_result, validation_result):
    lowest_error = 5
    best_k = 0

    for v in range(len(data)):

        answer = knnpredict.run(training_data, validation_data, training_result, data[v])
        temp_error = np.mean(validation_result != answer)

        if temp_error < lowest_error:
            lowest_error = temp_error
            best_k = data[v]

    return best_k
