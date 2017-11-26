from svm_port.svmutil import *

if __name__ == '__main__':
    y, x = svm_read_problem('svm_port/heart_scale')
    m = svm_train(y[:200], x[:200], '-c 4')
    p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)
