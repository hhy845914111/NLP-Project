from pathos.multiprocessing import ProcessingPool
import numpy as np
from copy import deepcopy
from tqdm import tqdm


def default_evaluate(rst_lst):
    return np.hstack([ctt[1] for ctt in rst_lst])


def rolling_test(model, X, y, train_count, evaluate_func=default_evaluate):
    _Pool = ProcessingPool()

    def _task(args):
        model, X_train_mat, y_train_mat, X_test_mat = args
        model.fit(X_train_mat, y_train_mat)
        return model.predict(X_train_mat), model.predict(X_test_mat)

    rst_lst = _Pool.map(_task, ((deepcopy(model), X[i - train_count:i, :], y[i - train_count:i], X[i, :].reshape(1, -1)) for i in range(train_count, len(y))))

    return np.hstack((np.zeros(train_count, ), evaluate_func(rst_lst)))


def rolling_test_single(model, X, y, train_count, evaluate_func=default_evaluate):

    def _task(args):
        model, X_train_mat, y_train_mat, X_test_mat = args
        model.fit(X_train_mat, y_train_mat)
        return model.predict(X_train_mat), model.predict(X_test_mat)

    rst_lst = list(map(_task, ((deepcopy(model), X[i - train_count:i, :], y[i - train_count:i], X[i, :].reshape(1, -1)) for i in range(train_count, len(y)))))

    return np.hstack((np.zeros(train_count, ), evaluate_func(rst_lst)))


if __name__ == "__main__":
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    X = np.random.rand(80000, 100)
    y = np.random.randint(-1, 1, (80000, 1))

    model = DecisionTreeClassifier()

    tt = rolling_test_single(model, X, y, 100)
    print(tt)

    print(1)