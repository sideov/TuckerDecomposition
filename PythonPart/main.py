import numpy as np

def svd_from_scratch(matrix):
    """
    Выполняет SVD-разложение матрицы с использованием только NumPy.

    :param matrix: Входная матрица формы (m, n)
    :return: U, Sigma, Vt
        U: Левые сингулярные векторы, форма (m, m)
        Sigma: Диагональная матрица сингулярных значений, форма (min(m, n), min(m, n))
        Vt: Транспонированная правая сингулярная матрица, форма (n, n)
    """
    matrix = matrix.T  # Транспонируем матрицу

    # Шаг 1: Вычисляем ковариационную матрицу
    covariance_matrix = np.dot(matrix.T, matrix)

    # Шаг 2: Вычисляем собственные значения и собственные векторы ковариационной матрицы
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Шаг 3: Вычисляем сингулярные значения и сортируем их в порядке убывания
    singular_values = np.sqrt(eigenvalues)
    sorted_indices = np.argsort(singular_values)[::-1]
    singular_values = singular_values[sorted_indices]

    # Шаг 4: Вычисляем левые сингулярные векторы (U)
    U = eigenvectors[:, sorted_indices]

    # Шаг 5: Вычисляем правые сингулярные векторы (Vt)
    Vt = np.dot(np.linalg.inv(np.diag(singular_values)), np.dot(U.T, matrix.T))

    return U, np.diag(singular_values), Vt

def unfold(X, n):
    """
    Разворачивает матрицу X по указанной моде n.

    :param X: Тензор
    :param n: Режим разворачивания
    :return: Развернутая матрица
    """
    return np.reshape(np.moveaxis(X, n, 0), (X.shape[n], -1))

def refold(X, n, shape):
    """
    Обратная операция разворачивания. Восстанавливает тензор по развернутой матрице и указанной форме.

    :param X: Развернутая матрица
    :param n: Режим разворачивания
    :param shape: Форма исходного тензора
    :return: Восстановленный тензор
    """
    shape = list(shape)
    mode_dim = shape.pop(n)
    shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(X, shape), 0, n)

def mode(X, Y, n):
    """
    Скалярное произведение матрицы X и матрицы Y по указанной моде n.

    :param X: Тензор
    :param Y: Матрица
    :param n: Режим произведения
    :return: Результат многодинового произведения
    """
    shape = list(X.shape)
    shape[n] = Y.shape[0]
    res = np.dot(Y, unfold(X, n))
    return refold(res, n, shape)

def HOSVD(X):
    """
    Выполняет HOSVD (Higher Order Singular Value Decomposition) для тензора X.

    :param X: Входной тензор
    :return: Ядро тензора G и список матриц As, полученных SVD по каждой моде
    """
    As = []
    for n in range(X.ndim):
        A, _v, _M = svd_from_scratch(unfold(X, n))
        As.append(A)

    # Вычисляем ядро тензора G
    G = X
    for i, A in enumerate(As):
        G = mode(G, A.T, i)

    return G, As

def HOOI(X):
    """
    Выполняет HOOI (Higher Order Orthogonal Iteration) для тензора X.

    :param X: Входной тензор
    :return: Ядро тензора G и список матриц As
    """
    _, As = HOSVD(X)

    for _ in range(100):
        for n in range(len(As)):
            Y = X
            for i, A in enumerate(As):
                if n == i:
                    continue
                Y = mode(Y, A.T, i)
            A, _v, _m = svd_from_scratch(unfold(Y, n))
            As[n] = A

    # Вычисляем ядро тензора G
    G = X
    for i, A in enumerate(As):
        G = mode(G, A.T, i)

    return G, As

if __name__ == "__main__":
    A = np.arange(1,25).reshape(2,3,4)
    print(A)
    print("---")
    print(unfold(A, 1))
