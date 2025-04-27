from main import hadamard_n_dimensional
import numpy as np
def test_hadamard_0_dimensional():
    """
    Test para la función hadamard_n_dimensional con 0 qubits.
    Se espera que la matriz de Hadamard sea [[1]].
    """
    result = hadamard_n_dimensional(0)
    expected = np.array([[1]])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_hadamard_1_dimensional():
    """
    Test para la función hadamard_n_dimensional con 1 qubit.
    Se espera que la matriz de Hadamard sea [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]].
    """
    result = hadamard_n_dimensional(1)
    amplitud = 1 / np.sqrt(2)
    expected = amplitud * np.array([
        [1,  1],
        [1, -1],
    ])
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_hadamard_2_dimensional():
    """
    Test para la función hadamard_n_dimensional con 2 qubits.
    Se espera que la matriz de Hadamard sea [[1/2, 1/2, 1/2, 1/2], [1/2, -1/2, 1/2, -1/2], [1/2, 1/2, -1/2, -1/2], [1/2, -1/2, -1/2, 1/2]].
    """
    result = hadamard_n_dimensional(2)
    amplitud = 1 / 2  # porque 2^2 = 4
    expected = amplitud * np.array([
        [ 1,  1,  1,  1],
        [ 1, -1,  1, -1],
        [ 1,  1, -1, -1],
        [ 1, -1, -1,  1],
    ])
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_hadamard_3_dimensional():
    """
    Test para la función hadamard_n_dimensional con 3 qubits.
    Se espera que la matriz de Hadamard sea de tamaño 8x8 y tenga los valores correctos.
    """
    result = hadamard_n_dimensional(3)
    amplitud = 1 / np.sqrt(8) # Que es lo mismo que 1/sqrt(2^3) como se puede ver en la pagina de la transformada de hadamard
    expected = amplitud * np.array([
        [ 1,  1,  1,  1,  1,  1,  1,  1],
        [ 1, -1,  1, -1,  1, -1,  1, -1],
        [ 1,  1, -1, -1,  1,  1, -1, -1],
        [ 1, -1, -1,  1,  1, -1, -1,  1],
        [ 1,  1,  1,  1, -1, -1, -1, -1],
        [ 1, -1,  1, -1, -1,  1, -1,  1],
        [ 1,  1, -1, -1, -1, -1,  1,  1],
        [ 1, -1, -1,  1, -1,  1,  1, -1],
    ])
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"