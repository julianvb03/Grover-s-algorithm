import numpy as np
from typing import List, Tuple, Set, Int

"""
===========================
Archivo: main.py 
Descripción: Este archivo tiene la definición de las funciones necesarias para
implementar el algoritmo de grover.

El cual consta de los siguientes pasos:
1. Superposición inicial, aplicando la puerta Hadamard a todos los qubits.
2. Aplicar la función de oráculo.
3. Aplicar la difusión de Grover.
    - Aplicar la puerta Hadamard a todos los qubits.
    - Un desplazamiento de fase condicional de -1 aplicado a todos los estados
        de la base computacional menos |0>.
    - Aplicar la puerta Hadamard a todos los qubits.
4. Repetir los pasos 2 y 3 un número de veces determinado por la raíz cuadrada del número de elementos en la lista.
5. Medir el estado final de los qubits.

Autor: Julian valencia

Funciones:
    - mi_funcion(): Descripción breve.

Constantes:
    - MI_CONSTANTE: Valor y descripción.
"""

############# Funciones para validación ################
def debbug_print(state: np.ndarray, hadamard: np.ndarray) -> None:
    """
    Imprime el estado cuántico y la matriz de Hadamard.
    
    Parámetros:
    state (np.ndarray): Estado cuántico a imprimir.
    hadamard (np.ndarray): Matriz de Hadamard a imprimir.
    """
    print("Estado cuántico:")
    print(state)
    print("Matriz de Hadamard:")
    print(hadamard)
    print("Validación de amplitud:", amplitud_validation(state))


def amplitud_validation(state: np.ndarray) -> bool:
    """
    Verifica si la amplitud de un estado cuántico es válida.
    
    Parámetros:
    state (np.ndarray): Estado cuántico a verificar.

    Retorna:
    bool: True si la amplitud es válida, False en caso contrario.
    """
    return np.isclose(np.sum(np.abs(state)**2), 1)

########################################################

def hadamard_n_dimensional(n: int) -> np.ndarray:
    """
    Genera la matriz de Hadamard de n dimensiones.
    Definida de la siguiente manera: https://en.wikipedia.org/wiki/Hadamard_transform
    
    Parámetros:
    n (int): Número de qubits.

    Retorna:
    np.ndarray: Matriz de Hadamard de n dimensiones.
    """
    if n == 0:
        return np.array([[1]])
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for _ in range(1, n):
        H = np.kron(H, np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    return H

def oracle(state: np.ndarray, dimension: int, targets: Set[Tuple[int, int]]) -> np.ndarray:
    """
    Aplica la función de oráculo a un estado cuántico, la cual consiste en cambiar el signo de aquellos
    estados que estemos buscando.
    
    Parámetros:
    state (np.ndarray): Estado cuántico a modificar.

    Retorna:
    np.ndarray: Estado cuántico modificado.
    """
    oracle = np.eye(N, dtype=complex)
    for target in targets:
        oracle[target[0],target[1]] = -oracle[target[0],target[1]]


    return state

if __name__ == "__main__":
    n = input("Ingrese el número de qubits: ")
    N = 2**int(n)

    # Todos los qubits en estado |0>
    initial_state = np.zeros(N, dtype=complex)
    initial_state[0] = 1  # |000...0>
    hadamart_n = hadamard_n_dimensional(int(n))

    # Aplicar la matriz de Hadamard al estado inicial
    final_state = np.dot(hadamart_n, initial_state)
    debbug_print(final_state, hadamart_n)
