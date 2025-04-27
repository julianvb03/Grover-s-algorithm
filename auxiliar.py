import numpy as np
from typing import Set

def debug_print(state: np.ndarray) -> None:
    """
    Imprime información de depuración.
    """
    print("\n" + "="*50)
    print("Estado cuántico:")
    for i, amp in enumerate(state):
        print(f"|{format(i, '0' + str(int(np.log2(len(state)))) + 'b')}⟩: {amp}")
    print("Validación de amplitud:", amplitud_validation(state))
    print("="*50)

def amplitud_validation(state: np.ndarray) -> bool:
    """
    Verifica que la suma de los cuadrados de las amplitudes sea 1.
    """
    return np.isclose(np.sum(np.abs(state)**2), 1)

def hadamard_n(n: int) -> np.ndarray:
    """
    Genera la matriz de Hadamard para n qubits. https://en.wikipedia.org/wiki/Hadamard_transform
    """
    if n == 0:
        return np.array([[1]])
    
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H = H1
    for _ in range(1, n):
        H = np.kron(H, H1)
    return H

def state_to_index(binary_str: str) -> int:
    """
    Convierte un estado binario (ej: '10') a su índice entero correspondiente.
    """
    return int(binary_str, 2)

def create_targets(n: int, states: Set[str]) -> Set[int]:
    """
    Convierte un conjunto de estados binarios a sus índices numéricos.
    """
    targets = set()
    for state in states:
        targets.add(state_to_index(state))
    return targets

def oracle_matrix(N: int, targets: Set[int]) -> np.ndarray:
    """
    Crea el operador de oráculo que cambia el signo de los estados marcados.
    """
    oracle = np.eye(N, dtype=complex)
    for target in targets:
        oracle[target, target] = -1
    return oracle

def diffusion_matrix(n: int) -> np.ndarray:
    """
    Crea el operador de difusión (inversión sobre la media).
    """
    N = 2**n
    s = np.ones(N) / np.sqrt(N)
    diffusion = 2 * np.outer(s, s.conj()) - np.eye(N)
    
    return diffusion