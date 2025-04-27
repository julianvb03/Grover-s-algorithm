import numpy as np
from typing import Set
from auxiliar import (
    oracle_matrix, 
    diffusion_matrix, 
    create_targets,
)

"""
===========================
Implementación del algoritmo de Grover
La información la saqué de: https://es.wikipedia.org/wiki/Algoritmo_de_Grover

El cual consta de los siguientes pasos:
1. Inicializar el estado cuántico
2. Hacemos iteraciones de grover:
    - Aplicar la función de oráculo
    - Aplicar la inversión sobre la media
3. Medir el estado cuántico
===========================
"""

def grover_algorithm(n_qubits: int, target_states: set) -> np.ndarray:
    """
    Implementa el algoritmo de Grover para búsqueda cuántica.
    """
    N = 2**n_qubits
    targets = create_targets(n_qubits, target_states)
    M = len(targets)
    iterations = int(np.pi/4 * np.sqrt(N/M))
    state = np.ones(N) / np.sqrt(N)
    oracle = oracle_matrix(N, targets)
    diffusion = diffusion_matrix(n_qubits)
    
    for i in range(iterations):
        state = np.dot(oracle, state)
        state = np.dot(diffusion, state)
    
    return state

if __name__ == "__main__":
    n = int(input("Ingrese el número de qubits: "))
    if n <= 0:
        raise ValueError("El número de qubits debe ser mayor que 0.")
    
    states_input = input("Ingrese los estados a marcar (ej: '00,11'): ").strip().split(',')
    target_states = {s.strip() for s in states_input}
    
    for s in target_states:
        if len(s) != n or any(c not in {'0','1'} for c in s):
            raise ValueError(f"Estado inválido: {s}. Debe tener {n} bits (0 o 1).")
    
    print(f"\nEjecutando algoritmo de Grover con {n} qubits.")
    print(f"Estados marcados: {', '.join(sorted(target_states))}")
    print(f"Número de iteraciones: {int(np.round(np.pi/4 * np.sqrt(2**n/len(target_states))))}")
    
    final_state = grover_algorithm(n, target_states)
    
    print("\nResultado final:")
    for i in range(len(final_state)):
        state_str = format(i, f'0{n}b')
        prob = np.abs(final_state[i])**2
        print(f"|{state_str}⟩: {prob:.4f} ({prob:.2%})")