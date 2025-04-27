from auxiliar import hadamard_n, state_to_index, create_targets, oracle_matrix, diffusion_matrix
import numpy as np
import pytest

def test_hadamard_0_dimensional():
    result = hadamard_n(0)
    expected = np.array([[1]])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_hadamard_1_dimensional():
    result = hadamard_n(1)
    amplitud = 1 / np.sqrt(2)
    expected = amplitud * np.array([
        [1,  1],
        [1, -1],
    ])
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_hadamard_2_dimensional():
    result = hadamard_n(2)
    amplitud = 1 / 2
    expected = amplitud * np.array([
        [ 1,  1,  1,  1],
        [ 1, -1,  1, -1],
        [ 1,  1, -1, -1],
        [ 1, -1, -1,  1],
    ])
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_hadamard_3_dimensional():
    result = hadamard_n(3)
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

def test_state_to_index_zero():
    assert state_to_index('0') == 0, "El índice de '0' debería ser 0"

def test_state_to_index_one():
    assert state_to_index('1') == 1, "El índice de '1' debería ser 1"

def test_state_to_index_two_qubits():
    assert state_to_index('00') == 0, "El índice de '00' debería ser 0"
    assert state_to_index('01') == 1, "El índice de '01' debería ser 1"
    assert state_to_index('10') == 2, "El índice de '10' debería ser 2"
    assert state_to_index('11') == 3, "El índice de '11' debería ser 3"

def test_state_to_index_three_qubits():
    assert state_to_index('000') == 0, "El índice de '000' debería ser 0"
    assert state_to_index('111') == 7, "El índice de '111' debería ser 7"
    assert state_to_index('101') == 5, "El índice de '101' debería ser 5"

def test_create_targets_single_state():
    targets = create_targets(2, {'01'})
    assert targets == {1}, f"Expected {{1}}, but got {targets}"

def test_create_targets_multiple_states():
    targets = create_targets(2, {'00', '11'})
    assert targets == {0, 3}, f"Expected {{0, 3}}, but got {targets}"

def test_create_targets_three_qubits():
    targets = create_targets(3, {'000', '101', '111'})
    assert targets == {0, 5, 7}, f"Expected {{0, 5, 7}}, but got {targets}"

def test_create_targets_invalid_length():
    with pytest.raises(ValueError):
        create_targets(2, {'0', '01', '001'})

def test_oracle_matrix_single_target():
    oracle = oracle_matrix(4, {1})
    expected = np.eye(4, dtype=complex)
    expected[1, 1] = -1
    assert np.array_equal(oracle, expected), f"Expected {expected}, but got {oracle}"

def test_oracle_matrix_multiple_targets():
    oracle = oracle_matrix(4, {0, 3})
    expected = np.eye(4, dtype=complex)
    expected[0, 0] = -1
    expected[3, 3] = -1
    assert np.array_equal(oracle, expected), f"Expected {expected}, but got {oracle}"

def test_oracle_matrix_eight_dimensions():
    oracle = oracle_matrix(8, {2, 5, 7})
    expected = np.eye(8, dtype=complex)
    expected[2, 2] = -1
    expected[5, 5] = -1
    expected[7, 7] = -1
    assert np.array_equal(oracle, expected), f"Expected {expected}, but got {oracle}"

def test_oracle_matrix_effect():
    state = np.ones(4) / 2
    oracle = oracle_matrix(4, {1})
    result = np.dot(oracle, state)
    
    expected = np.array([0.5, -0.5, 0.5, 0.5])
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_diffusion_matrix_one_qubit():
    diffusion = diffusion_matrix(1)
    expected = np.array([[0, 1], [1, 0]], dtype=complex)
    assert np.allclose(diffusion, expected), f"Expected {expected}, but got {diffusion}"

def test_diffusion_matrix_two_qubits():
    diffusion = diffusion_matrix(2)
    s = np.ones(4) / 2 
    expected = 2 * np.outer(s, s) - np.eye(4)
    assert np.allclose(diffusion, expected), f"Expected {expected}, but got {diffusion}"

def test_diffusion_matrix_hermitian():
    diffusion = diffusion_matrix(2)
    assert np.allclose(diffusion, diffusion.conj().T), "La matriz de difusión debe ser hermitiana"

def test_diffusion_matrix_unitary():
    diffusion = diffusion_matrix(2)
    product = np.dot(diffusion, diffusion.conj().T)
    assert np.allclose(product, np.eye(4)), "La matriz de difusión debe ser unitaria"

def test_diffusion_matrix_effect():
    state = np.array([0.5, -0.5, 0.5, 0.5])
    diffusion = diffusion_matrix(2)
    result = np.dot(diffusion, state)
    
    assert result[1] > 0, "El estado con fase invertida debería tener amplitud positiva después de la difusión"
    assert abs(result[1]) > abs(result[0]), "El estado marcado debería tener mayor amplitud"

def test_oracle_diffusion_integration():
    state = np.ones(4) / 2
    oracle = oracle_matrix(4, {2})
    state_after_oracle = np.dot(oracle, state)
    
    diffusion = diffusion_matrix(2)
    final_state = np.dot(diffusion, state_after_oracle)
    
    max_index = np.argmax(np.abs(final_state))
    assert max_index == 2, f"El estado con mayor amplitud debería ser |10⟩, pero es |{format(max_index, '02b')}⟩"
    
    prob = np.abs(final_state[2])**2
    assert prob > 0.25, f"La probabilidad de |10⟩ debería ser > 0.25, pero es {prob}"