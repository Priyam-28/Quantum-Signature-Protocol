from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import hashlib
import numpy as np
import random

class QuantumPKG:
    """Private Key Generator (PKG) - Trusted Authority"""
    
    def __init__(self, security_param=8):
        self.security_param = security_param
        self.master_secret = ''.join(random.choice('01') for _ in range(security_param * 4))
        print(f"PKG Master Secret: {self.master_secret}")
        self.simulator = AerSimulator(method='statevector')
        
    def generate_user_keys(self, user_id):
        hash_input = user_id + self.master_secret
        hash_obj = hashlib.sha256(hash_input.encode()).hexdigest()
        bits = bin(int(hash_obj, 16))[2:].zfill(256)
        return {
            'id': user_id,
            'sign_key': bits[:self.security_param * 2],
            'verify_key': bits[self.security_param * 2:self.security_param * 4]
        }
    
    def distribute_entangled_keys(self, alice_id, bob_id):
        num_pairs = self.security_param
        alice_qubits = []
        bob_qubits = []
        
        print(f"PKG generating keys for {alice_id} and {bob_id} with {num_pairs} entangled Bell pairs")
        
        for i in range(num_pairs):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.save_statevector()
            
            # Execute with statevector preservation
            transpiled_qc = transpile(qc, self.simulator)
            job = self.simulator.run(transpiled_qc)
            result = job.result()
            
            # Get statevector from saved results - use the correct method for qiskit-aer 0.10+
            state = result.data().get('statevector')
            
            alice_qubits.append({'state': state, 'index': 0})
            bob_qubits.append({'state': state, 'index': 1})
            
        alice_keys = self.generate_user_keys(alice_id)
        bob_keys = self.generate_user_keys(bob_id)
        alice_keys['entangled_qubits'] = alice_qubits
        bob_keys['entangled_qubits'] = bob_qubits
        
        # Print some debug info
        print(f"Alice's signing key (first 16 bits): {alice_keys['sign_key'][:16]}...")
        print(f"Bob's verification key (first 16 bits): {bob_keys['verify_key'][:16]}...")
        
        return alice_keys, bob_keys

class QuantumSigner:
    """Quantum Signer (Alice)"""
    
    def __init__(self, user_keys):
        self.keys = user_keys
        self.simulator = AerSimulator(method='statevector')
        
    def prepare_message_qubits(self, message):
        num_qubits = len(message)
        qc = QuantumCircuit(num_qubits)
        for i, bit in enumerate(message):
            if bit == '1':
                qc.x(i)
        return qc
    
    def apply_quantum_otp(self, qc, otp_key):
        # Make a copy to avoid modifying the original
        signed_qc = qc.copy()
        
        for i in range(signed_qc.num_qubits):
            key_index = i * 2
            if key_index + 1 < len(otp_key):
                if otp_key[key_index] == '1':
                    signed_qc.x(i)
                if otp_key[key_index + 1] == '1':
                    signed_qc.z(i)
        return signed_qc
    
    def add_decoy_qubits(self, qc):
        # Get original circuit size
        orig_size = qc.num_qubits
        num_decoys = orig_size
        total_qubits = orig_size + num_decoys
        
        # Create new circuit with expanded size
        qr = QuantumRegister(total_qubits)
        new_qc = QuantumCircuit(qr)
        
        # Copy original circuit to the first qubits
        for i in range(orig_size):
            # We need to manually copy any gates from the original circuit
            for gate, qargs, cargs in qc._data:
                if len(qargs) == 1 and qargs[0]._index == i:
                    # Single qubit gate
                    new_qc.append(gate, [qr[i]])
                elif len(qargs) == 2 and (qargs[0]._index == i or qargs[1]._index == i):
                    # Two qubit gate - only copy if both qubits are in the original range
                    if qargs[0]._index < orig_size and qargs[1]._index < orig_size:
                        new_qc.append(gate, [qr[qargs[0]._index], qr[qargs[1]._index]])
        
        # Add decoy qubits to random positions
        decoy_info = []
        for i in range(orig_size, total_qubits):
            basis = random.choice(['Z', 'X'])
            bit = random.choice(['0', '1'])
            
            if bit == '1':
                new_qc.x(i)
            if basis == 'X':
                new_qc.h(i)
                
            decoy_info.append({'position': i, 'basis': basis, 'bit': bit})
        
        # Print decoy info for debugging
        message_positions = list(range(orig_size))
        decoy_positions = [info['position'] for info in decoy_info]
        print(f"Alice created quantum signature with {total_qubits} qubits")
        print(f"- Message qubits at positions: {message_positions}")
        print(f"- Added {len(decoy_info)} decoy qubits for eavesdropping detection")
            
        return new_qc, decoy_info
    
    def create_signature(self, message):
        qc_message = self.prepare_message_qubits(message)
        # Make sure we have enough key bits for the message
        otp_key = self.keys['sign_key'][:len(message)*2]
        qc_signed = self.apply_quantum_otp(qc_message, otp_key)
        qc_with_decoys, decoy_info = self.add_decoy_qubits(qc_signed)
        
        return qc_with_decoys, {
            'signer_id': self.keys['id'],
            'message_length': len(message),
            'decoy_info': decoy_info,
            'otp_key': otp_key
        }


class QuantumVerifier:
    """Quantum Verifier (Bob)"""
    
    def __init__(self, user_keys):
        self.keys = user_keys
        self.simulator = AerSimulator()
    
    def check_decoy_qubits(self, qc, decoy_info):
        errors = 0
        total = len(decoy_info)
        
        for decoy in decoy_info:
            pos = decoy['position']
            
            # Create a new circuit for measuring this specific decoy qubit
            measure_qc = QuantumCircuit(qc.num_qubits, 1)
            
            # Apply appropriate basis change if needed
            if decoy['basis'] == 'X':
                measure_qc.h(pos)
                
            # Measure in computational basis
            measure_qc.measure(pos, 0)
            
            # Compose with the input circuit
            full_qc = qc.copy()
            full_qc = full_qc.compose(measure_qc)
            
            # Run the measurement
            job = self.simulator.run(full_qc, shots=1)
            result = job.result().get_counts()
            measured_bit = list(result.keys())[0]
            
            # Check if measurement matches expected bit
            if measured_bit != decoy['bit']:
                errors += 1
                
        error_rate = errors / total if total > 0 else 0
        return error_rate < 0.5, error_rate
    
    def verify_signature(self, qc, metadata, orig_message):
    # First check decoy qubits to detect tampering
        is_secure, error_rate = self.check_decoy_qubits(qc, metadata['decoy_info'])
        
        if not is_secure:
            return False, "Eavesdropping detected"
        
        # Now measure the message qubits
        msg_length = metadata['message_length']
        
        # Create proper registers
        qr = QuantumRegister(qc.num_qubits)
        cr = ClassicalRegister(msg_length, 'cr')
        measure_qc = QuantumCircuit(qr, cr)
        
        # Measure the first msg_length qubits (which contain the message)
        for i in range(msg_length):
            measure_qc.measure(i, i)
        
        # Execute the measurement
        composed_qc = qc.copy()
        composed_qc = composed_qc.compose(measure_qc)
        
        job = self.simulator.run(composed_qc, shots=1)
        result = job.result().get_counts()
        
        # Get the measured bits
        measured = list(result.keys())[0][:msg_length]
        
        # Verify against the original message
        is_valid = (measured == orig_message)
        return is_valid, "Valid" if is_valid else "Invalid"


class EavesdroppingSimulator:
    """Simulates Eve attempting to intercept the quantum signature"""
    
    def __init__(self):
        self.simulator = AerSimulator()
    
    def intercept_and_measure(self, qc, num_to_measure=3):
        """Simulates Eve measuring random qubits in the transmission"""
        eve_qc = qc.copy()
        total_qubits = eve_qc.num_qubits
        
        # Choose random positions to measure
        positions_to_measure = sorted(random.sample(range(total_qubits), min(num_to_measure, total_qubits)))
        
        # Create measurement circuit
        cr = ClassicalRegister(len(positions_to_measure), 'e')
        eve_measure = QuantumCircuit(total_qubits, cr)
        
        # Add measurements
        for i, pos in enumerate(positions_to_measure):
            eve_measure.measure(pos, i)
        
        # Compose with original circuit
        eve_full_qc = eve_qc.compose(eve_measure)
        
        # Execute the measurement
        job = self.simulator.run(eve_full_qc, shots=1)
        result = job.result().get_counts()
        measured_bits = list(result.keys())[0][:len(positions_to_measure)]
        
        print(f"Eve measured qubits at positions {positions_to_measure} and got result: {measured_bits}")
        
        # Return modified circuit with measurements applied (which causes decoherence)
        return eve_full_qc


def demonstrate_protocol():
    # Security parameter (reduced for demo)
    security_param = 4
    
    # Initialize PKG and generate keys
    pkg = QuantumPKG(security_param=security_param)
    alice_id = "alice@quantum.com"
    bob_id = "bob@quantum.com"
    message = '1101'
    
    print(f"Protocol Demo: Alice ({alice_id}) signing message '{message}' for Bob ({bob_id})")
    print("-" * 80)
    
    # Generate and distribute keys
    alice_keys, bob_keys = pkg.distribute_entangled_keys(alice_id, bob_id)
    
    # Initialize participants
    alice = QuantumSigner(alice_keys)
    bob = QuantumVerifier(bob_keys)
    
    # Alice creates signature
    signature_qc, signature_metadata = alice.create_signature(message)
    print("-" * 80)
    
    # Bob verifies the signature
    is_valid, result_msg = bob.verify_signature(signature_qc, signature_metadata, message)
    print(f"\nVerification result: {result_msg}")
    print(f"Signature validity: {is_valid}")
    print(f"The signature is {'VALID' if is_valid else 'INVALID'}")
    
    # Demonstrate eavesdropping detection
    print("\nSimulating eavesdropping attack...")
    eve = EavesdroppingSimulator()
    eve_qc = eve.intercept_and_measure(signature_qc, num_to_measure=3)
    
    # Bob verifies again after Eve's interception
    is_valid_after_attack, result_msg = bob.verify_signature(eve_qc, signature_metadata, message)
    print(f"\nVerification after eavesdropping: {result_msg}")
    print(f"Signature validity after attack: {is_valid_after_attack}")
    print(f"The signature after eavesdropping is {'VALID' if is_valid_after_attack else 'INVALID'}")


if __name__ == "__main__":
    demonstrate_protocol()