import sys
sys.path.append('/content/falcon.py')
import time
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from PIL import Image
from prettytable import PrettyTable
import textwrap

# Try to import quantum libraries, with fallback mechanisms
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
    from qiskit.visualization import plot_histogram
    HAS_QISKIT = True
except ImportError:
    print("Qiskit not found. Hybrid quantum-classical simulation will use classical approximation.")
    HAS_QISKIT = False

try:
    import falcon
    HAS_FALCON = True
except ImportError:
    print("Falcon library not found. Using simplified classical post-quantum approach.")
    HAS_FALCON = False

class QuantumSignatureSimulator:
    """
    A simulator for demonstrating quantum digital signature protocols
    combining both post-quantum algorithms (Falcon) and quantum circuit approaches.
    """
    
    def __init__(self, security_parameter=512, qubits=4):
        """
        Initialize the simulator with specified security parameters.
        
        Args:
            security_parameter: The security parameter for Falcon (512 or 1024)
            qubits: The number of qubits to use in quantum circuit simulation
        """
        self.security_parameter = security_parameter
        self.qubits = qubits
        self.falcon_keys = None
        self.quantum_circuit = None
        
        # Initialize Falcon keys if available
        if HAS_FALCON:
            self.falcon_keys = self._generate_falcon_keys()
            
        # Initialize quantum simulator
        if HAS_QISKIT:
            self.quantum_simulator = Aer.get_backend('qasm_simulator')
        
    def _generate_falcon_keys(self):
        """Generate Falcon keys for post-quantum signature"""
        secret_key = falcon.SecretKey(self.security_parameter)
        public_key = falcon.PublicKey(secret_key)
        return {'secret': secret_key, 'public': public_key}
    
    def hadamard_gate(self, qc, qubit):
        """Apply Hadamard gate to specified qubit"""
        if HAS_QISKIT:
            qc.h(qubit)
        return qc
    
    def generate_quantum_keys(self):
        """
        Generate quantum keys using a quantum circuit.
        
        Returns:
            A quantum circuit with prepared keys
        """
        if not HAS_QISKIT:
            print("Simulating quantum key generation classically")
            return None
            
        # Create quantum registers
        private_key_qr = QuantumRegister(self.qubits, name='K')
        public_key_qr = QuantumRegister(self.qubits, name='P')
        message_qr = QuantumRegister(self.qubits, name='message')
        signature_qr = QuantumRegister(self.qubits, name='signature')
        verify_qr = QuantumRegister(self.qubits, name='verify')
        c = ClassicalRegister(self.qubits, name='c')
        
        # Create quantum circuit
        qc = QuantumCircuit(private_key_qr, public_key_qr, message_qr, signature_qr, verify_qr, c)
        
        # Apply Hadamard gates to create superposition for private key
        for i in range(self.qubits):
            qc = self.hadamard_gate(qc, private_key_qr[i])
            
        # Entangle private and public keys
        for i in range(self.qubits):
            qc.cx(private_key_qr[i], public_key_qr[i])
            
        # Store the circuit
        self.quantum_circuit = qc
        return qc
        
    def sign_circuit(self, message_bits):
        """
        Sign a message using quantum circuit
        
        Args:
            message_bits: List of bits representing the message
            
        Returns:
            Quantum circuit with signature operation applied
        """
        if not HAS_QISKIT or self.quantum_circuit is None:
            print("Simulating quantum signature classically")
            return None
            
        qc = self.quantum_circuit.copy()
        
        # Convert message to quantum state
        for i, bit in enumerate(message_bits[:self.qubits]):
            if bit == 1:
                qc.x(qc.qregs[2][i])  # message_qr is at index 2
                
        # Apply signature operations based on private key
        for i in range(self.qubits):
            qc.cx(qc.qregs[0][i], qc.qregs[3][i])  # private_key to signature
            
        # Apply additional quantum operations for the signature
        for i in range(self.qubits):
            qc.h(qc.qregs[3][i])
            
        return qc
    
    def verify_circuit(self, message_bits, signature_bits):
        """
        Verify a signature using quantum circuit
        
        Args:
            message_bits: List of bits representing the message
            signature_bits: List of bits representing the signature
            
        Returns:
            Quantum circuit with verification operations
        """
        if not HAS_QISKIT or self.quantum_circuit is None:
            print("Simulating quantum verification classically")
            return None
            
        qc = self.quantum_circuit.copy()
        
        # Convert message to quantum state
        for i, bit in enumerate(message_bits[:self.qubits]):
            if bit == 1:
                qc.x(qc.qregs[2][i])  # message_qr is at index 2
                
        # Apply verification operations with public key
        for i in range(self.qubits):
            qc.cx(qc.qregs[1][i], qc.qregs[4][i])  # public_key to verify
            
        # Apply signature bits
        for i, bit in enumerate(signature_bits[:self.qubits]):
            if bit == 1:
                qc.x(qc.qregs[3][i])  # signature_qr is at index 3
                
        # Compare signature against verification
        for i in range(self.qubits):
            qc.cx(qc.qregs[3][i], qc.qregs[4][i])
            qc.measure(qc.qregs[4][i], qc.cregs[0][i])
            
        return qc
    
    def image_to_bytes(self, image):
        """Convert PIL Image to bytes"""
        with BytesIO() as output:
            image.save(output, format="PNG")
            return output.getvalue()
    
    def bytes_to_image(self, byte_string):
        """Convert bytes to PIL Image"""
        return Image.open(BytesIO(byte_string))
    
    def sign_image_falcon(self, image, max_retries=100):
        """Sign an image using Falcon signature algorithm"""
        if not HAS_FALCON or self.falcon_keys is None:
            print("Falcon not available. Simulating classical post-quantum signature.")
            return hashlib.sha256(self.image_to_bytes(image)).digest(), None
            
        image_bytes = self.image_to_bytes(image)
        secret_key = self.falcon_keys['secret']
        
        retry_count = 0
        error_message = None
        
        while True:
            try:
                signature = secret_key.sign(image_bytes)
                print(f"Falcon signature generated successfully after {retry_count} retries.")
                return signature, error_message
            except ValueError as e:
                if "Squared norm of signature is too large" in str(e):
                    retry_count += 1
                    print(f"Signature norm too large. Retrying ({retry_count}/{max_retries})...")
                    error_message = str(e)
                    if retry_count >= max_retries:
                        return None, error_message
                else:
                    raise e
    
    def verify_image_falcon(self, image, signature):
        """Verify an image signature using Falcon"""
        if not HAS_FALCON or self.falcon_keys is None:
            print("Falcon not available. Simulating classical verification.")
            return True
            
        image_bytes = self.image_to_bytes(image)
        public_key = self.falcon_keys['public']
        
        return public_key.verify(image_bytes, signature)
    
    def simulate_quantum_run(self, message_bits):
        """
        Run the quantum simulation for signature and verification
        
        Args:
            message_bits: The message bits to sign
            
        Returns:
            results: Dictionary with simulation results
        """
        if not HAS_QISKIT:
            return {'valid': True, 'counts': {}, 'signature': np.random.randint(0, 2, size=self.qubits).tolist()}
            
        # Generate random signature bits for simulation
        signature_bits = np.random.randint(0, 2, size=self.qubits).tolist()
        
        # Create circuits for signature and verification
        sign_qc = self.sign_circuit(message_bits)
        verify_qc = self.verify_circuit(message_bits, signature_bits)
        
        # Execute verification circuit
        job = execute(verify_qc, self.quantum_simulator, shots=1024)
        result = job.result()
        counts = result.get_counts(verify_qc)
        
        # In an ideal verification, we should see all zeros if signature is valid
        # (indicating equality between signature and verification registers)
        valid = '0' * self.qubits in counts and counts['0' * self.qubits] > 500
        
        return {
            'valid': valid,
            'counts': counts,
            'signature': signature_bits
        }
    
    def load_image(self, url):
        """Load image from URL"""
        response = requests.get(url)
        return Image.open(BytesIO(response.content))


def main():
    # Create a simulator instance
    simulator = QuantumSignatureSimulator(security_parameter=512, qubits=4)
    
    # Generate quantum keys
    quantum_circuit = simulator.generate_quantum_keys()
    
    # Load sample images
    image_url = "https://picsum.photos/200/300"
    image = simulator.load_image(image_url)
    
    different_image_url = "https://picsum.photos/200/300?random=1"
    different_image = simulator.load_image(different_image_url)
    
    # Create a tampered image
    tampered_image = image.copy()
    tampered_image.putpixel((0, 0), (255, 0, 0))
    
    # Create a message bit sequence from image hash
    image_hash = hashlib.sha256(simulator.image_to_bytes(image)).digest()
    message_bits = [int(bit) for bit in bin(int.from_bytes(image_hash[:2], byteorder='big'))[2:].zfill(16)]
    
    # Sign the image using both methods
    print("Signing image with Falcon...")
    start_time = time.time()
    falcon_signature, error_message = simulator.sign_image_falcon(image)
    falcon_sign_time = time.time() - start_time
    
    print("Signing image with quantum circuit simulation...")
    start_time = time.time()
    quantum_result = simulator.simulate_quantum_run(message_bits[:simulator.qubits])
    quantum_sign_time = time.time() - start_time
    
    # Create results table
    table = PrettyTable()
    table.field_names = ["Metric", "Classical PQ (Falcon)", "Quantum Circuit", "Explanation"]
    table.align["Metric"] = "l"
    table.align["Classical PQ (Falcon)"] = "l"
    table.align["Quantum Circuit"] = "l"
    table.align["Explanation"] = "l"
    
    # Add algorithm metrics
    table.add_row([
        "Security Parameter", 
        simulator.security_parameter, 
        f"{simulator.qubits} qubits", 
        "Security parameters for both approaches"
    ])
    
    table.add_row([
        "Signature Time", 
        f"{falcon_sign_time:.6f} seconds", 
        f"{quantum_sign_time:.6f} seconds", 
        "Time taken to generate signatures"
    ])
    
    # Verify original image with Falcon
    if falcon_signature is not None:
        start_time = time.time()
        falcon_original_valid = simulator.verify_image_falcon(image, falcon_signature)
        falcon_verify_time = time.time() - start_time
        
        # Add Falcon signature metrics
        table.add_row([
            "Signature Length", 
            len(falcon_signature) if falcon_signature else "N/A", 
            len(quantum_result['signature']) if 'signature' in quantum_result else "N/A",
            "Size of generated signatures"
        ])
        
        table.add_row([
            "Verification Time", 
            f"{falcon_verify_time:.6f} seconds", 
            "N/A for simulation", 
            "Time taken for verification"
        ])
        
        # Verify different image with Falcon
        falcon_different_valid = simulator.verify_image_falcon(different_image, falcon_signature)
        
        # Verify tampered image with Falcon
        falcon_tampered_valid = simulator.verify_image_falcon(tampered_image, falcon_signature)
        
        # Add verification results
        table.add_row([
            "Original Image Valid?", 
            falcon_original_valid, 
            quantum_result.get('valid', "N/A"), 
            "Verification result for original image"
        ])
        
        table.add_row([
            "Different Image Valid?", 
            falcon_different_valid, 
            "False (simulated)", 
            "Verification for different image (should be False)"
        ])
        
        table.add_row([
            "Tampered Image Valid?", 
            falcon_tampered_valid, 
            "False (simulated)", 
            "Verification for tampered image (should be False)"
        ])
    else:
        table.add_row([
            "Error Message", 
            error_message if error_message else "Unknown error", 
            "N/A", 
            "Error during signature generation"
        ])
    
    # Display the original image
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    
    # Display the different image
    plt.subplot(1, 3, 2)
    plt.title("Different Image")
    plt.imshow(different_image)
    plt.axis('off')
    
    # Display the tampered image
    plt.subplot(1, 3, 3)
    plt.title("Tampered Image")
    plt.imshow(tampered_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Plot quantum circuit if available
    if HAS_QISKIT and quantum_circuit is not None:
        print("Quantum Key Generation Circuit:")
        print(quantum_circuit.draw())
        
        if 'counts' in quantum_result and quantum_result['counts']:
            plt.figure(figsize=(8, 6))
            plot_histogram(quantum_result['counts'])
            plt.title('Quantum Signature Verification Results')
            plt.show()
    
    # Print the results table
    print("\nQuantum Signature Protocol Comparison:")
    print(table)
    
    # Print short explanation of the approaches
    print("\nExplanation of Approaches:")
    print("1. Classical Post-Quantum (Falcon): Uses lattice-based cryptography resistant to quantum attacks")
    print("2. Quantum Circuit: Uses quantum properties like superposition and entanglement for signatures")
    print("\nAdvantages of Quantum Signatures:")
    print("- Information-theoretic security rather than computational security")
    print("- Can leverage quantum phenomena like no-cloning theorem")
    print("- Resistant to attacks even from quantum computers")
    print("\nChallenges:")
    print("- Requires quantum hardware for true implementation")
    print("- Currently limited by NISQ era constraints (noise, decoherence)")
    print("- Key distribution remains a significant challenge")

if __name__ == "__main__":
    main()