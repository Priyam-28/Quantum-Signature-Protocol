# 🧬 Quantum-Enhanced Digital Signature Protocol

This project demonstrates a **hybrid classical-quantum digital signature protocol** using:

- **Falcon** (a NIST-approved post-quantum digital signature scheme)
- **Qiskit** (IBM’s quantum computing framework)
- **Image Processing** (sign and verify images securely)


---

## 📌 Features

- 🔐 Secure signing using **Falcon**
- 🧠 Quantum-enhanced tamper detection
- 🖼️ Signs and verifies image files
- ⚠️ Detects tampering via pixel-level analysis
- ⏱️ Logs performance metrics (time, signature length)
- 📊 Uses quantum simulation via Qiskit AER

---

## 🚀 Workflow

### 1. Key Generation

- Falcon generates classical public/private key pairs.
- Quantum circuit initialized with Hadamard and CNOT gates to simulate quantum key distribution.

### 2. Image Signing

- Convert image to byte stream.
- Sign using Falcon private key.
- Use image-derived bitstring to conditionally apply **Z gates** on a quantum register.

### 3. Verification

- Verify classical Falcon signature with public key.
- Check image against quantum-encoded register state.
- Any pixel change results in verification failure.

### 4. Tamper Detection

- Modify a pixel in the image.
- Re-run verification → should fail due to mismatch.

---

## 🧾 Requirements

- Python 3.8+
- `qiskit`
- `qiskit-aer`
- `falcon`
- `numpy`, `matplotlib`, `Pillow`, `requests`, `prettytable`

Install all dependencies:
```bash
pip install qiskit qiskit-aer falcon numpy matplotlib pillow requests prettytable

🖼️ Images
🧷 Original Image
![Original Image](images/image1.jpg)

🧪 Tampered Image (1 pixel modified)
![Original Image](images/image1.png)

🆕 New Image (completely different)
![Original Image](images/image2.png)

📊 Verification Results

Metric	Value
Algorithm	Falcon + Quantum Hybrid
Signature Length	~666 bytes
Signing Time	~0.045 seconds
Verification Time	~0.052 seconds
Original Image Valid?	✅ Yes
Tampered Image Valid?	❌ No
New Image Valid?	❌ No
📚 Concepts Used
Falcon: Lattice-based, quantum-safe signature

Quantum Circuit: Hadamard, Z gates, CNOT

Polynomial Rings: Fundamental to lattice cryptography

Z Gate: Applies phase flip in quantum state

Private/Public Keys: Signing vs verification

❓ Why Quantum?
Adds a second layer of validation using quantum logic gates

Can eventually transition to quantum-only signatures

Demonstrates how quantum and classical cryptography can integrate

