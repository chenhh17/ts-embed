import numpy as np
from sklearn.random_projection import SparseRandomProjection



class LSHBinaryEncoder:
    def __init__(self, input_dim, output_bits=1000, n_projections=None, seed=42):
        """
        LSH-based binary encoding
        
        Args:
            input_dim: Original embedding dimension
            output_bits: Target binary bits (e.g., 1000)
            n_projections: Number of random projections (default: output_bits)
        """
        self.output_bits = output_bits
        self.n_projections = n_projections or output_bits
        
        # Create random projection matrix
        self.projector = SparseRandomProjection(
            n_components=self.n_projections,
            random_state=seed
        )
        
        np.random.seed(seed)
        dummy_data = np.random.randn(100, input_dim)
        self.projector.fit(dummy_data)
    
    def encode(self, embedding) -> str:
        """Convert embedding to binary using LSH"""
        # Reshape for sklearn
        embedding = embedding.reshape(1, -1)
        
        # Apply random projections
        projected = self.projector.transform(embedding)
        
        # Take first output_bits projections if we have more
        if projected.shape[1] > self.output_bits:
            projected = projected[:, :self.output_bits]
        elif projected.shape[1] < self.output_bits:
            # Pad with zeros if needed
            padding = self.output_bits - projected.shape[1]
            projected = np.pad(projected, ((0, 0), (0, padding)), mode='constant')
        
        # Convert to binary and flatten
        binary = (projected > 0).astype(int).flatten()
        return ''.join(binary.astype(str))

