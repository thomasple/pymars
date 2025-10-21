import numpy as np

class CollisionSystems:
    """Data model for collision systems in pymars.

    Attributes:
     - species: array of shape (N,) with atomic species indices
     - coordinates: array of shape (N,3) with atomic coordinates
     - velocities: array of shape (N,3) with atomic velocities
    """

    def __init__(self, species, coordinates, velocities):
        self.species = species
        self.coordinates = coordinates
        self.velocities = velocities