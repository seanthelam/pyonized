"""
emissivity.py

Voxel-level line emissivity. This will, for now, implement a simplified Case B
recombination emissivity for Halpha (Osterbrock and Ferland),
parameterized by electron density and temperature, as a stand-in until I add
pyneb for multi-line, abundance-dependent emissivity
"""

from __future__ import annotations
import numpy as np
import pyneb as pn
