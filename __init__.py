"""
aiie_phase4 — AIIE Phase 4: Cultural Transport Dynamics
Gemmina Intelligence LLC / Pure Information Laboratory
"""

from .phase4_config    import Phase4Config
from .domain_loader    import load_all_domains, DOMAIN_LOADERS
from .phase4_extractor import run_extraction
from .conductivity     import build_sigma_matrix, compute_sigma, format_sigma_table
from .sigma_visualizer import generate_all_figures
