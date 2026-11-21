"""Top-level deliverable entry module exposing VL-P1b Object Proposal Research classes."""
from src.proposal_research import (
    ObjectProposerVL1b,
    ProposalBox,
    ProposalRanker,
    YOLOXNanoExtractor,
    propose_multiscale_saliency_windows,
    propose_person_interaction_zones,
    propose_thin_and_small_geometry,
)

__all__ = [
    "ObjectProposerVL1b",
    "ProposalBox",
    "ProposalRanker",
    "YOLOXNanoExtractor",
    "propose_multiscale_saliency_windows",
    "propose_person_interaction_zones",
    "propose_thin_and_small_geometry",
]
