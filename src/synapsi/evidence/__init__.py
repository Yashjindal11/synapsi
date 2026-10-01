"""Evidence: information with provenance that claims can rest on."""

from synapsi.evidence.models import Evidence, Provenance, SourceKind
from synapsi.evidence.pool import EvidencePool, SharedSource, SourceDependence

__all__ = [
    "Evidence",
    "EvidencePool",
    "Provenance",
    "SharedSource",
    "SourceDependence",
    "SourceKind",
]
