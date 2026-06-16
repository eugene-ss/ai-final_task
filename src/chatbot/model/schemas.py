"""Shared Pydantic schemas: RAG, access control, healthcare entities, disasters."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

class Role(str, Enum):
    ADMIN = "admin"
    HR_MANAGER = "hr_manager"
    RECRUITER = "recruiter"
    ANALYST = "analyst"

class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ANALYZE = "analyze"

class User(BaseModel):
    user_id: str = Field(..., min_length=1)
    role: Role
    department: Optional[str] = None
    allowed_categories: Optional[Set[str]] = None

    class Config:
        use_enum_values = True

class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    category: str
    source: str
    file_path: Optional[str] = None
    original_index: Optional[int] = None
    headline: Optional[str] = None
    skills: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    chunk_uid: Optional[str] = None
    owner_id: Optional[str] = None
    access_list: Optional[List[str]] = None
    source_type: Optional[str] = None

class ResumeDocument(BaseModel):
    page_content: str = Field(..., min_length=1)
    metadata: DocumentMetadata

class SearchResult(BaseModel):
    document: ResumeDocument
    score: float = Field(..., ge=0.0, le=1.0)
    method: str

class EvaluationQuery(BaseModel):
    query: str = Field(..., min_length=1)
    relevant_categories: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

class EvaluationMetrics(BaseModel):
    precision_at_1: float = Field(..., ge=0.0, le=1.0)
    precision_at_3: float = Field(..., ge=0.0, le=1.0)
    precision_at_5: float = Field(..., ge=0.0, le=1.0)
    precision_at_10: float = Field(..., ge=0.0, le=1.0)
    recall_at_1: float = Field(..., ge=0.0, le=1.0)
    recall_at_3: float = Field(..., ge=0.0, le=1.0)
    recall_at_5: float = Field(..., ge=0.0, le=1.0)
    recall_at_10: float = Field(..., ge=0.0, le=1.0)
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    groundedness: float = Field(..., ge=0.0, le=1.0)
    answer_completeness: float = Field(default=0.5, ge=0.0, le=1.0)
    avg_relevance: float = Field(..., ge=0.0, le=1.0)
    query: str
    has_labels: bool = False

class EvaluationResults(BaseModel):
    summary: Dict[str, float]
    individual_results: List[EvaluationMetrics]
    total_queries: int = Field(..., ge=0)

class CitedCandidate(BaseModel):
    doc_id: str = Field(
        ..., description="Stable document id (e.g. EM-DAT 1995-0010-JPN)"
    )
    category: str = ""
    evidence_snippet: str = ""
    relevance_note: str = ""

class RAGStructuredAnswer(BaseModel):
    summary: str
    candidates: List[CitedCandidate] = Field(default_factory=list)
    confidence: str = "medium"

class ExcerptRelevanceScore(BaseModel):
    index: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0)

class EvaluationQualityScores(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    groundedness: float = Field(..., ge=0.0, le=1.0)
    answer_completeness: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_scores: List[ExcerptRelevanceScore] = Field(default_factory=list)

class HealthcareEntity(BaseModel):
    text: str = Field(..., description="Verbatim span from the input text")
    type: str = Field(
        ...,
        description="Entity type: Condition, Symptom, Medication, Anatomy, "
        "Procedure, Dosage, RiskFactor",
    )
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    normalized: Optional[str] = Field(
        None,
        description="Canonical / normalised form (e.g. generic drug name)",
    )

class HealthcareRelation(BaseModel):
    subject: str = Field(..., description="Verbatim subject span")
    relation: str = Field(..., description="treats / causes / dosage_of / located_in / contraindicated_with")
    object: str = Field(..., description="Verbatim object span")

class HealthcareICD10Suggestion(BaseModel):
    code: str = Field(..., description="ICD-10 code, e.g. I10")
    label: str = Field(..., description="Human-readable label, e.g. Essential hypertension")
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    rationale: str = ""

class HealthcareEntities(BaseModel):
    """Structured response of ``extract_medical_entities``."""

    entities: List[HealthcareEntity] = Field(default_factory=list)
    relations: List[HealthcareRelation] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    icd10_suggestions: List[HealthcareICD10Suggestion] = Field(default_factory=list)

class HealthcareSummary(BaseModel):
    audience: str = Field("clinician", description="clinician or patient")
    summary: str = ""
    bullet_points: List[str] = Field(default_factory=list)
