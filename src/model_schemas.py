from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class StudentFeatures(BaseModel):
    student_id: str = Field(..., description="Unique identifier for the student")
    skills_hard: List[str] = Field(default_factory=list, description="Technical skills")
    skills_soft: List[str] = Field(default_factory=list, description="Soft skills")
    years_experience: float = Field(default=0.0, description="Total years of professional experience")
    education_level: int = Field(default=1, description="Ordinal encoding of highest degree (1=HS, 2=BS, 3=MS, 4=PhD)")
    expected_salary: float = Field(default=0.0, description="Expected annual salary in USD")
    preferred_location: str = Field(default="", description="Preferred city/region for work")
    remote_preference: str = Field(default="On-site", description="Remote, Hybrid, or On-site")
    coding_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized score [0-1] from technical assessments")
    communication_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized score [0-1] from communication assessments")

class JobFeatures(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the job")
    required_skills: List[str] = Field(default_factory=list, description="Must-have technical skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have technical skills")
    min_experience: float = Field(default=0.0, description="Minimum years of experience required")
    max_experience: Optional[float] = Field(default=None, description="Maximum years of experience expected")
    min_education: int = Field(default=1, description="Ordinal encoding of minimum required degree")
    salary_min: float = Field(default=0.0, description="Bottom of the salary band")
    salary_max: float = Field(default=0.0, description="Top of the salary band")
    job_location: str = Field(default="", description="City/region of the job")
    work_model: str = Field(default="On-site", description="Remote, Hybrid, or On-site")
    min_coding_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum acceptable coding score [0-1]")
    min_communication_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum acceptable communication score [0-1]")

class MatchVector(BaseModel):
    student_id: str
    job_id: str
    skill_overlap_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    experience_gap: float = Field(default=0.0)
    salary_match_ratio: float = Field(default=0.0)
    location_match: bool = Field(default=False)
    education_met: bool = Field(default=False)
    coding_threshold_met: bool = Field(default=False)
    communication_threshold_met: bool = Field(default=False)


class MatchScoreRequest(BaseModel):
    student: StudentFeatures
    job: JobFeatures

class MatchExplanation(BaseModel):
    positive_factors: List[str]
    negative_factors: List[str]


# ---------------------------------------------------------------------------
# Task 04 — Explainability Payload Schemas
# ---------------------------------------------------------------------------

class FeatureContribution(BaseModel):
    """A single feature's contribution to the overall match score."""
    feature: str = Field(..., description="Internal feature name (e.g. 'skill_overlap_ratio')")
    label: str = Field(..., description="Human-readable label (e.g. 'Skill Match')")
    value: float = Field(..., description="Actual feature value used in scoring")
    contribution: float = Field(..., description="Normalised contribution to final score [0,1]")
    verdict: str = Field(..., description="'strength', 'weakness', or 'neutral'")


class ExplanationPayload(BaseModel):
    """
    Structured, multi-layered explanation for a single (student, job) match.

    Designed for three audiences:
      • Students  — summary + strengths/weaknesses in plain language
      • Recruiters — shortlist flag + confidence band
      • Platform   — full feature_contributions for analytics / auditing
    """
    summary: str = Field(..., description="One-sentence natural-language explanation")
    strengths: List[FeatureContribution] = Field(default_factory=list)
    weaknesses: List[FeatureContribution] = Field(default_factory=list)
    neutral: List[FeatureContribution] = Field(default_factory=list)
    feature_contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="All features sorted descending by |contribution|",
    )
    confidence: str = Field(..., description="Signal quality: 'high', 'medium', or 'low'")
    shortlist: bool = Field(..., description="Whether this match is recommended for shortlisting")
    shortlist_reason: str = Field(..., description="Human-readable shortlist rationale")
    match_score: float = Field(..., ge=0.0, le=1.0)
    factors_met: int = Field(..., ge=0, description="Number of threshold factors met")
    factors_total: int = Field(..., ge=0, description="Total threshold factors evaluated")


class MatchScoreResponse(BaseModel):
    student_id: str
    job_id: str
    match_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: MatchExplanation
    timestamp: str

class JobRecommendationRequest(BaseModel):
    student_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RecommendedJob(BaseModel):
    job_id: str
    match_score: float
    explanation: List[str]

class JobRecommendationResponse(BaseModel):
    student_id: str
    recommended_jobs: List[RecommendedJob]
    total_evaluated: int
    timestamp: str

# ---------------------------------------------------------------------------
# Task 03 — Ranking Response Schemas
# ---------------------------------------------------------------------------

class RankedResult(BaseModel):
    """A single ranked item (job or candidate) with its AI score and explanation."""
    id: str = Field(..., description="Job ID or Student ID of the ranked item")
    score: float = Field(..., ge=0.0, le=1.0, description="AI-predicted relevance score [0,1]")
    explanation: List[str] = Field(default_factory=list, description="Human-readable scoring factors (legacy)")
    explanation_payload: Optional[ExplanationPayload] = Field(
        default=None,
        description="Structured explainability payload (Task 04+)",
    )


class StudentRankingResponse(BaseModel):
    """Ranked list of jobs returned for a specific student."""
    student_id: str
    ranked_jobs: List[RankedResult]
    total_evaluated: int = Field(..., description="Total number of jobs evaluated before ranking")
    timestamp: str


class JobRankingResponse(BaseModel):
    """Ranked list of candidates returned for a specific job."""
    job_id: str
    ranked_candidates: List[RankedResult]
    total_evaluated: int = Field(..., description="Total number of candidates evaluated before ranking")
    timestamp: str


if __name__ == "__main__":
    import json
    # Simple validation test
    print("Validating schemas...")
    dummy_student = StudentFeatures(
        student_id="STU-1", 
        skills_hard=["Python"], 
        years_experience=2.0
    )
    dummy_job = JobFeatures(
        job_id="JOB-1", 
        required_skills=["Python"], 
        salary_min=80000, 
        salary_max=100000
    )
    req = MatchScoreRequest(student=dummy_student, job=dummy_job)
    print("Schema Validation Successful!")
    print("Example Request JSON:")
    print(req.model_dump_json(indent=2))
