# PlaceMux Matching Foundation: Data Model & API Contract

## 1. Student <-> Job Feature Space

To compute an accurate and interpretable match score between a candidate and a job opening, we define the following feature space. These features serve as inputs to the Matching ML model.

### 1.1 Student (Candidate) Features
These features represent the candidate's profile, including their validated assessment scores from Phase 1.

| Feature Name | Type | Description |
|--------------|------|-------------|
| `student_id` | String | Unique identifier for the student |
| `skills_hard` | List[String] | Technical skills (e.g., Python, SQL, React) |
| `skills_soft` | List[String] | Soft skills (e.g., Communication, Leadership) |
| `years_experience`| Float | Total years of professional experience |
| `education_level` | Int | Ordinal encoding of highest degree (1=High School, 2=Bachelors, 3=Masters, 4=PhD) |
| `expected_salary` | Float | Expected annual salary in USD |
| `preferred_location`| String | Preferred city/region for work |
| `remote_preference` | String | Preference for remote work ('Remote', 'Hybrid', 'On-site') |
| `coding_score` | Float | Normalized score [0-1] from technical assessments |
| `communication_score`| Float | Normalized score [0-1] from communication assessments |

### 1.2 Job Features
These features represent the requirements and offerings of the specific job posting.

| Feature Name | Type | Description |
|--------------|------|-------------|
| `job_id` | String | Unique identifier for the job |
| `required_skills` | List[String] | Must-have technical skills |
| `preferred_skills`| List[String] | Nice-to-have technical skills |
| `min_experience` | Float | Minimum years of experience required |
| `max_experience` | Float | Maximum years of experience expected (can be null) |
| `min_education` | Int | Ordinal encoding of minimum required degree |
| `salary_min` | Float | Bottom of the salary band |
| `salary_max` | Float | Top of the salary band |
| `job_location` | String | City/region of the job |
| `work_model` | String | 'Remote', 'Hybrid', or 'On-site' |

### 1.3 Derived Matching Features
The ML pipeline will compute these interaction features on-the-fly before scoring.

| Feature Name | Type | Description |
|--------------|------|-------------|
| `skill_overlap_ratio` | Float | Jaccard similarity between candidate skills and required skills |
| `experience_gap` | Float | `years_experience` - `min_experience`. Negative means underqualified. |
| `salary_match_ratio` | Float | `salary_max` / `expected_salary`. Higher is better. |
| `location_match` | Boolean | True if `preferred_location` matches `job_location` OR `work_model` is Remote |
| `education_met` | Boolean | True if `education_level` >= `min_education` |

---

## 2. Matching API Contract

The ML Matching Service exposes the following endpoints to the PlaceMux Backend. Communication is handled via REST/JSON. 

### 2.1 Score Endpoint
`POST /api/v1/match/score`

Calculates a match score for a specific Student-Job pair.

**Request Payload:**
```json
{
  "student": {
    "student_id": "STU-12345",
    "skills_hard": ["Python", "Machine Learning", "SQL"],
    "skills_soft": ["Communication"],
    "years_experience": 2.5,
    "education_level": 2,
    "expected_salary": 85000,
    "preferred_location": "San Francisco",
    "remote_preference": "Hybrid",
    "coding_score": 0.92,
    "communication_score": 0.85
  },
  "job": {
    "job_id": "JOB-9876",
    "required_skills": ["Python", "SQL"],
    "preferred_skills": ["AWS"],
    "min_experience": 2.0,
    "max_experience": 5.0,
    "min_education": 2,
    "salary_min": 80000,
    "salary_max": 100000,
    "job_location": "San Francisco",
    "work_model": "Hybrid"
  }
}
```

**Response Payload:**
```json
{
  "student_id": "STU-12345",
  "job_id": "JOB-9876",
  "match_score": 0.89,
  "confidence": 0.95,
  "explanation": {
    "positive_factors": ["High skill overlap (100% required skills met)", "Location match"],
    "negative_factors": ["Missing preferred skill: AWS"]
  },
  "timestamp": "2026-07-11T21:03:00Z"
}
```

### 2.2 Recommendation Endpoint
`POST /api/v1/match/recommend-jobs`

Given a student profile, recommends the top K matching jobs. (Assumes the ML Service has access to an embedded job index/feature store).

**Request Payload:**
```json
{
  "student_id": "STU-12345",
  "top_k": 10,
  "filters": {
    "remote_only": false,
    "min_salary": 80000
  }
}
```

**Response Payload:**
```json
{
  "student_id": "STU-12345",
  "recommended_jobs": [
    {
      "job_id": "JOB-9876",
      "match_score": 0.89,
      "explanation": ["High skill overlap", "Location match"]
    }
  ],
  "total_evaluated": 1500,
  "timestamp": "2026-07-11T21:03:00Z"
}
```
