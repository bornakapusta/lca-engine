import importlib
import os
import pathlib
import sys
import types


PACKAGE_ROOT = pathlib.Path(__file__).parent
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")
package = types.ModuleType("recruiting_agent")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("recruiting_agent", package)

data_service = importlib.import_module("recruiting_agent.data_service")
recruiting_agent = importlib.import_module("recruiting_agent.recruiting_agent")


def test_add_candidate_skill_persists_to_record_and_profile():
    candidate_id = "CAND-90001"
    record = data_service.CANDIDATES[candidate_id]
    original_skills = list(record["skills"])
    data_service._PROFILES.clear()

    try:
        recruiting_agent.build_candidate_profile.invoke({"candidate_id": candidate_id})
        result = data_service.add_candidate_skill(candidate_id, "gRPC")

        assert result == {
            "updated": True,
            "already_present": False,
            "found": True,
            "skills": original_skills + ["gRPC"],
        }
        assert "gRPC" in data_service.fetch_skills(candidate_id)

        profile = recruiting_agent.build_candidate_profile.invoke(
            {"candidate_id": candidate_id}
        )["candidate_profile"]
        assert "gRPC" in profile["skills"]
    finally:
        record["skills"] = original_skills
        data_service._PROFILES.clear()


def test_score_candidate_retries_invalid_breakdown_and_returns_fixed_maxima(monkeypatch):
    valid = recruiting_agent.CandidateScore(
        score=75,
        justification="Required skills present: Python. Missing required skills: Rust.",
        rubric_breakdown=recruiting_agent.RubricBreakdown(
            experience=25, skills_match=30, seniority_fit=20
        ),
    )
    invalid = recruiting_agent.CandidateScore(
        score=100,
        justification="Required skills present: Python. Missing required skills: Rust.",
        rubric_breakdown=recruiting_agent.RubricBreakdown(
            experience=30, skills_match=40, seniority_fit=20
        ),
    )

    class FakeScoringLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return invalid if self.calls == 1 else valid

    scoring_llm = FakeScoringLLM()
    monkeypatch.setattr(recruiting_agent, "_scoring_llm", scoring_llm)
    result = recruiting_agent.score_candidate.invoke(
        {
            "candidate_profile": {"candidate_id": "CAND-90001", "skills": ["Python"]},
            "job_description": {
                "required_skills": ["Python", "Rust"],
                "min_years_experience": 3,
                "description": "Build services.",
            },
        }
    )

    assert scoring_llm.calls == 2
    assert result["max_score"] == 100
    assert result["rubric_breakdown"] == {
        "experience": 25.0,
        "skills_match": 30.0,
        "seniority_fit": 20.0,
        "experience_max": 30,
        "skills_match_max": 40,
        "seniority_fit_max": 30,
    }


def test_score_candidate_rejects_two_invalid_breakdowns(monkeypatch):
    invalid = recruiting_agent.CandidateScore(
        score=100,
        justification="Required skills present: Python. Missing required skills: Rust.",
        rubric_breakdown=recruiting_agent.RubricBreakdown(
            experience=30, skills_match=40, seniority_fit=20
        ),
    )

    class FakeScoringLLM:
        def invoke(self, messages):
            return invalid

    monkeypatch.setattr(recruiting_agent, "_scoring_llm", FakeScoringLLM())
    result = recruiting_agent.score_candidate.invoke(
        {
            "candidate_profile": {"skills": ["Python"]},
            "job_description": {
                "required_skills": ["Python"],
                "min_years_experience": 3,
                "description": "Build services.",
            },
        }
    )

    assert result == {
        "score": None,
        "error": "Scoring model returned an invalid rubric breakdown.",
    }
