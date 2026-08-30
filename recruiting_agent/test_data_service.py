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
