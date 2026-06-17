"""Tests for the RAG layer's role-based access control over disaster categories"""
from __future__ import annotations

from chatbot.model.schemas import Permission, ResumeDocument, Role, User
from chatbot.rag.security_facade import AccessControl

def _admin() -> User:
    return User(user_id="admin1", role=Role.ADMIN)

def _ops_user() -> User:
    return User(user_id="ops1", role=Role.ANALYST, department="ClimateOps")

def _response_user() -> User:
    return User(user_id="resp1", role=Role.HR_MANAGER, department="DisasterResponse")

def test_admin_has_all_permissions(config_manager):
    ac = AccessControl(config_manager)
    admin = _admin()
    for perm in Permission:
        assert ac.check_permission(admin, perm) is True

def test_recruiter_read_only(config_manager):
    ac = AccessControl(config_manager)
    recruiter = User(user_id="rec1", role=Role.RECRUITER, department="ClimateOps")
    assert ac.check_permission(recruiter, Permission.READ) is True
    assert ac.check_permission(recruiter, Permission.WRITE) is False
    assert ac.check_permission(recruiter, Permission.DELETE) is False

def test_admin_has_no_category_filter(config_manager):
    ac = AccessControl(config_manager)
    assert ac.get_allowed_categories(_admin()) is None
    assert ac.create_filter(_admin()) is None

def test_department_categories_apply_to_non_admin(config_manager):
    ac = AccessControl(config_manager)
    cats = ac.get_allowed_categories(_ops_user())
    assert cats is not None
    # ClimateOps department maps to weather/climate disaster types.
    assert "Flood" in cats
    assert "Storm" in cats
    # DisasterResponse-only categories are not allowed.
    assert "Earthquake" not in cats

def test_filter_results_keeps_only_allowed_categories(config_manager):
    ac = AccessControl(config_manager)
    ops = _ops_user()
    docs = [
        {
            "document": ResumeDocument(
                page_content="2021 storm Ida",
                metadata={"id": "2021-0500-USA", "category": "Storm", "source": "emdat"},
            ),
            "score": 0.9,
            "method": "hybrid",
        },
        {
            "document": ResumeDocument(
                page_content="2010 Haiti earthquake",
                metadata={"id": "2010-0100-HTI", "category": "Earthquake", "source": "emdat"},
            ),
            "score": 0.8,
            "method": "hybrid",
        },
    ]
    filtered = ac.filter_results(ops, docs)
    cats = {r["document"].metadata.category for r in filtered}
    assert cats == {"Storm"}

def test_owner_override_grants_access(config_manager):
    ac = AccessControl(config_manager)
    ops = _ops_user()
    doc = {
        "document": ResumeDocument(
            page_content="earthquake owned by ops user",
            metadata={
                "id": "2015-0200-NPL",
                "category": "Earthquake",
                "source": "emdat",
                "owner_id": "ops1",
            },
        ),
        "score": 0.5,
        "method": "hybrid",
    }
    out = ac.filter_results(ops, [doc])
    assert len(out) == 1

def test_explicit_access_list_grants_access(config_manager):
    ac = AccessControl(config_manager)
    ops = _ops_user()
    doc = {
        "document": ResumeDocument(
            page_content="restricted volcanic event",
            metadata={
                "id": "2018-0300-IDN",
                "category": "Volcanic activity",
                "source": "emdat",
                "access_list": ["ops1", "resp1"],
            },
        ),
        "score": 0.5,
        "method": "hybrid",
    }
    assert len(ac.filter_results(ops, [doc])) == 1

def test_create_filter_for_empty_categories_yields_deny_all(config_manager):
    ac = AccessControl(config_manager)
    no_dept_user = User(user_id="noone", role=Role.ANALYST)
    f = ac.create_filter(no_dept_user)
    assert f == {"category": {"$in": ["__NONE__"]}}

def test_can_access_category(config_manager):
    ac = AccessControl(config_manager)
    assert ac.can_access_category(_admin(), "ANYTHING")
    assert ac.can_access_category(_response_user(), "Earthquake")
    assert not ac.can_access_category(_response_user(), "Flood")

def test_log_access_writes_jsonl(tmp_path, config_manager):
    ac = AccessControl(config_manager)
    user = _admin()
    ac.log_access(user, "search", "earthquake query", True)
    audit = config_manager.results_dir / "audit" / "access_audit.jsonl"
    assert audit.is_file()
    content = audit.read_text(encoding="utf-8")
    assert "admin1" in content
    assert "SUCCESS" in content
