"""Role-based access control for the RAG layer (audit + category filtering)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from chatbot.model.schemas import Permission, Role, User
from chatbot.security.guardrails import sanitize_for_log

logger = logging.getLogger(__name__)


class AccessControl:
    def __init__(self, config_manager) -> None:
        self.config = config_manager
        access_config = config_manager.get_access_control_config()

        self.role_permissions: Dict[Role, Set[Permission]] = {
            Role.ADMIN: {
                Permission.READ,
                Permission.WRITE,
                Permission.DELETE,
                Permission.ANALYZE,
            },
            Role.HR_MANAGER: {Permission.READ, Permission.ANALYZE},
            Role.RECRUITER: {Permission.READ},
            Role.ANALYST: {Permission.READ, Permission.ANALYZE},
        }
        self.department_categories = access_config.department_categories

        self._audit_dir = Path(config_manager.results_dir) / "audit"
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._audit_file = self._audit_dir / "access_audit.jsonl"

    @staticmethod
    def validate_user(user_data: Dict[str, Any]) -> User:
        return User(**user_data)

    def check_permission(self, user: User, permission: Permission) -> bool:
        if not isinstance(user, User):
            return False
        role_key = Role(user.role) if isinstance(user.role, str) else user.role
        return permission in self.role_permissions.get(role_key, set())

    def get_allowed_categories(self, user: User) -> Optional[Set[str]]:
        if not isinstance(user, User):
            return None
        role_key = Role(user.role) if isinstance(user.role, str) else user.role
        if role_key == Role.ADMIN:
            return None
        if user.allowed_categories:
            return set(user.allowed_categories)
        if user.department and user.department in self.department_categories:
            return set(self.department_categories[user.department])
        return set()

    def create_filter(self, user: User) -> Optional[Dict[str, Any]]:
        if not isinstance(user, User):
            return None
        allowed = self.get_allowed_categories(user)
        if allowed is None:
            return None
        if not allowed:
            return {"category": {"$in": ["__NONE__"]}}
        return {"category": {"$in": list(allowed)}}

    def filter_results(
        self, user: User, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not isinstance(user, User) or not results:
            return results or []
        role_key = Role(user.role) if isinstance(user.role, str) else user.role
        if role_key == Role.ADMIN:
            return results
        allowed = self.get_allowed_categories(user)
        if allowed is None:
            return results

        out: List[Dict[str, Any]] = []
        for r in results:
            document = r.get("document") if isinstance(r, dict) else None
            metadata = getattr(document, "metadata", None)
            if isinstance(metadata, dict):
                cat = metadata.get("category", "Unknown")
                access_list = metadata.get("access_list")
                owner_id = metadata.get("owner_id")
            else:
                cat = getattr(metadata, "category", "Unknown")
                access_list = getattr(metadata, "access_list", None)
                owner_id = getattr(metadata, "owner_id", None)
            if owner_id and owner_id == user.user_id:
                out.append(r)
            elif access_list and user.user_id in access_list:
                out.append(r)
            elif cat in allowed:
                out.append(r)
        return out

    def log_access(self, user: User, action: str, resource: str, success: bool) -> None:
        if not isinstance(user, User):
            return
        status = "SUCCESS" if success else "DENIED"
        safe_resource = sanitize_for_log(resource, 200)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user.user_id,
            "role": str(user.role),
            "department": user.department,
            "action": action,
            "resource": safe_resource,
            "status": status,
        }
        try:
            with open(self._audit_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Could not write audit entry: %s", exc)

    def get_user_permissions(self, user: User) -> Set[Permission]:
        if not isinstance(user, User):
            return set()
        role_key = Role(user.role) if isinstance(user.role, str) else user.role
        return self.role_permissions.get(role_key, set())

    def can_access_category(self, user: User, category: str) -> bool:
        if not isinstance(user, User) or not category:
            return False
        allowed = self.get_allowed_categories(user)
        if allowed is None:
            return True
        return category in allowed
