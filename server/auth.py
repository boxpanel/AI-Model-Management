"""登录认证：PBKDF2 密码哈希 + 内存会话 Token（无需第三方依赖）。"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from typing import Any, Optional


class AuthService:
    def __init__(self) -> None:
        self._tokens: dict[str, tuple[str, float]] = {}  # token -> (username, expire)
        self._lock = threading.Lock()
        self._token_ttl = 7 * 24 * 3600  # 7 天

    def hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
        return f"{salt}${digest}"

    def verify_password(self, password: str, stored: str) -> bool:
        try:
            salt, digest = stored.split("$", 1)
            calc = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
            return hmac.compare_digest(calc, digest)
        except Exception:
            return False

    def issue_token(self, username: str) -> str:
        token = secrets.token_hex(32)
        with self._lock:
            self._tokens[token] = (username, time.time() + self._token_ttl)
        return token

    def validate_token(self, token: str) -> Optional[str]:
        with self._lock:
            entry = self._tokens.get(token)
            if not entry:
                return None
            username, expire = entry
            if time.time() > expire:
                self._tokens.pop(token, None)
                return None
            return username

    def revoke_token(self, token: str) -> None:
        with self._lock:
            self._tokens.pop(token, None)


auth_service = AuthService()


def ensure_default_user(db: Any, username: str = "admin", password: str = "") -> str:
    """确保默认用户存在：不存在则创建（密码为空时随机生成），返回实际密码；已存在返回空串。"""
    if db.get_user(username):
        return ""
    pwd = password or secrets.token_urlsafe(12)
    db.create_user(username, auth_service.hash_password(pwd))
    return pwd
