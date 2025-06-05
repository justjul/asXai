import os
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from fastapi import Request, HTTPException, Depends
from typing import List
import config
from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)

DEV_BYPASS_TOKEN = os.getenv("DEV_BYPASS_TOKEN", None)
ADMIN_UIDS = os.getenv("ADMIN_UIDS", "").split(",")

key_dir = os.getenv("GOOGLE_CREDENTIALS")
key_path = os.path.join(key_dir, 'firebaseKey.json')
if not key_path or not os.path.exists(key_path):
    raise RuntimeError(
        "Firebase service account key not found. Set GOOGLE_CREDENTIALS env variable.")

cred = credentials.Certificate(key_path)
firebase_admin.initialize_app(cred)


def verify_token(request: Request):
    dev_bypass = request.headers.get("X-Dev-Bypass")
    if dev_bypass == DEV_BYPASS_TOKEN:
        logger.info("Authentication bypassed for dev user")
        return {"uid": "dev_user"}
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid auth header")

    token = auth_header.split(" ")[1]
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token  # includes "uid"
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def set_admin_claim(target_uids: List[str] | None = None):
    if not target_uids:
        target_uids = ADMIN_UIDS
    try:
        for uid in target_uids:
            user_record = firebase_auth.get_user(uid)
            existing = user_record.custom_claims or {}
            new_claims = {**existing, "admin": True}
            firebase_auth.set_custom_user_claims(uid, new_claims)
            logger.info(f"admin rights set for user {uid}")
    except Exception as e:
        raise logger.error(f"Failed to set admin claim for {target_uids}: {e}")


def revoke_admin_claim(target_uids: list[str] | None = None):
    try:
        for uid in target_uids:
            user_record = firebase_auth.get_user(uid)
            existing = user_record.custom_claims or {}
            if not existing.get("admin"):
                # already not an admin, skip
                continue

            # Remove the "admin" key but keep any other claims
            existing.pop("admin", None)
            firebase_auth.set_custom_user_claims(uid, existing)
            logger.info(f"admin rights revoked for user {uid}")
    except Exception as e:
        raise logger.error(
            f"Failed to revoke admin rights for {target_uids}: {e}")
