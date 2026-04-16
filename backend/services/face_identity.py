def get_face_identity(face, fallback_index=None):
    if not isinstance(face, dict):
        if fallback_index is None:
            return ""
        return f"person_{fallback_index}"

    for key in ("stable_person_id", "cluster_person_id", "person_id"):
        value = str(face.get(key) or "").strip()
        if value:
            return value

    if fallback_index is None:
        return ""
    return f"person_{fallback_index}"


def ensure_face_identity(face, fallback_index=None):
    identity = get_face_identity(face, fallback_index=fallback_index)
    if isinstance(face, dict) and identity:
        face["stable_person_id"] = identity
        if not str(face.get("person_id") or "").strip():
            face["person_id"] = identity
    return identity
