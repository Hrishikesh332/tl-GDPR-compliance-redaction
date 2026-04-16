import logging

import numpy as np

logger = logging.getLogger("video_redaction.clustering")

COSINE_SIM_THRESHOLD = 0.55


def cosine_sim(a, b):
    return float(np.dot(a, b))


def cluster_faces(all_faces, similarity_threshold=None):
    """Cluster face detections into unique identities using cosine similarity
    on InsightFace 512-d ArcFace embeddings. Two-pass merge for robustness.
    """
    if similarity_threshold is None:
        similarity_threshold = COSINE_SIM_THRESHOLD

    if not all_faces:
        return []

    faces_with_enc = [f for f in all_faces if f.get("encoding") is not None]
    faces_without = [f for f in all_faces if f.get("encoding") is None]

    clusters = []

    for face in faces_with_enc:
        enc = np.array(face["encoding"], dtype=np.float32)
        norm = np.linalg.norm(enc)
        if norm > 0:
            enc = enc / norm

        best_cluster = None
        best_sim = -1.0

        for cluster in clusters:
            sim = cosine_sim(enc, cluster["centroid"])
            if sim > similarity_threshold and sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster is not None:
            best_cluster["appearances"].append(face)
            best_cluster["all_encodings"].append(enc)
            best_cluster["centroid"] = np.mean(best_cluster["all_encodings"], axis=0)
            c_norm = np.linalg.norm(best_cluster["centroid"])
            if c_norm > 0:
                best_cluster["centroid"] /= c_norm
            area = face.get("crop_area", 0)
            score = face.get("det_score", 0) + face.get("sharpness", 0) * 0.01
            if score > best_cluster.get("best_score", 0):
                best_cluster["best_snap"] = face
                best_cluster["best_area"] = area
                best_cluster["best_score"] = score
        else:
            clusters.append({
                "centroid": enc.copy(),
                "all_encodings": [enc],
                "best_snap": face,
                "best_area": face.get("crop_area", 0),
                "best_score": face.get("det_score", 0) + face.get("sharpness", 0) * 0.01,
                "appearances": [face],
            })

    merged = True
    while merged:
        merged = False
        new_clusters = []
        used = set()
        for i in range(len(clusters)):
            if i in used:
                continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if j in used:
                    continue
                sim = cosine_sim(current["centroid"], clusters[j]["centroid"])
                if sim > similarity_threshold:
                    current["appearances"].extend(clusters[j]["appearances"])
                    current["all_encodings"].extend(clusters[j]["all_encodings"])
                    current["centroid"] = np.mean(current["all_encodings"], axis=0)
                    c_norm = np.linalg.norm(current["centroid"])
                    if c_norm > 0:
                        current["centroid"] /= c_norm
                    if clusters[j].get("best_score", 0) > current.get("best_score", 0):
                        current["best_snap"] = clusters[j]["best_snap"]
                        current["best_area"] = clusters[j]["best_area"]
                        current["best_score"] = clusters[j]["best_score"]
                    used.add(j)
                    merged = True
            new_clusters.append(current)
        clusters = new_clusters

    if faces_without:
        logger.info("Discarding %d face detections without embeddings", len(faces_without))

    logger.info("Clustered %d face detections into %d unique faces (cosine_sim_threshold=%.3f)",
                len(faces_with_enc), len(clusters), similarity_threshold)

    results = []
    for i, c in enumerate(clusters):
        best = c["best_snap"]
        stable_person_id = f"person_{i}"
        appearances = []
        for a in c["appearances"]:
            appearances.append({
                "frame_idx": a.get("frame_idx"),
                "timestamp": a.get("timestamp"),
                "bbox": a.get("bbox"),
            })

        centroid = c["centroid"]
        if centroid is not None and hasattr(centroid, "tolist"):
            centroid = centroid.tolist()

        results.append({
            "person_id": stable_person_id,
            "stable_person_id": stable_person_id,
            "cluster_person_id": stable_person_id,
            "snap_base64": best.get("snap_base64"),
            "bbox": best.get("bbox"),
            "encoding": centroid,
            "appearances": appearances,
            "appearance_count": len(c["appearances"]),
        })

    results.sort(key=lambda r: r["appearance_count"], reverse=True)
    return results


def cluster_objects(all_objects):
    if not all_objects:
        return []

    groups = {}
    for obj in all_objects:
        name = obj.get("identification", "unknown")
        if name not in groups:
            groups[name] = {"items": [], "best": None, "best_area": 0}
        groups[name]["items"].append(obj)
        area = obj.get("crop_area", 0)
        if area > groups[name]["best_area"]:
            groups[name]["best"] = obj
            groups[name]["best_area"] = area

    results = []
    for idx, (name, g) in enumerate(groups.items()):
        best = g["best"] or g["items"][0]
        appearances = []
        for item in g["items"]:
            appearances.append({
                "frame_idx": item.get("frame_idx"),
                "timestamp": item.get("timestamp"),
                "bbox": item.get("bbox"),
                "confidence": item.get("confidence"),
            })
        results.append({
            "object_id": f"object_{idx}",
            "identification": name,
            "snap_base64": best.get("snap_base64"),
            "bbox": best.get("bbox"),
            "appearances": appearances,
            "appearance_count": len(g["items"]),
        })
    return results
