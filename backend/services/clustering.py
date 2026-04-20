import logging

import numpy as np

logger = logging.getLogger("video_redaction.clustering")

COSINE_SIM_THRESHOLD = 0.42
MERGE_BEST_LINK_THRESHOLD = 0.38
CENTROID_MERGE_THRESHOLD = 0.40
MAX_MERGE_PASSES = 10


def cosine_sim(a, b):
    return float(np.dot(a, b))


def _normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _update_centroid(cluster):
    centroid = np.mean(cluster["all_encodings"], axis=0)
    cluster["centroid"] = _normalize(centroid)


def _max_pairwise_sim(cluster_a, cluster_b, sample_limit=30):
    """Best-link similarity: max cosine sim between any member of A and any member of B.

    For very large clusters, sub-sample to keep runtime bounded while still
    catching the strongest cross-cluster link.
    """
    encs_a = cluster_a["all_encodings"]
    encs_b = cluster_b["all_encodings"]
    if len(encs_a) > sample_limit:
        indices = np.linspace(0, len(encs_a) - 1, sample_limit, dtype=int)
        encs_a = [encs_a[i] for i in indices]
    if len(encs_b) > sample_limit:
        indices = np.linspace(0, len(encs_b) - 1, sample_limit, dtype=int)
        encs_b = [encs_b[i] for i in indices]

    mat_a = np.stack(encs_a)
    mat_b = np.stack(encs_b)
    sims = mat_a @ mat_b.T
    return float(sims.max())


def _representative_encoding(cluster):
    """Return the single highest-quality encoding in the cluster."""
    best_enc = cluster["all_encodings"][0]
    best_score = -1.0
    for enc, app in zip(cluster["all_encodings"], cluster["appearances"]):
        score = float(app.get("det_score", 0) or 0) + float(app.get("sharpness", 0) or 0) * 0.01
        if score > best_score:
            best_score = score
            best_enc = enc
    return best_enc


def _absorb_cluster(target, source):
    target["appearances"].extend(source["appearances"])
    target["all_encodings"].extend(source["all_encodings"])
    _update_centroid(target)
    if source.get("best_score", 0) > target.get("best_score", 0):
        target["best_snap"] = source["best_snap"]
        target["best_area"] = source["best_area"]
        target["best_score"] = source["best_score"]


def cluster_faces(all_faces, similarity_threshold=None):
    """Cluster face detections into unique identities using cosine similarity
    on InsightFace 512-d ArcFace embeddings.

    Uses three merging strategies for robustness:
      1. Initial greedy assignment (new face vs best cluster member + centroid)
      2. Centroid merge pass (merge clusters whose centroids are close)
      3. Best-link merge pass (merge clusters that share ANY close member pair)
    """
    if similarity_threshold is None:
        similarity_threshold = COSINE_SIM_THRESHOLD

    if not all_faces:
        return []

    faces_with_enc = [f for f in all_faces if f.get("encoding") is not None]
    faces_without = [f for f in all_faces if f.get("encoding") is None]

    clusters = []

    # ── Pass 1: greedy assignment ────────────────────────────────────
    # Each face is compared against BOTH the centroid and the best
    # individual member of every cluster.  This avoids the problem where
    # a centroid drifts due to averaging across poses/angles while
    # individual members would clearly match.
    for face in faces_with_enc:
        enc = np.array(face["encoding"], dtype=np.float32)
        enc = _normalize(enc)

        best_cluster = None
        best_sim = -1.0

        for cluster in clusters:
            centroid_sim = cosine_sim(enc, cluster["centroid"])
            member_max = max(
                cosine_sim(enc, m) for m in cluster["all_encodings"]
            ) if len(cluster["all_encodings"]) <= 60 else centroid_sim

            effective_sim = max(centroid_sim, member_max)

            if effective_sim > similarity_threshold and effective_sim > best_sim:
                best_sim = effective_sim
                best_cluster = cluster

        if best_cluster is not None:
            best_cluster["appearances"].append(face)
            best_cluster["all_encodings"].append(enc)
            _update_centroid(best_cluster)
            score = float(face.get("det_score", 0) or 0) + float(face.get("sharpness", 0) or 0) * 0.01
            if score > best_cluster.get("best_score", 0):
                best_cluster["best_snap"] = face
                best_cluster["best_area"] = face.get("crop_area", 0)
                best_cluster["best_score"] = score
        else:
            clusters.append({
                "centroid": enc.copy(),
                "all_encodings": [enc],
                "best_snap": face,
                "best_area": face.get("crop_area", 0),
                "best_score": float(face.get("det_score", 0) or 0) + float(face.get("sharpness", 0) or 0) * 0.01,
                "appearances": [face],
            })

    # ── Pass 2: centroid merge ───────────────────────────────────────
    for _pass in range(MAX_MERGE_PASSES):
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
                if sim > CENTROID_MERGE_THRESHOLD:
                    _absorb_cluster(current, clusters[j])
                    used.add(j)
                    merged = True
            new_clusters.append(current)
        clusters = new_clusters
        if not merged:
            break

    # ── Pass 3: best-link merge ──────────────────────────────────────
    # Even when centroids diverge (e.g. frontal vs profile clusters),
    # individual members may still clearly match.  Merge any pair of
    # clusters whose best cross-pair similarity exceeds the threshold.
    for _pass in range(MAX_MERGE_PASSES):
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
                best_link = _max_pairwise_sim(current, clusters[j])
                if best_link > MERGE_BEST_LINK_THRESHOLD:
                    _absorb_cluster(current, clusters[j])
                    used.add(j)
                    merged = True
            new_clusters.append(current)
        clusters = new_clusters
        if not merged:
            break

    # ── Pass 4: representative encoding merge ────────────────────────
    # Compare the single highest-quality face from each cluster as a
    # final safety net to catch clusters that slipped through.
    for _pass in range(MAX_MERGE_PASSES):
        merged = False
        new_clusters = []
        used = set()
        reps = [_representative_encoding(c) for c in clusters]
        for i in range(len(clusters)):
            if i in used:
                continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if j in used:
                    continue
                sim = cosine_sim(reps[i], reps[j])
                if sim > similarity_threshold:
                    _absorb_cluster(current, clusters[j])
                    used.add(j)
                    merged = True
                    reps[i] = _representative_encoding(current)
            new_clusters.append(current)
        clusters = new_clusters
        if not merged:
            break

    if faces_without:
        logger.info("Discarding %d face detections without embeddings", len(faces_without))

    logger.info(
        "Clustered %d face detections into %d unique faces "
        "(sim_threshold=%.3f, centroid_merge=%.3f, best_link=%.3f)",
        len(faces_with_enc), len(clusters),
        similarity_threshold, CENTROID_MERGE_THRESHOLD, MERGE_BEST_LINK_THRESHOLD,
    )

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
