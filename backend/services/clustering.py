import logging

import numpy as np

logger = logging.getLogger("video_redaction.clustering")

COSINE_SIM_THRESHOLD = 0.42
MERGE_BEST_LINK_THRESHOLD = 0.38
CENTROID_MERGE_THRESHOLD = 0.40
MAX_MERGE_PASSES = 10


def cosine_sim(a, b):
    return float(np.dot(a, b))


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def update_centroid(cluster):
    centroid = np.mean(cluster["all_encodings"], axis=0)
    cluster["centroid"] = normalize_vector(centroid)


def max_pairwise_similarity(cluster_a, cluster_b, sample_limit=30):
    """Compare clusters by their strongest member-to-member match.

    Centroids can drift when one person appears across very different poses,
    so this best-link check catches matches that a centroid-only pass misses.
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


def representative_encoding(cluster):
    """Use the sharpest, highest-confidence face as the cluster's anchor."""
    best_enc = cluster["all_encodings"][0]
    best_score = -1.0
    for enc, app in zip(cluster["all_encodings"], cluster["appearances"]):
        score = float(app.get("det_score", 0) or 0) + float(app.get("sharpness", 0) or 0) * 0.01
        if score > best_score:
            best_score = score
            best_enc = enc
    return best_enc


def absorb_cluster(target, source):
    target["appearances"].extend(source["appearances"])
    target["all_encodings"].extend(source["all_encodings"])
    update_centroid(target)
    if source.get("best_score", 0) > target.get("best_score", 0):
        target["best_snap"] = source["best_snap"]
        target["best_area"] = source["best_area"]
        target["best_score"] = source["best_score"]


def cluster_faces(all_faces, similarity_threshold=None):
    """Cluster InsightFace embeddings into stable people for the editor."""
    if similarity_threshold is None:
        similarity_threshold = COSINE_SIM_THRESHOLD

    if not all_faces:
        return []

    faces_with_enc = [f for f in all_faces if f.get("encoding") is not None]
    faces_without = [f for f in all_faces if f.get("encoding") is None]

    clusters = []

    # First assign each detection greedily using both the centroid and direct
    # member matches. This keeps profile/frontal views from splitting too early.
    for face in faces_with_enc:
        enc = np.array(face["encoding"], dtype=np.float32)
        enc = normalize_vector(enc)

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
            update_centroid(best_cluster)
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

    # Merge clusters whose averaged embeddings are close.
    for merge_pass in range(MAX_MERGE_PASSES):
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
                    absorb_cluster(current, clusters[j])
                    used.add(j)
                    merged = True
            new_clusters.append(current)
        clusters = new_clusters
        if not merged:
            break

    # Then merge by best direct member match so pose-heavy clusters can reunite.
    for merge_pass in range(MAX_MERGE_PASSES):
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
                best_link = max_pairwise_similarity(current, clusters[j])
                if best_link > MERGE_BEST_LINK_THRESHOLD:
                    absorb_cluster(current, clusters[j])
                    used.add(j)
                    merged = True
            new_clusters.append(current)
        clusters = new_clusters
        if not merged:
            break

    # Finally compare each cluster's best face crop as a last conservative pass.
    for merge_pass in range(MAX_MERGE_PASSES):
        merged = False
        new_clusters = []
        used = set()
        reps = [representative_encoding(c) for c in clusters]
        for i in range(len(clusters)):
            if i in used:
                continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if j in used:
                    continue
                sim = cosine_sim(reps[i], reps[j])
                if sim > similarity_threshold:
                    absorb_cluster(current, clusters[j])
                    used.add(j)
                    merged = True
                    reps[i] = representative_encoding(current)
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
