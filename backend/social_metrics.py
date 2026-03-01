"""
Social Metrics: Compute pairwise social distances and engagement scores.
"""

import math
from typing import Optional
from cv_pipeline import PersonData

# ---------------------------------------------------------------------------
# Orientation engagement score (0-1): how much are two people facing each other?
# ---------------------------------------------------------------------------

ORIENTATION_ENGAGEMENT = {
    "FACING_CAMERA": 0.8,
    "ANGLED":        0.6,
    "PROFILE_LEFT":  0.3,
    "PROFILE_RIGHT": 0.3,
    "TURNED_AWAY":   0.1,
    "UNKNOWN":       0.5,
}

EXPRESSION_ENGAGEMENT = {
    "LAUGHING":  1.0,
    "SMILING":   0.9,
    "TALKING":   0.8,
    "SURPRISED": 0.7,
    "NEUTRAL":   0.4,
    "FOCUSED":   0.5,
    "UNKNOWN":   0.4,
}


def _pixel_distance(p1: PersonData, p2: PersonData) -> float:
    return math.sqrt(
        (p1.face_center[0] - p2.face_center[0]) ** 2
        + (p1.face_center[1] - p2.face_center[1]) ** 2
    )


def _depth_adjusted_distance(p1: PersonData, p2: PersonData, img_diagonal: float) -> float:
    """
    Pixel distance normalised by image diagonal, then adjusted for depth difference.
    Returns 0-1 (lower = physically closer).
    """
    pixel_dist = _pixel_distance(p1, p2)
    norm_dist = pixel_dist / (img_diagonal + 1e-6)

    depth_diff = abs(p1.estimated_depth - p2.estimated_depth)
    # Penalise large depth differences (people on very different planes)
    adjusted = norm_dist + depth_diff * 0.05
    return min(adjusted, 1.0)


def _orientation_score(p1: PersonData, p2: PersonData) -> float:
    """Score 0-1: higher if both people tend to face each other."""
    o1 = ORIENTATION_ENGAGEMENT.get(p1.body_orientation, 0.5)
    o2 = ORIENTATION_ENGAGEMENT.get(p2.body_orientation, 0.5)
    return (o1 + o2) / 2.0


def _expression_score(p1: PersonData, p2: PersonData) -> float:
    e1 = EXPRESSION_ENGAGEMENT.get(p1.expression, 0.4)
    e2 = EXPRESSION_ENGAGEMENT.get(p2.expression, 0.4)
    return (e1 + e2) / 2.0


def compute_pairwise_matrix(
    persons: list[PersonData],
    img_w: int,
    img_h: int,
) -> dict[tuple[str, str], float]:
    """
    Returns pairwise social closeness scores (0-100, higher = more connected).
    Key: (person_id_a, person_id_b) with a < b alphabetically.
    """
    img_diagonal = math.sqrt(img_w ** 2 + img_h ** 2)
    matrix: dict[tuple[str, str], float] = {}

    for i, p1 in enumerate(persons):
        for j, p2 in enumerate(persons):
            if j <= i:
                continue

            d = _depth_adjusted_distance(p1, p2, img_diagonal)
            proximity = 1.0 - d           # 0-1, higher = closer

            orient = _orientation_score(p1, p2)
            expr = _expression_score(p1, p2)

            raw = 0.5 * proximity + 0.2 * orient + 0.3 * expr
            score = round(raw * 100, 1)

            key = (p1.person_id, p2.person_id)
            matrix[key] = score

    return matrix


def assign_social_scores(
    persons: list[PersonData],
    img_w: int,
    img_h: int,
) -> list[PersonData]:
    """
    Compute per-person social engagement score and social rank.
    Modifies persons in place and returns them sorted by rank.
    """
    if not persons:
        return persons

    if len(persons) == 1:
        p = persons[0]
        base = EXPRESSION_ENGAGEMENT.get(p.expression, 0.4) * 0.7
        orient_bonus = ORIENTATION_ENGAGEMENT.get(p.body_orientation, 0.5) * 0.3
        p.social_engagement_score = round((base + orient_bonus) * 100, 1)
        p.social_rank = 1
        return persons

    matrix = compute_pairwise_matrix(persons, img_w, img_h)
    id_map = {p.person_id: p for p in persons}

    # Average closeness score with all other persons
    totals: dict[str, float] = {p.person_id: 0.0 for p in persons}
    counts: dict[str, int] = {p.person_id: 0 for p in persons}

    for (pid_a, pid_b), score in matrix.items():
        totals[pid_a] = totals.get(pid_a, 0.0) + score
        totals[pid_b] = totals.get(pid_b, 0.0) + score
        counts[pid_a] = counts.get(pid_a, 0) + 1
        counts[pid_b] = counts.get(pid_b, 0) + 1

    for p in persons:
        n = counts[p.person_id]
        avg_pair_score = totals[p.person_id] / max(n, 1)

        # Blend with individual expression score
        expr_bonus = EXPRESSION_ENGAGEMENT.get(p.expression, 0.4) * 20.0
        p.social_engagement_score = round(min(avg_pair_score * 0.8 + expr_bonus, 100.0), 1)

    # Rank by descending score
    sorted_persons = sorted(persons, key=lambda p: p.social_engagement_score, reverse=True)
    for rank, p in enumerate(sorted_persons, start=1):
        p.social_rank = rank

    return sorted_persons


def social_ranking_ids(persons: list[PersonData]) -> list[str]:
    """Return person IDs sorted by social rank."""
    return [p.person_id for p in sorted(persons, key=lambda p: p.social_rank)]
