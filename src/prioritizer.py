"""Review Priority Index (RPI): a transparent, rule-based ordering of images for human review.

RPI answers "which image should an inspector look at first?". It is NOT:
  * model confidence (how sure the network is that a hotspot pattern is present),
  * a physical fault severity (needs calibrated temperatures / expert labels - not available),
  * an electrical-risk estimate.

Components (each normalised to 0..1):
  max_confidence   strongest detection confidence above the threshold
  detection_count  number of detections above the threshold, saturating at ``count_saturation``
  thermal_contrast relative raw-count contrast (z-score) of the strongest-contrast detection;
                   only available for radiometric images - otherwise its weight is redistributed
  uncertainty      how close the most borderline candidate is to the threshold (within +/-0.2);
                   borderline cases are exactly where a human look is most valuable

RPI = 100 * sum(w_i * c_i) / sum(w_i over available components). If nothing is detected and
no candidate is borderline, RPI = 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

UNCERTAINTY_BAND = 0.20

ACTIONS = {
    "High": "Prioritise for on-site verification (handheld IR / I-V curve check) at the next maintenance window.",
    "Medium": "Schedule manual review of the image by a qualified inspector; verify on site if confirmed.",
    "Low": "No immediate action suggested by the model; keep for trend comparison in the next inspection cycle.",
}


@dataclass
class PriorityResult:
    score: float
    category: str
    components: dict
    weights_used: dict
    provisional: bool
    recommended_action: str
    explanation: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": self.score, "category": self.category, "components": self.components,
                "weights_used": self.weights_used, "provisional": self.provisional,
                "recommended_action": self.recommended_action, "explanation": self.explanation}


def compute_priority(detections: list, candidates: list, threshold: float, cfg: dict,
                     radiometric: bool = False) -> PriorityResult:
    """``detections``: Detection objects above ``threshold``; ``candidates``: all model outputs."""
    w = dict(cfg["weights"])
    sat = max(1, int(cfg.get("count_saturation", 4)))
    zsat = float(cfg.get("contrast_z_saturation", 12.0))

    comp = {}
    comp["max_confidence"] = max((d.confidence for d in detections), default=0.0)
    comp["detection_count"] = min(len(detections), sat) / sat
    zs = [d.contrast.get("contrast_z") for d in detections
          if d.contrast and d.contrast.get("contrast_z") is not None]
    if radiometric and detections and zs:
        comp["thermal_contrast"] = max(0.0, min(1.0, max(zs) / zsat))
    else:
        comp["thermal_contrast"] = None
    near = [1 - abs(c.confidence - threshold) / UNCERTAINTY_BAND for c in candidates
            if abs(c.confidence - threshold) <= UNCERTAINTY_BAND]
    comp["uncertainty"] = max(near, default=0.0)

    used = {k: v for k, v in w.items() if comp.get(k) is not None}
    total_w = sum(used.values()) or 1.0
    used = {k: v / total_w for k, v in used.items()}
    if not detections and comp["uncertainty"] == 0:
        score = 0.0
    else:
        score = 100.0 * sum(used[k] * comp[k] for k in used)
    score = round(min(100.0, max(0.0, score)), 1)

    th = cfg["thresholds"]
    category = "High" if score >= th["high"] else "Medium" if score >= th["medium"] else "Low"

    expl = []
    if detections:
        expl.append(f"{len(detections)} suspected hotspot(s) at confidence >= {threshold:.2f}; "
                    f"strongest {comp['max_confidence']:.2f}.")
    else:
        expl.append(f"No detection at confidence >= {threshold:.2f}.")
    if comp["thermal_contrast"] is not None:
        expl.append(f"Relative raw-count contrast z = {max(zs):.1f} (uncalibrated; not a temperature).")
    else:
        expl.append("Thermal-contrast component unavailable (no radiometric raw data) - weight redistributed.")
    if comp["uncertainty"] > 0:
        expl.append("At least one model output lies close to the threshold - human verification recommended.")
    expl.append("Score is provisional: no calibrated temperatures or expert severity labels are available.")

    return PriorityResult(score=score, category=category,
                          components={k: (None if v is None else round(v, 3)) for k, v in comp.items()},
                          weights_used={k: round(v, 3) for k, v in used.items()},
                          provisional=True, recommended_action=ACTIONS[category], explanation=expl)
