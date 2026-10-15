"""
Curated knowledge base for the educational lung and pneumonia assistant.

The assistant answers from these entries rather than from the language model's
own recall, so that every claim it makes can be traced to a named source. Each
entry carries the URL it was summarised from; the retriever returns entries and
the answer cites them. An entry is a short factual summary, not a quotation.

Scope is deliberately narrow: general lung health, pneumonia, chest
radiography, terminology, and how this project works. Nothing here is specific
to an individual, and nothing here is a treatment instruction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KBEntry:
    key: str
    title: str
    body: str
    source_name: str
    source_url: str
    tags: tuple[str, ...] = field(default_factory=tuple)


KB: tuple[KBEntry, ...] = (
    KBEntry(
        key="pneumonia_overview",
        title="What pneumonia is",
        body=(
            "Pneumonia is an infection that inflames the air sacs (alveoli) of one or both "
            "lungs. The sacs may fill with fluid or pus, which is what makes breathing "
            "harder and reduces the amount of oxygen reaching the blood. It can be caused "
            "by bacteria, viruses or fungi, and severity ranges from mild to life "
            "threatening."
        ),
        source_name="World Health Organization",
        source_url="https://www.who.int/news-room/fact-sheets/detail/pneumonia",
        tags=("pneumonia", "definition", "alveoli", "infection", "what is"),
    ),
    KBEntry(
        key="pneumonia_symptoms",
        title="Commonly described symptoms",
        body=(
            "Commonly described symptoms include cough (sometimes productive), fever, "
            "chills, difficulty breathing or rapid breathing, and chest discomfort that is "
            "often worse on breathing in. In young children, fast or laboured breathing and "
            "chest indrawing are emphasised. Older adults may show fewer or milder signs, "
            "sometimes confusion rather than fever."
        ),
        source_name="World Health Organization",
        source_url="https://www.who.int/news-room/fact-sheets/detail/pneumonia",
        tags=("symptoms", "signs", "cough", "fever", "breathing", "chest pain"),
    ),
    KBEntry(
        key="pneumonia_children",
        title="Pneumonia in children",
        body=(
            "Pneumonia is a leading infectious cause of death in children under five "
            "worldwide. Risk is raised by malnutrition, lack of exclusive breastfeeding, "
            "indoor air pollution and crowding. The dataset this project uses consists of "
            "paediatric chest radiographs from children aged roughly one to five years, "
            "which is why its findings do not transfer to adults."
        ),
        source_name="World Health Organization",
        source_url="https://www.who.int/news-room/fact-sheets/detail/pneumonia",
        tags=("children", "paediatric", "pediatric", "kids", "infant", "mortality"),
    ),
    KBEntry(
        key="prevention",
        title="Prevention at a population level",
        body=(
            "Vaccination (including against Streptococcus pneumoniae, Haemophilus "
            "influenzae type b, measles and pertussis), adequate nutrition, hand hygiene, "
            "and reducing indoor air pollution are the measures most often described for "
            "reducing pneumonia incidence. These are general public-health measures, not "
            "advice for any individual."
        ),
        source_name="World Health Organization",
        source_url="https://www.who.int/news-room/fact-sheets/detail/pneumonia",
        tags=("prevention", "vaccine", "vaccination", "hygiene", "avoid", "reduce risk"),
    ),
    KBEntry(
        key="cxr_basics",
        title="What a chest X-ray is",
        body=(
            "A chest radiograph passes a small dose of ionising radiation through the chest "
            "onto a detector. Dense tissue such as bone absorbs more and appears lighter; "
            "air-filled lung appears darker. It is quick and widely available, which is why "
            "it is a common first imaging test for suspected chest infection."
        ),
        source_name="NIH National Heart, Lung, and Blood Institute",
        source_url="https://www.nhlbi.nih.gov/health/chest-x-ray",
        tags=("x-ray", "xray", "radiograph", "imaging", "how does", "chest x-ray"),
    ),
    KBEntry(
        key="cxr_appearance",
        title="How infection can look on a chest X-ray",
        body=(
            "Areas where air has been displaced by fluid or inflammatory material absorb "
            "more radiation and appear as lighter regions, often described as opacity or "
            "consolidation. Appearance varies with cause, timing and the patient, and other "
            "conditions can look similar, so an image alone is not sufficient to establish "
            "a cause."
        ),
        source_name="NIH National Heart, Lung, and Blood Institute",
        source_url="https://www.nhlbi.nih.gov/health/chest-x-ray",
        tags=("opacity", "consolidation", "appearance", "look like", "white", "shadow"),
    ),
    KBEntry(
        key="terminology",
        title="Terms used in this interface",
        body=(
            "Sensitivity is the share of true pneumonia cases a model flags. Specificity is "
            "the share of normal radiographs it correctly leaves alone. AUROC summarises "
            "how well a model ranks cases across all thresholds. A threshold is the "
            "probability above which the output is called pneumonia; this project selects "
            "it on validation data rather than fixing it at 0.5."
        ),
        source_name="Project documentation",
        source_url="https://github.com/Nandanhs006/Rad-Intel",
        tags=("sensitivity", "specificity", "auroc", "threshold", "terminology",
              "precision", "recall", "what does", "mean"),
    ),
    KBEntry(
        key="gradcam",
        title="What the heatmap shows",
        body=(
            "Grad-CAM produces a heatmap indicating which regions most influenced the "
            "model's score. It shows model sensitivity, not pathology: a highlighted region "
            "is where the network responded, which is not the same as a radiological "
            "finding. In this project the displayed map is restricted to segmented lung "
            "fields, and the fraction of response falling outside them is reported rather "
            "than hidden."
        ),
        source_name="Project documentation",
        source_url="https://github.com/Nandanhs006/Rad-Intel",
        tags=("grad-cam", "gradcam", "heatmap", "saliency", "explanation", "red", "colour",
              "highlight", "overlay"),
    ),
    KBEntry(
        key="lime",
        title="What LIME adds",
        body=(
            "LIME splits the image into superpixels and tests how the score changes as "
            "regions are hidden, approximating the model locally. It rests on different "
            "assumptions from Grad-CAM, so the two are complementary rather than "
            "interchangeable. In this project they agree only weakly, which is reported as "
            "a limitation rather than smoothed over."
        ),
        source_name="Project documentation",
        source_url="https://github.com/Nandanhs006/Rad-Intel",
        tags=("lime", "superpixel", "perturbation", "second explanation", "agree"),
    ),
    KBEntry(
        key="project_scope",
        title="What this project is and is not",
        body=(
            "Rad-Intel is a research and education system. It classifies paediatric chest "
            "radiographs as normal or pneumonia, shows which regions drove the score, and "
            "writes a structured summary from those recorded outputs. It does not provide "
            "diagnosis, treatment, triage or autonomous decision support, it has not been "
            "validated on an external cohort, and it is not a medical device."
        ),
        source_name="Project documentation",
        source_url="https://github.com/Nandanhs006/Rad-Intel",
        tags=("project", "rad-intel", "scope", "purpose", "what does this do", "limitations",
              "safe", "trust", "clinical"),
    ),
    KBEntry(
        key="project_models",
        title="The models available here",
        body=(
            "Five configurations were trained under one protocol and evaluated on a "
            "held-out 624-image test partition: DenseNet121, Swin-T and ResNet50 as "
            "single-branch baselines, a DenseNet121-Swin-T fusion without CBAM, and the "
            "complete fused classifier with CBAM. The fusion without CBAM reached the "
            "highest measured accuracy at 93.4 percent with 0.977 AUROC. Removing CBAM "
            "changed nothing measurable, which the project reports rather than hides."
        ),
        source_name="Project documentation",
        source_url="https://github.com/Nandanhs006/Rad-Intel",
        tags=("model", "densenet", "swin", "resnet", "cbam", "accuracy", "which model",
              "architecture", "fusion", "hybrid"),
    ),
    KBEntry(
        key="project_limits",
        title="Known limitations of the results",
        body=(
            "The dataset is paediatric and from a single institution, the labels collapse "
            "bacterial and viral pneumonia into one class, and the public release carries "
            "no dependable patient identifier so the split cannot guarantee patient-level "
            "independence. Each configuration was trained once, so no result separates an "
            "architectural effect from seed variation. Between 21 and 48 percent of model "
            "response falls outside the segmented lung fields, consistent with shortcut "
            "learning."
        ),
        source_name="Project documentation",
        source_url="https://github.com/Nandanhs006/Rad-Intel",
        tags=("limitations", "weakness", "bias", "shortcut", "dataset", "generalise",
              "accurate", "reliable", "wrong"),
    ),
    KBEntry(
        key="when_to_seek_care",
        title="When care is sought urgently",
        body=(
            "Health services describe difficulty breathing, chest pain, a bluish tint to "
            "lips or face, confusion or difficulty staying awake, and symptoms that worsen "
            "rather than improve as reasons to seek urgent medical attention. This is "
            "general information about recognised warning signs, not an assessment of any "
            "individual."
        ),
        source_name="US Centers for Disease Control and Prevention",
        source_url="https://www.cdc.gov/pneumonia/about/index.html",
        tags=("urgent", "emergency", "when to", "seek", "worse", "warning", "danger"),
    ),
    KBEntry(
        key="diagnosis_process",
        title="How pneumonia is generally assessed in practice",
        body=(
            "Clinicians generally combine history and examination with tests that may "
            "include a chest radiograph, blood tests and sometimes sputum or other "
            "microbiological samples. Imaging is one input among several; the cause and the "
            "management plan are clinical judgements made by a qualified professional."
        ),
        source_name="US Centers for Disease Control and Prevention",
        source_url="https://www.cdc.gov/pneumonia/about/index.html",
        tags=("diagnosis", "diagnosed", "tests", "how do doctors", "confirm", "process"),
    ),
)

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "of", "to", "in",
    "on", "for", "and", "or", "it", "its", "this", "that", "with", "what", "how", "why",
    "can", "you", "i", "my", "me", "be", "if", "as", "at", "by", "from", "about", "there",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#-]+", text.lower()) if t not in _STOP and len(t) > 1]


def retrieve(question: str, k: int = 3) -> list[KBEntry]:
    """
    Return the entries most relevant to the question, best first.

    Deliberately a transparent keyword score rather than an embedding model: the
    knowledge base is small, and a scheme a reader can audit by eye is worth more
    here than a marginal gain in recall.
    """
    q = question.lower()
    q_tokens = set(_tokens(q))
    if not q_tokens:
        return []

    scored: list[tuple[float, int, KBEntry]] = []
    for idx, entry in enumerate(KB):
        score = 0.0
        for tag in entry.tags:
            if tag in q:                       # whole phrase present
                score += 3.0
            elif set(_tokens(tag)) & q_tokens:  # partial word overlap
                score += 1.0
        body_tokens = set(_tokens(entry.title + " " + entry.body))
        score += 0.4 * len(q_tokens & body_tokens)
        if score > 0:
            scored.append((score, idx, entry))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [e for _, _, e in scored[:k]]


def entry_by_key(key: str) -> KBEntry | None:
    for entry in KB:
        if entry.key == key:
            return entry
    return None
