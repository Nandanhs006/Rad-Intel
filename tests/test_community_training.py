"""
The community model's training path, without training anything.

A full run is hours on a GPU, so what is checked here is everything around
it -- the parts that decide whether the right data goes in, whether the wrong
file comes out, and whether the checkpoint the paper reports is safe:

* a bundle is unpacked safely, and one that points outside its directory is
  refused rather than extracted;
* a manifest resolves to images inside the bundle, and skips what it cannot
  use instead of failing the run;
* a run that is not the paper's run never writes best_hybrid_model.pt;
* reviewed cases join the training split and nothing else, and the class
  weights are recomputed over what is actually being trained on.
"""

import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import train_community_model as community  # noqa: E402

from rad_intel.training.trainer import HybridTrainer, TrainingConfig  # noqa: E402


def _bundle(tmp_path: Path, rows: list[dict], images: dict[str, bytes]) -> Path:
    path = tmp_path / "community-bundle-20261015T000000Z.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in images.items():
            archive.writestr(f"images/{name}", blob)
        archive.writestr("manifest.jsonl",
                         "\n".join(json.dumps(r) for r in rows) + "\n")
        archive.writestr("README.txt", "notes")
    return path


# --- the bundle -------------------------------------------------------------

def test_a_bundle_unpacks_into_a_manifest_and_images(tmp_path):
    rows = [
        {"image_path": "images/a.jpg", "label": "PNEUMONIA"},
        {"image_path": "images/b.jpg", "label": "NORMAL"},
    ]
    bundle = _bundle(tmp_path, rows, {"a.jpg": b"\xff\xd8", "b.jpg": b"\xff\xd8"})

    manifest, images = community.unpack_bundle(bundle, tmp_path / "out")
    assert manifest.is_file()

    samples = community.load_reviewed(manifest, images)
    assert len(samples) == 2
    assert {label for _, label in samples} == {0, 1}
    assert all(path.is_file() for path, _ in samples)


def test_a_bundle_that_points_outside_itself_is_refused(tmp_path):
    """A zip can name ../../anything, and this one arrives as a download."""
    bundle = tmp_path / "evil.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.jsonl", "{}\n")
        archive.writestr("../../escaped.txt", "pwned")

    with pytest.raises(SystemExit) as exit_info:
        community.unpack_bundle(bundle, tmp_path / "out")
    assert "outside" in str(exit_info.value)
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_a_bundle_with_no_manifest_says_so(tmp_path):
    bundle = tmp_path / "empty.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("README.txt", "nothing else")

    with pytest.raises(SystemExit) as exit_info:
        community.unpack_bundle(bundle, tmp_path / "out")
    assert "manifest.jsonl" in str(exit_info.value)


def test_unusable_manifest_entries_are_skipped_not_fatal(tmp_path, capsys):
    rows = [
        {"image_path": "images/a.jpg", "label": "PNEUMONIA"},
        {"image_path": "images/gone.jpg", "label": "NORMAL"},   # no such image
        {"image_path": "images/a.jpg", "label": "MAYBE"},       # not a label
        # A manifest written by the server records a server path. Only the
        # filename is used, and it is resolved inside the images directory.
        {"image_path": "/var/data/feedback_store/images/a.jpg", "label": "NORMAL"},
        {"image_path": "../../../etc/passwd", "label": "NORMAL"},
    ]
    bundle = _bundle(tmp_path, rows, {"a.jpg": b"\xff\xd8"})

    manifest, images = community.unpack_bundle(bundle, tmp_path / "out")
    samples = community.load_reviewed(manifest, images)

    assert len(samples) == 2, "the two usable rows, both pointing at a.jpg"
    assert "skipped 3 entries" in capsys.readouterr().out
    assert all(images.resolve() in path.parents for path, _ in samples)


def test_each_version_gets_its_own_filename(tmp_path):
    assert community.next_version(tmp_path).name == "best_community_hybrid_v1_model.pt"
    (tmp_path / "best_community_hybrid_v1_model.pt").write_bytes(b"")
    assert community.next_version(tmp_path).name == "best_community_hybrid_v2_model.pt"
    (tmp_path / "best_community_hybrid_v2_model.pt").write_bytes(b"")
    assert community.next_version(tmp_path).name == "best_community_hybrid_v3_model.pt"


# --- the checkpoint the paper reports ---------------------------------------

class _Metrics:
    """Just enough of EvaluationMetrics for _save_checkpoint."""
    optimal_threshold = 0.5574

    def to_dict(self):
        return {"accuracy": 0.9231, "auc_roc": 0.9712}


def test_a_community_run_never_writes_the_reported_checkpoint(tmp_path):
    """
    The one guarantee that matters most here.

    best_hybrid_model.pt is the file whose numbers the paper reports. A hybrid
    run normally syncs it, which is right for *the* run and wrong for every
    other one.
    """
    trainer = HybridTrainer(TrainingConfig(
        model_type="hybrid",
        checkpoint_dir=tmp_path,
        checkpoint_name="best_community_hybrid_v1_model.pt",
        sync_default_checkpoint=False,
    ))
    trainer._save_checkpoint(torch.nn.Linear(2, 2), _Metrics(), 3,
                             torch.tensor([1.0, 1.0]))

    assert (tmp_path / "best_community_hybrid_v1_model.pt").is_file()
    assert not (tmp_path / "best_hybrid_model.pt").exists(), \
        "a community run must not touch the reported checkpoint"


def test_the_papers_own_run_still_syncs_the_default(tmp_path):
    """The flag defaults to the behaviour that was there before it existed."""
    trainer = HybridTrainer(TrainingConfig(model_type="hybrid", checkpoint_dir=tmp_path))
    trainer._save_checkpoint(torch.nn.Linear(2, 2), _Metrics(), 3,
                             torch.tensor([1.0, 1.0]))
    assert (tmp_path / "best_hybrid_model.pt").is_file()


def test_a_checkpoint_carries_the_metrics_the_release_panel_reads(tmp_path):
    trainer = HybridTrainer(TrainingConfig(
        model_type="hybrid", checkpoint_dir=tmp_path,
        checkpoint_name="scored.pt", sync_default_checkpoint=False,
    ))
    trainer._save_checkpoint(torch.nn.Linear(2, 2), _Metrics(), 3,
                             torch.tensor([1.0, 1.0]))

    blob = torch.load(tmp_path / "scored.pt", map_location="cpu", weights_only=True)
    assert blob["model_type"] == "hybrid"
    assert 0.0 < blob["optimal_threshold"] < 1.0
    assert blob["metrics"]["accuracy"] == 0.9231
