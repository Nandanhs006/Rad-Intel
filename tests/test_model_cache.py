"""
How many models may sit in memory at once.

A loaded model was never dropped. Each is between 30 MB and 600 MB of
weights, so anybody could click down the model list on the Analyse tab and
load every architecture, and every released version ever published, one after
another -- on a 2 GB host, by doing nothing wrong.

Measured on this machine: the five built-in models come to about 1.2 GB, and
six released community versions on top of them take it to 1.5 GB. With the
cache bounded the same six sit at 1.23 GB, and that difference grows with
every version released.
"""

import pytest

from rad_intel.api.dependencies import ModelManager, _cache_limit


class _Stub(ModelManager):
    """A manager that counts loads instead of building networks."""

    def __init__(self):
        super().__init__()
        self.loaded: list[str] = []

    def get_model(self, model_name=None):
        import torch

        target = self._clean_model_name(model_name)
        if target not in self._models:
            self.loaded.append(target)
            self._models[target] = torch.nn.Linear(1, 1)
            self._thresholds[target] = 0.5
            self._trained[target] = True
            self._evict_down_to_limit(keep=target)
        self._models.move_to_end(target)
        return self._models[target]


def test_the_limit_is_three_unless_told_otherwise(monkeypatch):
    monkeypatch.delenv("RAD_INTEL_MODEL_CACHE", raising=False)
    assert _cache_limit() == 3
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "8")
    assert _cache_limit() == 8
    # Nonsense does not take the service down, and nor does zero.
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "lots")
    assert _cache_limit() == 3
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "0")
    assert _cache_limit() == 1


def test_loading_many_models_does_not_grow_without_end(monkeypatch):
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "3")
    manager = _Stub()

    for name in ("hybrid", "densenet121", "swin_t", "resnet50", "hybrid_no_cbam",
                 "community_hybrid_v1", "community_hybrid_v2", "community_hybrid_v3"):
        manager.get_model(name)

    assert len(manager._models) == 3
    # And the bookkeeping goes with them, so nothing keeps a threshold for a
    # model that is no longer loaded.
    assert set(manager._thresholds) == set(manager._models)
    assert set(manager._trained) == set(manager._models)


def test_the_default_model_is_never_the_one_dropped(monkeypatch):
    """
    Almost every request wants it, and reloading it costs a second and a half.

    Dropping it to make room for a version one person is trying out would
    make that person's curiosity everybody else's latency.
    """
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "2")
    manager = _Stub()
    manager.get_model("hybrid")

    for name in ("densenet121", "swin_t", "resnet50", "community_hybrid_v1"):
        manager.get_model(name)

    assert "hybrid" in manager._models
    assert manager.loaded.count("hybrid") == 1, "it was never reloaded"


def test_the_least_recently_used_is_the_one_that_goes(monkeypatch):
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "3")
    manager = _Stub()
    manager.get_model("hybrid")
    manager.get_model("densenet121")
    manager.get_model("swin_t")
    # Asking for it again puts it back at the front of the queue.
    manager.get_model("densenet121")
    manager.get_model("resnet50")

    assert "swin_t" not in manager._models, "the one nobody had asked for in longest"
    assert {"hybrid", "densenet121", "resnet50"} == set(manager._models)


def test_a_cache_of_one_still_serves_everything(monkeypatch):
    """The smallest setting a tiny host might want, and it must still work."""
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "1")
    manager = _Stub()
    for name in ("hybrid", "swin_t", "resnet50"):
        assert manager.get_model(name) is not None
    # The default plus whichever was asked for last.
    assert "hybrid" in manager._models
    assert len(manager._models) <= 2


# --- listing the models must not load them ----------------------------------

async def test_listing_the_models_does_not_load_any_of_them():
    """
    The interface fetches this on every page load, to fill the chooser.

    It used to count parameters with sum(p.numel() for p in
    get_model(name).parameters()), so every visitor arriving loaded all five
    architectures and every released version — 1.2 GB and several seconds,
    for a number that is written in the checkpoint's own shapes.
    """
    from rad_intel.api.dependencies import model_manager
    from rad_intel.api.routes.health import list_models

    model_manager._models.clear()
    listing = await list_models(model_manager)

    assert model_manager._models == {}, "not one model was loaded to list them"
    counts = {m.name: m.parameters_count for m in listing.available_models}
    assert counts["hybrid"] and counts["hybrid"] > 30_000_000
    # Read from the file, so it matches the architecture rather than a guess.
    assert counts["densenet121"] and counts["densenet121"] < counts["hybrid"]


async def test_a_model_with_no_weights_here_is_not_offered(monkeypatch):
    """
    A deployment uploads one checkpoint, not five.

    The others are names with nothing behind them, and offering one means a
    visitor picks it and gets an error rather than a result.
    """
    from rad_intel.api.dependencies import model_manager
    from rad_intel.api.routes.health import list_models

    monkeypatch.setattr(model_manager, "has_checkpoint",
                        lambda name: name == "hybrid")
    listing = await list_models(model_manager)
    assert [m.name for m in listing.available_models] == ["hybrid"]


async def test_the_default_is_offered_even_with_nothing_behind_it(monkeypatch):
    """Otherwise the chooser is empty and says nothing about why."""
    from rad_intel.api.dependencies import model_manager
    from rad_intel.api.routes.health import list_models

    monkeypatch.setattr(model_manager, "has_checkpoint", lambda name: False)
    listing = await list_models(model_manager)
    assert [m.name for m in listing.available_models] == [model_manager.active_model_name]
