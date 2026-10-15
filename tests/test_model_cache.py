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


def test_one_at_a_time_unless_told_otherwise(monkeypatch):
    monkeypatch.delenv("RAD_INTEL_MODEL_CACHE", raising=False)
    assert _cache_limit() == 1
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "8")
    assert _cache_limit() == 8
    # Nonsense does not take the service down, and nor does zero.
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "lots")
    assert _cache_limit() == 1
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


def test_one_at_a_time_still_serves_everything(monkeypatch):
    """
    The default. Every model stays usable; only one non-default is resident.

    This is the whole bargain: the chooser keeps offering everything, and
    choosing something costs a few hundred milliseconds rather than a
    permanent slice of the machine.
    """
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "1")
    manager = _Stub()
    for name in ("hybrid", "swin_t", "resnet50"):
        assert manager.get_model(name) is not None
    # The default plus whichever was asked for last.
    assert "hybrid" in manager._models
    assert len(manager._models) <= 2
    # And a model dropped earlier loads again when it is asked for.
    manager.get_model("swin_t")
    assert manager.loaded.count("swin_t") == 2
    assert manager.get_model("swin_t") is manager._models["swin_t"]


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


# --- switching models is rationed, running one is not -----------------------

def _fake_request(address="203.0.113.7"):
    from starlette.datastructures import Headers

    class _Client:
        host = address

    class _Request:
        client = _Client()
        headers = Headers({})
        cookies: dict = {}

    return _Request()


def test_running_the_same_model_over_and_over_is_never_rationed(monkeypatch):
    """
    The limit is on loading weights, not on asking a question.

    Everybody uses the default model, and it is never unloaded, so the route
    must not even reach the limiter for it. If this counted analyses it would
    be a second, stricter limit on the thing the site is for.
    """
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "1")
    manager = _Stub()
    manager.get_model("hybrid")

    # The guard the route uses, which is what decides whether anything is
    # counted at all.
    assert manager.is_loaded("hybrid") is True
    assert manager.is_loaded(None) is True, "the default, named or not"

    manager.get_model("swin_t")
    assert manager.is_loaded("hybrid") is True, "and it stays loaded"
    assert manager.is_loaded("densenet121") is False, "this one would cost a load"


def test_switching_again_and_again_is_stopped_and_says_why():
    from fastapi import HTTPException

    from rad_intel.api import ratelimit

    ratelimit.reset()
    request = _fake_request()
    for _ in range(ratelimit.MODEL_LOADS_PER_PERSON):
        ratelimit.check_model_load(request, None, "swin_t")

    with pytest.raises(HTTPException) as exc:
        ratelimit.check_model_load(request, None, "densenet121")
    assert exc.value.status_code == 429
    detail = exc.value.detail
    assert "changed model" in detail
    # It says what to do instead, not just that the door is shut.
    assert "Carry on with the model you have" in detail
    assert "Retry-After" in exc.value.headers
    ratelimit.reset()


def test_one_persons_switching_does_not_use_up_anothers_budget():
    from fastapi import HTTPException

    from rad_intel.api import ratelimit

    ratelimit.reset()
    mine, theirs = _fake_request("203.0.113.7"), _fake_request("198.51.100.4")
    for _ in range(ratelimit.MODEL_LOADS_PER_PERSON):
        ratelimit.check_model_load(mine, None, "swin_t")

    with pytest.raises(HTTPException):
        ratelimit.check_model_load(mine, None, "swin_t")
    # Somebody else arriving is unaffected.
    ratelimit.check_model_load(theirs, None, "swin_t")
    ratelimit.reset()


def test_the_server_as_a_whole_is_protected_from_everybody_at_once():
    """
    Several people switching costs the same whether or not any of them means
    it, so the per-person limit alone is not enough.
    """
    from fastapi import HTTPException

    from rad_intel.api import ratelimit

    ratelimit.reset()
    for n in range(ratelimit.MODEL_LOADS_OVERALL):
        ratelimit.check_model_load(_fake_request(f"198.51.100.{n % 250}"), None, "swin_t")

    with pytest.raises(HTTPException) as exc:
        ratelimit.check_model_load(_fake_request("203.0.113.99"), None, "swin_t")
    assert exc.value.status_code == 429
    assert "busy loading models" in exc.value.detail
    ratelimit.reset()


def test_a_signed_in_person_is_counted_as_themselves_not_as_an_address():
    """Two clinicians behind one hospital address are two people."""
    from fastapi import HTTPException

    from rad_intel.api import ratelimit

    ratelimit.reset()
    request = _fake_request("203.0.113.7")
    for _ in range(ratelimit.MODEL_LOADS_PER_PERSON):
        ratelimit.check_model_load(request, {"id": "doctor-a"}, "swin_t")

    with pytest.raises(HTTPException):
        ratelimit.check_model_load(request, {"id": "doctor-a"}, "swin_t")
    ratelimit.check_model_load(request, {"id": "doctor-b"}, "swin_t")
    ratelimit.reset()


def test_the_limit_is_checked_before_the_model_is_loaded(monkeypatch):
    """
    The ordering bug this had, which made the whole limit do nothing.

    require_trained answers "is it trained?" by calling is_trained, which
    loads the model to find out. The guard sat after that, so by the time it
    asked "would this load something?" the answer was always no -- it had
    just been loaded. Fifteen switches never counted as one.
    """
    from fastapi import HTTPException

    from rad_intel.api import ratelimit
    from rad_intel.api.dependencies import model_manager

    class _Request:
        class client:
            host = "203.0.113.11"

        headers: dict = {}
        cookies: dict = {}

    ratelimit.reset()
    monkeypatch.setenv("RAD_INTEL_MODEL_CACHE", "1")
    model_manager._models.clear()
    try:
        blocked_at = None
        for n in range(1, 25):
            name = "swin_t" if n % 2 else "densenet121"
            try:
                model_manager.require_trained(name, _Request())
            except HTTPException as exc:
                assert exc.status_code == 429
                blocked_at = n
                break
        assert blocked_at == ratelimit.MODEL_LOADS_PER_PERSON + 1

        # Being stopped from switching never stops somebody using the site.
        # The default is exempt by name, so this holds even here, where the
        # cache was emptied and it is not resident either.
        assert "hybrid" not in model_manager._models
        assert model_manager.require_trained("hybrid", _Request()) == "hybrid"
    finally:
        ratelimit.reset()
        model_manager._models.clear()
