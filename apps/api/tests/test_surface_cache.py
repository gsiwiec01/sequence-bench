from api.routes.experiments import _surface_cache_key, _surface_params, SURFACE_METHOD_VERSION

def test_params_contain_method_version() -> None:
    p = _surface_params(25, 0.15)
    assert p == {"resolution": 25, "margin": 0.15, "method_version": SURFACE_METHOD_VERSION}

def test_same_inputs_give_same_key() -> None:
    p = _surface_params(25, 0.15)
    k1 = _surface_cache_key("exp-1", p)
    k2 = _surface_cache_key("exp-1", _surface_params(25, 0.15))
    assert k1 == k2

def test_key_sensitive_to_experiment_and_params() -> None:
    base = _surface_cache_key("exp-1", _surface_params(25, 0.15))
    assert base != _surface_cache_key("exp-2", _surface_params(25, 0.15))
    assert base != _surface_cache_key("exp-1", _surface_params(50, 0.15))
    assert base != _surface_cache_key("exp-1", _surface_params(25, 0.2))

def test_key_is_hex_sha256() -> None:
    k = _surface_cache_key("exp-1", _surface_params(25, 0.15))
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)
