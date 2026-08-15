"""The curated top-level public API (`__init__.py`'s `_LAZY_IMPORTS` +
`__getattr__`) must actually resolve every name it advertises in `__all__`."""
import multilayer_optical_network


def test_all_public_names_resolve_to_importable_objects():
    for name in multilayer_optical_network.__all__:
        obj = getattr(multilayer_optical_network, name)
        assert obj is not None, f"{name!r} resolved to None"
