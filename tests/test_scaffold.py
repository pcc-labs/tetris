import tetris_agent


def test_package_importable_with_version():
    assert tetris_agent.__version__ == "0.1.0"
