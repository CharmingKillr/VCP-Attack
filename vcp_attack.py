"""Backward-compatible entrypoint for running VCP-Attack."""

from vcpattack.attack import attack_pipeline


if __name__ == "__main__":
    attack_pipeline()
