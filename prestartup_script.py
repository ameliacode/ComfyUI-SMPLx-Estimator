try:
    from comfy_env import setup_env
except ImportError:
    setup_env = None


if setup_env is not None:
    setup_env() 