from pathlib import Path
import subprocess
import sys


def _install_with_pip() -> None:
    requirements = Path(__file__).with_name("requirements.txt")
    if not requirements.exists():
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


try:
    from comfy_env import install as comfy_env_install
except ImportError:
    comfy_env_install = None


if comfy_env_install is not None:
    comfy_env_install()
else:
    _install_with_pip()
