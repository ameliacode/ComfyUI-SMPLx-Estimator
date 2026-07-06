from pathlib import Path
import subprocess
import sys

# Third-party estimator source cloned into vendor/ (loaded by path; not pip-installable
# without dependency conflicts). Weights + registration-walled assets are NOT fetched
# here — see the README's "Models & weights" table.
_VENDOR = {
    "WiLoR": "https://github.com/rolpotamias/WiLoR.git",
    "multi-hmr": "https://github.com/naver/multi-hmr.git",
}


def _install_with_pip() -> None:
    requirements = Path(__file__).with_name("requirements.txt")
    if not requirements.exists():
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def _clone_vendor() -> None:
    vendor = Path(__file__).with_name("vendor")
    vendor.mkdir(exist_ok=True)
    for name, url in _VENDOR.items():
        dest = vendor / name
        if dest.exists():
            print(f"[install] vendor/{name} already present, skipping")
            continue
        print(f"[install] cloning {url} -> {dest}")
        try:
            subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])
        except Exception as e:  # non-fatal: user can clone manually / set the env var
            print(f"[install] WARNING: could not clone {name}: {e}\n"
                  f"          Clone it manually into {dest} or set the "
                  f"{'WILOR_DIR' if name == 'WiLoR' else 'MULTIHMR_DIR'} env var.")


try:
    from comfy_env import install as comfy_env_install
except ImportError:
    comfy_env_install = None


if comfy_env_install is not None:
    comfy_env_install()
else:
    _install_with_pip()

_clone_vendor()
