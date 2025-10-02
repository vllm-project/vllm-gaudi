# install_prerequisites.py
import os
import subprocess
import sys
import argparse
import glob

# --- Configuration ---
WHEELS_CACHE_HOME = os.environ.get("WHEELS_CACHE_HOME", "/workspace/wheels_cache")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UCX_DIR = os.path.join(ROOT_DIR, 'ucx_source')
NIXL_DIR = os.path.join(ROOT_DIR, 'nixl_source')
UCX_INSTALL_DIR = os.path.join(ROOT_DIR, 'ucx_install')
UCX_REPO_URL = 'https://github.com/openucx/ucx.git'
NIXL_REPO_URL = 'https://github.com/ai-dynamo/nixl.git'


# --- Helper Functions ---
def run_command(command, cwd='.', env=None):
    """Helper function to run a shell command and check for errors."""
    print(f"--> Running command: {' '.join(command)} in '{cwd}'")
    subprocess.check_call(command, cwd=cwd, env=env)


def is_pip_package_installed(package_name):
    """Checks if a package is installed via pip without raising an exception."""
    result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    return result.returncode == 0


def find_nixl_wheel_in_cache(cache_dir):
    """Finds a nixl wheel file in the specified cache directory."""
    search_pattern = os.path.join(cache_dir, "nixl-*.whl")
    wheels = glob.glob(search_pattern)
    if wheels:
        return wheels[0]
    return None


def install_system_dependencies():
    """Installs required system packages using apt-get if run as root."""
    if os.geteuid() != 0:
        print("\n---")
        print("WARNING: Not running as root. Skipping system dependency installation.")
        print("Please ensure the following packages are installed on your system:")
        print("  build-essential git cmake ninja-build autotools-dev automake meson libtool libtool-bin")
        print("---\n")
        return

    print("--- Running as root. Installing system dependencies... ---")
    apt_packages = [
        "build-essential", "git", "cmake", "ninja-build", "autotools-dev", "automake", "meson", "libtool", "libtool-bin"
    ]
    run_command(['apt-get', 'update'])
    run_command(['apt-get', 'install', '-y'] + apt_packages)
    print("--- System dependencies installed successfully. ---\n")


def build_and_install_prerequisites(args):
    """Builds UCX and NIXL from source, with checks to skip if already installed or cached."""

    # --- REORDERED LOGIC AS REQUESTED ---

    # 1. First, check if nixl is already installed in the environment.
    if not args.force_reinstall and is_pip_package_installed('nixl'):
        print("--> NIXL is already installed. Nothing to do.")
        return

    # 2. Second, check for a cached wheel to skip all source builds.
    cached_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not args.force_reinstall and cached_wheel:
        print(f"\n--> Found cached wheel: {os.path.basename(cached_wheel)}.")
        print("--> ⚠️ WARNING: Installing from wheel. This assumes required UCX libraries are already on your system.")
        install_command = [sys.executable, '-m', 'pip', 'install', cached_wheel]
        run_command(install_command)
        print("\n--- Installation from cache complete. ---")
        return

    # 3. If neither of the above are true, proceed with the full build.
    print("\n--> No installed package or cached wheel found. Starting full build process...")
    install_system_dependencies()

    ucx_install_path = os.path.abspath(UCX_INSTALL_DIR)
    print(f"--> Using wheel cache directory: {WHEELS_CACHE_HOME}")
    os.makedirs(WHEELS_CACHE_HOME, exist_ok=True)

    # -- Step 1: Build and Install UCX from source --
    print("\n[1/2] Configuring and building UCX from source...")
    if not os.path.exists(UCX_DIR):
        run_command(['git', 'clone', UCX_REPO_URL, UCX_DIR])

    ucx_source_path = os.path.abspath(UCX_DIR)
    run_command(['git', 'checkout', 'v1.19.x'], cwd=ucx_source_path)
    run_command(['./autogen.sh'], cwd=ucx_source_path)

    configure_command = [
        './configure',
        f'--prefix={ucx_install_path}',
        '--enable-shared',
        '--disable-static',
        '--disable-doxygen-doc',
        '--enable-optimizations',
        '--enable-cma',
        '--enable-devel-headers',
        '--with-verbs',
        '--enable-mt',
    ]
    run_command(configure_command, cwd=ucx_source_path)
    run_command(['make', '-j', str(os.cpu_count() or 1)], cwd=ucx_source_path)
    run_command(['make', 'install'], cwd=ucx_source_path)
    print("--- UCX build and install complete ---")

    # -- Step 2: Build and Install NIXL from source --
    print("\n[2/2] Building NIXL from source...")
    if not os.path.exists(NIXL_DIR):
        run_command(['git', 'clone', NIXL_REPO_URL, NIXL_DIR])

    build_env = os.environ.copy()
    pkg_config_path = os.path.join(ucx_install_path, 'lib', 'pkgconfig')
    build_env['PKG_CONFIG_PATH'] = pkg_config_path

    ucx_lib_path = os.path.join(ucx_install_path, 'lib')
    existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    build_env['LD_LIBRARY_PATH'] = f"{ucx_lib_path}:{existing_ld_path}".strip(':')

    print(f"--> Using PKG_CONFIG_PATH: {build_env['PKG_CONFIG_PATH']}")
    print(f"--> Using LD_LIBRARY_PATH: {build_env['LD_LIBRARY_PATH']}")

    wheel_command = [sys.executable, '-m', 'pip', 'wheel', '.', '--no-deps', f'--wheel-dir={WHEELS_CACHE_HOME}']
    run_command(wheel_command, cwd=os.path.abspath(NIXL_DIR), env=build_env)

    newly_built_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not newly_built_wheel:
        raise RuntimeError("Failed to find the NIXL wheel after building it.")

    print(f"--> Successfully built wheel: {os.path.basename(newly_built_wheel)}. Now installing...")
    install_command = [sys.executable, '-m', 'pip', 'install', newly_built_wheel]
    if args.force_reinstall:
        install_command.insert(-1, '--force-reinstall')

    run_command(install_command)
    print("--- NIXL installation complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and install UCX and NIXL dependencies.")
    parser.add_argument('--force-reinstall',
                        action='store_true',
                        help='Force rebuild and reinstall of UCX and NIXL even if they are already installed.')
    args = parser.parse_args()
    build_and_install_prerequisites(args)
