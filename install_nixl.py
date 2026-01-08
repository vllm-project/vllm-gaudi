# install_nixl.py
import argparse
import glob
import os
import subprocess
import sys


# --- Configuration ---
WHEELS_CACHE_HOME = os.environ.get("WHEELS_CACHE_HOME", "/tmp/wheels_cache")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UCX_DIR = os.path.join("/tmp", "ucx_source")
NIXL_DIR = os.path.join("/tmp", "nixl_source")
UCX_INSTALL_DIR = os.path.join("/tmp", "ucx_install")
UCX_REPO_URL = "https://github.com/openucx/ucx.git"
NIXL_REPO_URL = "https://github.com/ai-dynamo/nixl.git"

# Latest good commit with gaudi_gdr support
DEFAULT_UCX_COMMIT = "1df7b045d36c1e84f2fe9f251de83fb9103fc80e"
NIXL_VERSION = os.environ.get("NIXL_VERSION", "0.7.0")


# --- Helper Functions ---
def run_command(command, cwd=".", env=None):
    """Helper function to run a shell command and check for errors."""
    print(f"--> Running command: {' '.join(command)} in '{cwd}'", flush=True)
    subprocess.check_call(command, cwd=cwd, env=env)


def is_pip_package_installed(package_name):
    """Checks if a package is installed via pip without raising an exception."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def find_nixl_wheel_in_cache(cache_dir):
    """Finds a nixl wheel file in the specified cache directory."""
    # The repaired wheel will have a 'manylinux' tag, but this glob still works.
    search_pattern = os.path.join(cache_dir, f"nixl*{NIXL_VERSION}*.whl")
    wheels = glob.glob(search_pattern)
    if wheels:
        # Sort to get the most recent/highest version if multiple exist
        wheels.sort()
        return wheels[-1]
    return None


def install_system_dependencies():
    """Installs required system packages using apt-get if run as root."""
    if os.geteuid() != 0:
        print("\n---", flush=True)
        print(
            "WARNING: Not running as root. Skipping system dependency installation.",
            flush=True,
        )
        print(
            "Please ensure the following packages are installed on your system:",
            flush=True,
        )
        print(
            "  patchelf build-essential git cmake ninja-build autotools-dev automake meson libtool libtool-bin",
            flush=True,
        )
        print("---\n", flush=True)
        return

    print("--- Running as root. Installing system dependencies... ---", flush=True)
    apt_packages = [
        "patchelf",  # <-- Add patchelf here
        "build-essential",
        "git",
        "cmake",
        "ninja-build",
        "autotools-dev",
        "automake",
        "meson",
        "libtool",
        "libtool-bin",
    ]
    run_command(["apt-get", "update"])
    run_command(["apt-get", "install", "-y"] + apt_packages)
    print("--- System dependencies installed successfully. ---\n", flush=True)


def build_and_install_prerequisites(args):
    """Builds UCX and NIXL from source, creating a self-contained wheel."""

    # ... (initial checks and setup are unchanged) ...
    if not args.force_reinstall and is_pip_package_installed("nixl"):
        print("--> NIXL is already installed. Nothing to do.", flush=True)
        return

    cached_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not args.force_reinstall and cached_wheel:
        print(
            f"\n--> Found self-contained wheel: {os.path.basename(cached_wheel)}.",
            flush=True,
        )
        print("--> Installing from cache, skipping all source builds.", flush=True)
        install_command = [sys.executable, "-m", "pip", "install", cached_wheel]
        run_command(install_command)
        print("\n--- Installation from cache complete. ---", flush=True)
        return

    print(
        "\n--> No installed package or cached wheel found. Starting full build process...",
        flush=True,
    )
    print("\n--> Installing auditwheel...", flush=True)
    run_command([sys.executable, "-m", "pip", "install", "auditwheel"])
    install_system_dependencies()
    ucx_install_path = os.path.abspath(UCX_INSTALL_DIR)
    print(f"--> Using wheel cache directory: {WHEELS_CACHE_HOME}", flush=True)
    os.makedirs(WHEELS_CACHE_HOME, exist_ok=True)

    # -- Step 1: Build UCX from source --
    # ... (UCX build process is unchanged) ...
    print("\n[1/4] Configuring and building UCX from source...", flush=True)
    if not os.path.exists(UCX_DIR):
        run_command(["git", "clone", UCX_REPO_URL, UCX_DIR])
    ucx_source_path = os.path.abspath(UCX_DIR)
    run_command(["git", "checkout", args.ucx_commit], cwd=ucx_source_path)
    run_command(["./autogen.sh"], cwd=ucx_source_path)
    configure_command = [
        "./configure",
        f"--prefix={ucx_install_path}",
        "--enable-shared",
        "--disable-static",
        "--disable-doxygen-doc",
        "--enable-optimizations",
        "--enable-cma",
        "--enable-devel-headers",
        "--with-verbs",
        "--enable-mt",
        "--with-gaudi=yes",
        "--with-mlx5=no",
        "--enable-examples",
    ]
    run_command(configure_command, cwd=ucx_source_path)
    run_command(["make", "-j", str(os.cpu_count() or 1)], cwd=ucx_source_path)
    run_command(["make", "install-strip"], cwd=ucx_source_path)
    print("--- UCX build and install complete ---", flush=True)

    # -- Step 2: Build NIXL wheel from source --
    print("\n[2/4] Building NIXL wheel from source...", flush=True)
    if not os.path.exists(NIXL_DIR):
        run_command(["git", "clone", NIXL_REPO_URL, NIXL_DIR])
    else:
        run_command(["git", "fetch", "--tags"], cwd=NIXL_DIR)
    run_command(["git", "checkout", NIXL_VERSION], cwd=NIXL_DIR)
    print(f"--> Checked out NIXL version: {NIXL_VERSION}", flush=True)

    build_env = os.environ.copy()
    build_env["PKG_CONFIG_PATH"] = os.path.join(ucx_install_path, "lib", "pkgconfig")
    ucx_lib_path = os.path.join(ucx_install_path, "lib")
    ucx_plugin_path = os.path.join(ucx_lib_path, "ucx")
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    build_env["LD_LIBRARY_PATH"] = (
        f"{ucx_lib_path}:{ucx_plugin_path}:{existing_ld_path}".strip(":")
    )
    print(f"--> Using LD_LIBRARY_PATH: {build_env['LD_LIBRARY_PATH']}", flush=True)

    temp_wheel_dir = os.path.join(ROOT_DIR, "temp_wheelhouse")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            f"--wheel-dir={temp_wheel_dir}",
            "-C",
            f"setup-args=-Ducx_path={ucx_install_path}",
            "-C",
            "setup-args=-Ddisable_gds_backend=false",
        ],
        cwd=os.path.abspath(NIXL_DIR),
        env=build_env,
    )

    # -- Step 3: Repair the wheel, excluding the already-bundled plugin --
    print("\n[3/4] Repairing NIXL wheel to include UCX libraries...", flush=True)
    unrepaired_wheel = find_nixl_wheel_in_cache(temp_wheel_dir)
    if not unrepaired_wheel:
        raise RuntimeError("Failed to find the NIXL wheel after building it.")

    # --- 👇 THE CORRECTED COMMAND 👇 ---
    # We tell auditwheel to ignore the plugin that mesonpy already handled.
    auditwheel_command = [
        "auditwheel",
        "repair",
        "--exclude",
        "libplugin_UCX.so",
        "--exclude",
        "libplugin_UCX_MO.so",
        unrepaired_wheel,
        f"--wheel-dir={WHEELS_CACHE_HOME}",
    ]
    run_command(auditwheel_command, env=build_env)
    # --- 👆 END CORRECTION 👆 ---

    # -- Step 4: Bundle UCX plugins into the repaired wheel --
    print("\n[4/4] Bundling UCX plugins into the wheel...", flush=True)
    repaired_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)

    ucx_plugins_src = os.path.join(ucx_install_path, "lib", "ucx")
    helper_script = os.path.join(NIXL_DIR, "contrib", "wheel_add_ucx_plugins.py")

    if os.path.exists(helper_script) and os.path.exists(ucx_plugins_src):
        # Patch the helper script to skip NIXL plugins (since we only want UCX)
        # This prevents it from failing when it can't find system NIXL plugins
        sed_expr = 's/add_plugins(wheel_path, args.nixl_plugins_dir, "nixl")/# &/'
        run_command(["sed", "-i", sed_expr, helper_script])

        print(f"--> Adding plugins from {ucx_plugins_src}", flush=True)
        bundle_cmd = [
            sys.executable,
            helper_script,
            "--ucx-plugins-dir",
            ucx_plugins_src,
            repaired_wheel,
        ]
        run_command(bundle_cmd, env=build_env)
    else:
        print(
            f"--> Warning: Helper script or UCX plugins not found. Skipping bundling.",
            flush=True,
        )

    # No more temporary files to remove, just the temp wheelhouse
    run_command(["rm", "-rf", temp_wheel_dir])

    newly_built_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not newly_built_wheel:
        raise RuntimeError("Failed to find the repaired NIXL wheel.")

    print(
        f"--> Successfully built self-contained wheel: {os.path.basename(newly_built_wheel)}. Now installing...",
        flush=True,
    )
    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",  # w/o "no-deps", it will install cuda-torch
        newly_built_wheel,
    ]
    if args.force_reinstall:
        install_command.insert(-1, "--force-reinstall")

    run_command(install_command)
    print("--- NIXL installation complete ---", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and install UCX and NIXL dependencies."
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Force rebuild and reinstall of UCX and NIXL even if they are already installed.",
    )
    parser.add_argument(
        "--ucx-commit",
        default=DEFAULT_UCX_COMMIT,
        help=f"UCX commit to build (default: {DEFAULT_UCX_COMMIT})",
    )
    args = parser.parse_args()
    build_and_install_prerequisites(args)
