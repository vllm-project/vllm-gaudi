# install_nixl.py
import argparse
import base64
import csv
import glob
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

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

# Configure logging to output to stdout, matching the previous print behavior
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def run_command(command, cwd=".", env=None, **kwargs):
    """Helper function to run a shell command and check for errors.

    Returns:
        subprocess.CompletedProcess: The result object containing returncode, stdout, etc.
    """
    logger.info("--> Running command: %s in '%s'", " ".join(command), cwd)
    # Default to check=True to raise exception on error, matching old check_call behavior
    if "check" not in kwargs:
        kwargs["check"] = True
    return subprocess.run(command, cwd=cwd, env=env, **kwargs)


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
        logger.warning(
            "\n---\n"
            "WARNING: Not running as root. Skipping system dependency installation.\n"
            "Please ensure the following packages are installed on your system:\n"
            "  patchelf build-essential git cmake ninja-build autotools-dev automake meson libtool libtool-bin\n"
            "---\n")
        return

    logger.info("--- Running as root. Installing system dependencies... ---")
    apt_packages = [
        "patchelf",
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
    logger.info("--- System dependencies installed successfully. ---\n")


# --- Wheel Manipulation Helpers ---
# Adapted from https://github.com/ai-dynamo/nixl/blob/b20598f2906b7037f711060b937e7156bb83f27f/contrib/wheel_add_ucx_plugins.py
def extract_wheel(wheel_path):
    """
    Extract the wheel to a temporary directory.
    Returns:
        Path to the temporary directory. The caller is responsible for cleaning up the directory.
    """
    temp_dir = tempfile.mkdtemp()
    logger.info("Extracting wheel %s to %s", wheel_path, temp_dir)
    with zipfile.ZipFile(wheel_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir


def update_wheel_record_file(temp_dir):
    """
    Update the RECORD file in the wheel to include the hashes and sizes of all files.
    """
    dist_info_dir = None
    for entry in os.listdir(temp_dir):
        if entry.endswith(".dist-info"):
            dist_info_dir = entry
            break
    if dist_info_dir is None:
        raise RuntimeError("No .dist-info directory found in wheel")

    record_path = os.path.join(temp_dir, dist_info_dir, "RECORD")

    def hash_and_size(file_path):
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        digest = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode("ascii")
        size = os.path.getsize(file_path)
        return f"sha256={digest}", str(size)

    entries = []
    for root, _, files in os.walk(temp_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, temp_dir).replace(os.sep, "/")
            if rel_path == f"{dist_info_dir}/RECORD":
                # RECORD file itself: no hash or size
                entries.append((rel_path, "", ""))
            else:
                file_hash, file_size = hash_and_size(full_path)
                entries.append((rel_path, file_hash, file_size))

    with open(record_path, "w", newline="") as rec_file:
        writer = csv.writer(rec_file)
        writer.writerows(entries)


def create_wheel(wheel_path, temp_dir):
    """
    Create a wheel from a temporary directory.
    """
    logger.info("Creating wheel %s from %s", wheel_path, temp_dir)
    update_wheel_record_file(temp_dir)
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=temp_dir)
                zip_ref.write(abs_path, arcname=rel_path)


def get_repaired_lib_name_map(libs_dir):
    """
    auditwheel repair renames all libs to include a hash of the library name
    e.g. "nixl.libs/libboost_atomic-fb1368c6.so.1.66.0"
    Extract mapping from base name (like "libboost_atomic") to full file name
    (like "libboost_atomic-fb1368c6.so.1.66.0").
    """
    name_map = {}
    for fname in sorted(os.listdir(libs_dir)):
        if os.path.isfile(os.path.join(libs_dir, fname)) and ".so" in fname and "-" in fname:
            base_name = fname.split("-")[0]
            name_map[base_name] = fname
            logger.info("Found already bundled lib: %s -> %s", base_name, fname)
    return name_map


def get_lib_deps(lib_path):
    """
    Get the dependencies of a library, as a map from library name to path.
    """
    # Replaced os.popen with subprocess.check_output
    deps = run_command(["ldd", lib_path], capture_output=True, text=True).stdout.strip().split("\n")
    ret = {}
    for dep in deps:
        if "=>" in dep:
            left, right = dep.split("=>", 1)
            dep_name = left.strip()
            right = right.strip()
            if right == "not found":
                ret[dep_name] = None
            else:
                dep_path = right.split(" ")[0].strip()
                ret[dep_name] = dep_path
    return ret


def copytree(src, dst):
    """
    Copy a tree of files from @src directory to @dst directory.
    Similar to shutil.copytree, but returns a list of all files copied.
    """
    copied_files = []
    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dst_dir = os.path.join(dst, rel_path)
        os.makedirs(dst_dir, exist_ok=True)
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)
            copied_files.append(dst_file)
    return copied_files


def add_plugins(wheel_path, sys_plugins_dir, install_dirname):
    """
    Adds the plugins from @sys_dir to the wheel.
    The plugins are copied to a subdirectory @install_dir relative to the wheel's nixl.libs.
    The plugins are patched to load their dependencies from the wheel.
    The wheel file is then recreated.
    """
    temp_dir = extract_wheel(wheel_path)

    pkg_name = wheel_path.split("/")[-1].split("-")[0]
    pkg_libs_dir = os.path.join(temp_dir, f"{pkg_name}.libs")
    if not os.path.exists(pkg_libs_dir):
        raise FileNotFoundError(f"{pkg_name}.libs directory not found in wheel: {wheel_path}")

    logger.debug("Listing existing libs:")
    name_map = get_repaired_lib_name_map(pkg_libs_dir)

    # Ensure that all of them in name_map have RPATH set to $ORIGIN
    for fname in name_map.values():
        fpath = os.path.join(pkg_libs_dir, fname)
        res = run_command(["patchelf", "--print-rpath", fpath], capture_output=True, text=True, check=False)
        rpath = res.stdout.strip() if res.returncode == 0 else ""
        if "$ORIGIN" in rpath.split(":"):
            continue
        rpath = "$ORIGIN" if not rpath else "$ORIGIN:" + rpath
        logger.debug("Setting rpath for %s to %s", fpath, rpath)
        run_command(["patchelf", "--set-rpath", rpath, fpath])

    pkg_plugins_dir = os.path.join(pkg_libs_dir, install_dirname)
    logger.debug("Copying plugins from %s to %s", sys_plugins_dir, pkg_plugins_dir)
    copied_files = copytree(sys_plugins_dir, pkg_plugins_dir)
    if not copied_files:
        raise RuntimeError(f"No plugins found in {sys_plugins_dir}")

    # Patch all libs to load plugin deps from the wheel
    for fname in copied_files:
        logger.debug("Patching %s", fname)
        fpath = os.path.join(pkg_plugins_dir, fname)
        if os.path.isfile(fpath) and ".so" in fname:
            res = run_command(["patchelf", "--print-rpath", fpath], capture_output=True, text=True, check=False)
            rpath = res.stdout.strip() if res.returncode == 0 else ""
            rpath = "$ORIGIN/..:$ORIGIN" if not rpath else "$ORIGIN/..:$ORIGIN:" + rpath
            logger.debug("Setting rpath for %s to %s", fpath, rpath)
            run_command(["patchelf", "--set-rpath", rpath, fpath])

            # Replace the original libs with the patched one
            for libname, _ in get_lib_deps(fpath).items():
                # "libuct.so.0" -> "libuct"
                base_name = libname.split(".")[0]
                if base_name in name_map:
                    packaged_name = name_map[base_name]
                    logger.debug("Replacing %s with %s in %s", libname, packaged_name, fpath)
                    run_command(["patchelf", "--replace-needed", libname, packaged_name, fpath])

            # Check that there is no breakage introduced in the patched lib
            logger.debug("Checking that %s loads", fpath)
            original_deps = get_lib_deps(os.path.join(sys_plugins_dir, fname))
            for libname, libpath in get_lib_deps(fpath).items():
                if libpath is None and (libname not in original_deps or original_deps[libname] is not None):
                    raise RuntimeError(f"Library {libname} not loaded by {fpath}")

    create_wheel(wheel_path, temp_dir)
    shutil.rmtree(temp_dir)
    logger.info("Added plugins to wheel: %s", wheel_path)


def build_and_install_prerequisites(args):
    """Builds UCX and NIXL from source, creating a self-contained wheel."""

    # ... (initial checks and setup are unchanged) ...
    if not args.force_reinstall and is_pip_package_installed("nixl"):
        logger.info("--> NIXL is already installed. Nothing to do.")
        return

    cached_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not args.force_reinstall and cached_wheel:
        logger.info("\n--> Found self-contained wheel: %s.", os.path.basename(cached_wheel))
        logger.info("--> Installing from cache, skipping all source builds.")
        install_command = [sys.executable, "-m", "pip", "install", cached_wheel]
        run_command(install_command)
        logger.info("\n--- Installation from cache complete. ---")
        return

    logger.info("\n--> No installed package or cached wheel found. Starting full build process...")
    logger.info("\n--> Installing auditwheel...")
    run_command([sys.executable, "-m", "pip", "install", "auditwheel"])
    install_system_dependencies()
    ucx_install_path = os.path.abspath(UCX_INSTALL_DIR)
    logger.info("--> Using wheel cache directory: %s", WHEELS_CACHE_HOME)
    os.makedirs(WHEELS_CACHE_HOME, exist_ok=True)

    # -- Step 1: Build UCX from source --
    logger.info("\n[1/4] Configuring and building UCX from source...")
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
    logger.info("--- UCX build and install complete ---")

    # -- Step 2: Build NIXL wheel from source --
    logger.info("\n[2/4] Building NIXL wheel from source...")
    if not os.path.exists(NIXL_DIR):
        run_command(["git", "clone", NIXL_REPO_URL, NIXL_DIR])
    else:
        run_command(["git", "fetch", "--tags"], cwd=NIXL_DIR)
    run_command(["git", "checkout", NIXL_VERSION], cwd=NIXL_DIR)
    logger.info("--> Checked out NIXL version: %s", NIXL_VERSION)

    build_env = os.environ.copy()
    build_env["PKG_CONFIG_PATH"] = os.path.join(ucx_install_path, "lib", "pkgconfig")
    ucx_lib_path = os.path.join(ucx_install_path, "lib")
    ucx_plugin_path = os.path.join(ucx_lib_path, "ucx")
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    build_env["LD_LIBRARY_PATH"] = f"{ucx_lib_path}:{ucx_plugin_path}:{existing_ld_path}".strip(":")
    logger.info("--> Using LD_LIBRARY_PATH: %s", build_env["LD_LIBRARY_PATH"])

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
    logger.info("\n[3/4] Repairing NIXL wheel to include UCX libraries...")
    unrepaired_wheel = find_nixl_wheel_in_cache(temp_wheel_dir)
    if not unrepaired_wheel:
        raise RuntimeError("Failed to find the NIXL wheel after building it.")

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

    # -- Step 4: Bundle UCX plugins into the repaired wheel --
    logger.info("\n[4/4] Bundling UCX plugins into the wheel...")
    repaired_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)

    ucx_plugins_src = os.path.join(ucx_install_path, "lib", "ucx")

    if os.path.exists(ucx_plugins_src):
        logger.info("--> Adding plugins from %s", ucx_plugins_src)
        # Direct call to the ported function
        add_plugins(repaired_wheel, ucx_plugins_src, "ucx")
    else:
        logger.info("--> Warning: UCX plugins not found. Skipping bundling.")

    # No more temporary files to remove, just the temp wheelhouse
    run_command(["rm", "-rf", temp_wheel_dir])

    newly_built_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not newly_built_wheel:
        raise RuntimeError("Failed to find the repaired NIXL wheel.")

    logger.info("--> Successfully built self-contained wheel: %s. Now installing...",
                os.path.basename(newly_built_wheel))
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
    logger.info("--- NIXL installation complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and install UCX and NIXL dependencies.")
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
