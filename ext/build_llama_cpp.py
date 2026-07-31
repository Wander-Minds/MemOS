#!/usr/bin/env python3
"""Cross-platform build script for llama.cpp (CUDA, Vulkan, Metal, SYCL backends).

Design:
- llama.cpp lives at ``src/ext/llama.cpp/``, added as a git submodule.
- **No hard-coded absolute paths.** Tools and SDKs are located exclusively
  through the ``PATH`` and standard environment variables (``CUDA_PATH``,
  ``VULKAN_SDK``, ``ProgramFiles`` on Windows).
- Each platform builds into its own default build directory
  (``build-windows`` / ``build-linux`` / ``build-macos``) so that builds
  from different platforms can coexist in the same source tree.
- CUDA and Vulkan backends are auto-detected; they can also be forced
  on/off from the command line.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & platform helpers
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
LLAMA_SRC = HERE / "llama.cpp"          # submodule checkout
SYSTEM = platform.system().lower()      # 'windows', 'linux', 'darwin'
PLATFORM_TAG = {"darwin": "macos"}.get(SYSTEM, SYSTEM)
DEFAULT_BUILD_DIR = HERE / f"build-{PLATFORM_TAG}"
EXE_SUFFIX = ".exe" if SYSTEM == "windows" else ""


def run(cmd, **kwargs):
    """Run a command and raise on failure."""
    print(f"> {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.check_call([str(c) for c in cmd], **kwargs)


def is_multiconfig_generator(gen: str | None) -> bool:
    """Return True when the CMake generator uses a per-config build step."""
    if gen is None:
        gen = detect_default_generator()
    return any(
        kw in (gen or "")
        for kw in ["Visual Studio", "Xcode", "MSBuild"]
    )


def detect_default_generator() -> str | None:
    """Ask CMake what generator it would use by default."""
    try:
        out = subprocess.check_output(
            ["cmake", "--system-information", "-N"],
            stderr=subprocess.DEVNULL, text=True, timeout=30,
        )
        for line in out.splitlines():
            if line.startswith("CMAKE_GENERATOR:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Windows: MSVC toolchain
# ---------------------------------------------------------------------------

def find_vs_vcvarsall():
    """Locate the Visual Studio vcvarsall.bat helper (Windows only)."""
    program_files = [
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramFiles"),
    ]
    editions = ["Enterprise", "Professional", "Community", "BuildTools"]
    versions = ["2022", "2019"]
    for pf in filter(None, program_files):
        for version in versions:
            for edition in editions:
                candidate = (
                    Path(pf) / "Microsoft Visual Studio" / version / edition
                    / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                )
                if candidate.exists():
                    return candidate
    return None


def import_vcvars(env, arch="x64"):
    """Run vcvarsall.bat and merge the resulting environment into *env*."""
    vcvarsall = find_vs_vcvarsall()
    if not vcvarsall:
        print("ERROR: Could not find Visual Studio vcvarsall.bat.", file=sys.stderr)
        print("Run this script from a VS x64 Native Tools Command Prompt, "
              "or install VS 2019/2022.", file=sys.stderr)
        sys.exit(1)

    print(f"Importing MSVC environment from: {vcvarsall}")
    cmd = f'"{vcvarsall}" {arch} && set'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print("ERROR: vcvarsall.bat failed.", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)

    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.upper()] = value


def ensure_msvc_env(env):
    """Make sure MSVC cl.exe/link.exe are available, importing vcvars if needed."""
    if shutil.which("cl.exe", path=env.get("PATH", "")):
        return
    import_vcvars(env)
    if not shutil.which("cl.exe", path=env.get("PATH", "")):
        print("ERROR: cl.exe still not found after running vcvarsall.bat.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Optional backend detection (no hard-coded install paths)
# ---------------------------------------------------------------------------

def find_cuda(env):
    """Locate CUDA via CUDA_PATH or nvcc on PATH; add its bin dir to PATH.

    Returns True if CUDA was found.
    """
    cuda_path = env.get("CUDA_PATH") or env.get("CUDA_HOME")
    if not cuda_path:
        nvcc = shutil.which("nvcc", path=env.get("PATH", ""))
        if nvcc:
            cuda_path = str(Path(nvcc).resolve().parent.parent)
    if not cuda_path:
        return False
    cuda_path = Path(cuda_path)
    cuda_bin = cuda_path / "bin"
    if cuda_bin.is_dir():
        env["CUDA_PATH"] = str(cuda_path)
        env["PATH"] = f"{cuda_bin}{os.pathsep}{env.get('PATH', '')}"
    print(f"Using CUDA: {cuda_path}")
    return True


def find_vulkan(env):
    """Locate the Vulkan SDK/loader without any hard-coded paths.

    Detection order:
      1. VULKAN_SDK environment variable (set by the official SDK).
      2. Windows: check for vulkan-1.lib alongside VULKAN_SDK or in well-known
         SDK installation locations.
      3. Linux: pkg-config metadata for the system Vulkan loader.
      4. glslc shader compiler (ships with the SDK) – only confirms tools,
         not dev headers; we still look for the library.

    Returns True only when both the build tools AND development headers/
    libraries are present.
    """
    # --- Helper: look for the Vulkan library ---
    def _vulkan_lib_exists(search_paths) -> bool:
        lib_name = "vulkan-1.lib" if SYSTEM == "windows" else "libvulkan.so"
        for base in search_paths:
            if base is None:
                continue
            base = Path(base)
            # Common sub-directories where the lib may live
            for lib_dir in [base, base / "Lib", base / "lib"]:
                if (lib_dir / lib_name).exists():
                    return True
                # On Windows, also check for .dll
                if SYSTEM == "windows" and (lib_dir / "vulkan-1.dll").exists():
                    return True
        return False

    # 1. VULKAN_SDK environment variable
    sdk = env.get("VULKAN_SDK")
    if sdk:
        sdk_path = Path(sdk)
        if _vulkan_lib_exists([sdk_path]):
            print(f"Using Vulkan SDK: {sdk}")
            return True
        else:
            print(f"WARNING: VULKAN_SDK is set ({sdk}) but Vulkan library not found.\n"
                  f"  Checked: {sdk_path / 'Lib'}, {sdk_path / 'lib'}", file=sys.stderr)
            # Fall through – maybe a different SDK layout

    # 2. Windows: check well-known install locations
    if SYSTEM == "windows":
        known_paths = [
            Path(os.environ.get("ProgramFiles", "")) / "VulkanSDK",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "VulkanSDK",
        ]
        for base in known_paths:
            if base.is_dir():
                # Find the newest version sub-directory
                versions = sorted(base.iterdir(), reverse=True) if base.is_dir() else []
                for ver_dir in versions:
                    if ver_dir.is_dir() and _vulkan_lib_exists([ver_dir / "Lib"]):
                        env["VULKAN_SDK"] = str(ver_dir)
                        print(f"Using Vulkan SDK (auto-detected): {ver_dir}")
                        return True

    # 3. pkg-config (Linux) – CMake's FindVulkan also requires glslc, so check it too
    pkg_config = shutil.which("pkg-config", path=env.get("PATH", ""))
    if pkg_config:
        proc = subprocess.run(
            [pkg_config, "--exists", "vulkan"],
            env=env, capture_output=True,
        )
        if proc.returncode == 0:
            # CMake's FindVulkan requires glslc on Linux; fail early if missing
            if not shutil.which("glslc", path=env.get("PATH", "")):
                print("WARNING: Vulkan library found via pkg-config but glslc is not on PATH.\n"
                      "  CMake's FindVulkan requires glslc. Install it (e.g. apt install glslang-tools).",
                      file=sys.stderr)
                return False
            # Double-check the library exists
            lib_paths = subprocess.run(
                [pkg_config, "--libs-only-L", "vulkan"],
                env=env, capture_output=True, text=True,
            ).stdout.strip()
            search = [Path(p) for p in lib_paths.split()] if lib_paths else []
            if not search or _vulkan_lib_exists(search):
                print("Using system Vulkan loader (pkg-config).")
                return True

    # 4. glslc alone is not enough – we already failed to find dev files
    if shutil.which("glslc", path=env.get("PATH", "")):
        print("NOTE: glslc found on PATH but Vulkan development headers/libs not detected.\n"
              "  Install the full Vulkan SDK from https://vulkan.lunarg.com/", file=sys.stderr)

    return False


def resolve_choice(choice, detected, name):
    """Resolve an auto/on/off backend choice against detection results."""
    if choice == "on" and not detected:
        print(f"ERROR: --{name}=on was requested but {name.upper()} was not found.",
              file=sys.stderr)
        sys.exit(1)
    enabled = detected if choice == "auto" else choice == "on"
    print(f"{name.upper()} backend: {'ON' if enabled else 'OFF'}")
    return enabled


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------

def detect_build_tool(env):
    """Return the generator and whether it is multi-config."""
    generator = None
    if shutil.which("ninja", path=env.get("PATH", "")):
        generator = "Ninja"
    elif SYSTEM == "windows":
        # Default to Visual Studio generator if Ninja is absent
        # CMake will pick whatever VS version is available.
        pass  # None = CMake default
    return generator


def configure(build_dir, build_type, env, cuda, vulkan, cuda_arch=None,
              generator=None):
    """Run CMake configure.

    For multi-config generators (MSVC, Xcode), the build type is controlled
    by ``--config`` at build time.  For single-config generators (Ninja,
    Makefiles) we pass ``-DCMAKE_BUILD_TYPE``.
    """
    multiconfig = is_multiconfig_generator(generator)
    cmake_args = [
        "cmake",
        "-S", str(LLAMA_SRC),
        "-B", str(build_dir),
    ]

    if generator:
        cmake_args += ["-G", generator]

    if multiconfig:
        # multi-config generators ignore CMAKE_BUILD_TYPE at configure time
        # but we still pass it so that CMakePresets etc. can see it.
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
    else:
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")

    cmake_args += [
        f"-DGGML_CUDA={'ON' if cuda else 'OFF'}",
        f"-DGGML_VULKAN={'ON' if vulkan else 'OFF'}",
    ]
    if cuda and cuda_arch:
        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")

    run(cmake_args, env=env)
    return multiconfig


def build(build_dir, build_type, env, multiconfig):
    """Run CMake build.

    For multi-config generators ``--config`` selects the configuration to
    build.  For single-config generators it is ignored (but harmless).
    """
    args = ["cmake", "--build", str(build_dir), "--parallel"]
    if multiconfig:
        args += ["--config", build_type]
    run(args, env=env)


def verify(build_dir, build_type):
    """Run llama-cli --list-devices to confirm the binary exists and backends are available.

    The binary location depends on the generator type:
    - Single-config (Ninja, Makefiles): ``build_dir/bin/llama-cli``
    - Multi-config (MSVC, Xcode): ``build_dir/bin/{Config}/llama-cli.exe``
    """
    candidates = [
        build_dir / "bin" / f"llama-cli{EXE_SUFFIX}",
        build_dir / "bin" / build_type / f"llama-cli{EXE_SUFFIX}",
    ]
    cli = None
    for c in candidates:
        if c.exists():
            cli = c
            break

    if cli is None:
        print(f"WARNING: llama-cli binary not found under {build_dir / 'bin'}; "
              f"skipping verification.", file=sys.stderr)
        return

    print(f"> {cli} --list-devices")
    subprocess.run([str(cli), "--list-devices"], check=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform llama.cpp build (CUDA/Vulkan/Metal auto-detected)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Build Debug configuration (default is Release)."
    )
    parser.add_argument(
        "--release", action="store_true",
        help="Build Release configuration (default)."
    )
    parser.add_argument(
        "--build-dir", type=Path, default=DEFAULT_BUILD_DIR,
        help=f"CMake build directory (default: {DEFAULT_BUILD_DIR})."
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Wipe the build directory before configuring."
    )
    parser.add_argument(
        "--configure-only", action="store_true",
        help="Stop after CMake configure."
    )
    parser.add_argument(
        "--cuda", choices=["auto", "on", "off"], default="auto",
        help="CUDA backend: auto-detect (default), force on, or force off."
    )
    parser.add_argument(
        "--vulkan", choices=["auto", "on", "off"], default="auto",
        help="Vulkan backend: auto-detect (default), force on, or force off."
    )
    parser.add_argument(
        "--cuda-arch", type=str, default="",
        help='CUDA architectures, e.g. "86;89". Auto-detected if omitted.'
    )
    parser.add_argument(
        "--generator", type=str, default=None,
        help="CMake generator (e.g. 'Ninja', 'Visual Studio 17 2022'). "
             "Auto-detected if omitted."
    )
    args = parser.parse_args()

    build_type = "Debug" if args.debug and not args.release else "Release"
    build_dir = args.build_dir.resolve()

    if args.rebuild and build_dir.exists():
        print(f"Removing {build_dir}")
        shutil.rmtree(build_dir)

    print(f"Platform: {SYSTEM} | Build type: {build_type} | Build dir: {build_dir}")
    print(f"llama.cpp source: {LLAMA_SRC}")

    # Work on a copy of the environment so we do not mutate the caller's shell.
    env = os.environ.copy()

    # CUDA on Windows needs the MSVC host toolchain for nvcc.
    if SYSTEM == "windows":
        ensure_msvc_env(env)

    cuda_found = find_cuda(env)
    vulkan_found = find_vulkan(env)
    cuda = resolve_choice(args.cuda, cuda_found, "cuda")
    vulkan = resolve_choice(args.vulkan, vulkan_found, "vulkan")

    generator = args.generator or detect_build_tool(env)
    multiconfig = configure(build_dir, build_type, env, cuda, vulkan,
                             cuda_arch=args.cuda_arch or None,
                             generator=generator)

    if args.configure_only:
        print("Configuration complete. Skipping build.")
        return

    build(build_dir, build_type, env, multiconfig)
    verify(build_dir, build_type)


if __name__ == "__main__":
    main()
