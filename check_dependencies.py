#!/usr/bin/env python3
"""
Dependency checker for Magnet Array Optimization.
Checks if required packages are installed and offers to install missing ones.
"""

import sys
import subprocess

# Required packages: (import_name, pip_name, min_version)
REQUIRED_PACKAGES = [
    ("numpy", "numpy", "1.20.0"),
    ("scipy", "scipy", "1.7.0"),
    ("matplotlib", "matplotlib", "3.5.0"),
    ("plotly", "plotly", "5.0.0"),
    ("pytictoc", "pytictoc", "1.5.0"),
]

# CuPy is special - needs CUDA version detection
CUPY_PACKAGE = ("cupy", "cupy-cuda12x", "12.0.0")


def check_package(import_name, min_version=None):
    """Check if a package is installed and meets version requirement."""
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def check_cupy():
    """Check CuPy and CUDA availability."""
    try:
        import cupy as cp
        version = cp.__version__

        # Test GPU access
        try:
            x = cp.array([1, 2, 3])
            del x
            cp.get_default_memory_pool().free_all_blocks()
            return True, version, True
        except Exception as e:
            return True, version, False
    except ImportError:
        return False, None, False


def install_package(pip_name):
    """Install a package using pip."""
    print(f"  Installing {pip_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("=" * 60)
    print("MAGNET ARRAY OPTIMIZATION - DEPENDENCY CHECK")
    print("=" * 60)
    print()

    all_ok = True
    missing = []

    # Check standard packages
    print("Checking required packages...")
    print("-" * 60)

    for import_name, pip_name, min_ver in REQUIRED_PACKAGES:
        installed, version = check_package(import_name)
        if installed:
            print(f"  [OK] {import_name:15} v{version}")
        else:
            print(f"  [X]  {import_name:15} NOT INSTALLED")
            missing.append((import_name, pip_name))
            all_ok = False

    # Check CuPy (GPU)
    print()
    print("Checking GPU support...")
    print("-" * 60)

    cupy_installed, cupy_version, gpu_works = check_cupy()
    if cupy_installed:
        print(f"  [OK] cupy            v{cupy_version}")
        if gpu_works:
            print(f"  [OK] GPU access      Working")
        else:
            print(f"  [!]  GPU access      FAILED - Check CUDA installation")
            all_ok = False
    else:
        print(f"  [X]  cupy            NOT INSTALLED")
        missing.append(("cupy", CUPY_PACKAGE[1]))
        all_ok = False

    print()
    print("=" * 60)

    if all_ok:
        print("All dependencies are installed and working!")
        print()
        print("You can now run the optimization notebooks:")
        print("  - ellipse_optimization_usage.ipynb")
        print("  - cylindrical_optimization_usage.ipynb")
        return 0

    # Offer to install missing packages
    if missing:
        print(f"Missing packages: {', '.join(p[0] for p in missing)}")
        print()

        response = input("Install missing packages? [y/N]: ").strip().lower()
        if response == 'y':
            print()
            for import_name, pip_name in missing:
                if install_package(pip_name):
                    print(f"  [OK] {import_name} installed successfully")
                else:
                    print(f"  [X]  {import_name} installation failed")

            print()
            print("Re-run this script to verify installation.")
        else:
            print()
            print("To install manually:")
            print(f"  pip install {' '.join(p[1] for p in missing)}")

    if not gpu_works and cupy_installed:
        print()
        print("GPU access failed. Please check:")
        print("  1. NVIDIA GPU driver is installed")
        print("  2. CUDA Toolkit is installed and in PATH")
        print("  3. CuPy version matches your CUDA version")
        print()
        print("For CUDA 12.x: pip install cupy-cuda12x")
        print("For CUDA 11.x: pip install cupy-cuda11x")

    return 1


if __name__ == "__main__":
    sys.exit(main())
