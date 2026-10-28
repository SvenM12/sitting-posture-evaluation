import sys
import platform
import subprocess

print("OS:", platform.platform())
print("Python:", sys.executable)

try:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=10
    )
    print(result.stdout if result.returncode == 0
          else "nvidia-smi error:\n" + result.stderr)
except Exception as e:
    print("NVIDIA GPU check:", e)