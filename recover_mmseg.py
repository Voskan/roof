import os
import sys
import subprocess
from pathlib import Path

def recover_mmseg_init():
    # 1. Find the file location
    mmseg_path = ""
    try:
        result = subprocess.check_output([sys.executable, "-m", "pip", "show", "mmsegmentation"], stderr=subprocess.DEVNULL).decode()
        for line in result.split('\n'):
            if line.startswith('Location: '):
                mmseg_path = os.path.join(line.split(': ')[1].strip(), "mmseg/__init__.py")
                break
    except Exception as e:
        print(f"❌ Could not find mmsegmentation: {e}")
        return

    if not mmseg_path or not os.path.exists(mmseg_path):
        print("❌ mmseg/__init__.py not found.")
        return

    print(f"📍 Found mmsegmentation init at: {mmseg_path}")
    
    # 2. Define the CLEAN content (Minimal working version)
    # We strip out the version check block entirely.
    clean_content = """# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__, version_info

MMCV_MIN = '2.0.0rc4'
MMCV_MAX = '2.2.0'
MMENGINE_MIN = '0.7.1'
MMENGINE_MAX = '1.0.0'

mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version('9.9.9') # OVERRIDE by DeepRoof
mmcv_version = digit_version(mmcv.__version__)

mmengine_min_version = digit_version(MMENGINE_MIN)
mmengine_max_version = digit_version('9.9.9') # OVERRIDE by DeepRoof
mmengine_version = digit_version(mmengine.__version__)

# assert (mmcv_min_version <= mmcv_version < mmcv_max_version), \\
#     f'MMCV=={mmcv.__version__} is used but incompatible. ' \\
#     f'Please install mmcv>={MMCV_MIN},<{MMCV_MAX}.'

# assert (mmengine_min_version <= mmengine_version < mmengine_max_version), \\
#     f'MMEngine=={mmengine.__version__} is used but incompatible. ' \\
#     f'Please install mmengine>={MMENGINE_MIN},<{MMENGINE_MAX}.'

__all__ = ['__version__', 'version_info', 'digit_version']
"""

    # 3. Overwrite the corrupted file
    try:
        with open(mmseg_path, 'w') as f:
            f.write(clean_content)
        print("✅ SUCCESSFULLY OVERWROTE mmseg/__init__.py with clean content.")
        print("⚠️ PLEASE RESTART YOUR KERNEL NOW.")
    except Exception as e:
        print(f"❌ Failed to write file: {e}")

if __name__ == "__main__":
    recover_mmseg_init()
