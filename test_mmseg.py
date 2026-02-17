import sys
print(sys.path)
try:
    import mmseg
    print("mmseg path:", mmseg.__path__)
    from mmseg.models import segmentors
    print("segmentors path:", segmentors.__path__)
    print("segmentors dir:", dir(segmentors))
except ImportError as e:
    print("ImportError:", e)
