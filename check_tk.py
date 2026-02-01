import sys
import os

print(f"--- PYTHON DEBUG INFO ---")
print(f"Executable: {sys.executable}")
print(f"Version: {sys.version}")

try:
    import tkinter
    import _tkinter
    print(f"\n✅ Tkinter Import: SUCCESS")
    print(f"Tkinter Path: {os.path.dirname(tkinter.__file__)}")
    print(f"TCL Library: {os.environ.get('TCL_LIBRARY', 'Not Set')}")
    print(f"TK Library: {os.environ.get('TK_LIBRARY', 'Not Set')}")
except ImportError as e:
    print(f"\n❌ Tkinter Import: FAILED")
    print(f"Error: {e}")

input("\nPress Enter to exit...")