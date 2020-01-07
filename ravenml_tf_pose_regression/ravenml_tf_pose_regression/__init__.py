import os
import sys

cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if os.path.join(cwd, 'slim') not in sys.path:
    sys.path.append(os.path.join(cwd, 'slim'))
