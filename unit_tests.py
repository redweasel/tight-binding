# this file has all unit tests

from symmetry import *
from unitary_representations import *
from hamiltonian_symmetry import *
from kpaths import *
from bandstructure import *
from json_tb_format import *


# all the functions can be enumerated automatically
import traceback
import time
for name, f in list(globals().items()):
    if name.startswith("test_") and callable(f):
        try:
            #print(f"run unit test {name}")
            now = time.time()
            f()
            print(f"\033[92mpassed unit test {name} in {time.time() - now:.3f}s\033[0m")
        except Exception as e:
            print(f"\033[91mfailed unit test {name}\033[0m")
            traceback.print_tb(e.__traceback__)
            print(repr(e))
