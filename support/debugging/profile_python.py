# below is needed to help python references lookups when this
# script is run separately, source https://stackoverflow.com/a/4383597
import sys

STARTUP_DIR = (
    "/mnt/asgard2/code/nadim/case2d_code_copy/"  # change this absolute path to yours
)
sys.path.insert(1, STARTUP_DIR)

import cProfile
import pstats
from os import makedirs
from os.path import join, exists

# docs https://docs.python.org/3/library/profile.html

"""
sorting keywords

'calls'         call count
'cumulative'    cumulative time
'cumtime'	    cumulative time
'file'	        file name
'filename'	    file name
'module'        file name
'ncalls'        call count
'pcalls'        primitive call count
'line'	        line number
'name'	        function name
'nfl'	        name/file/line
'stdname'	    standard name
'time'	        internal time
'tottime'	    internal time

"""
SORT_BY = "cumtime"
SCRIPT_PATH = "start.py"
ROOT_PATH = join("output", "python_profiling")
PROFILING_OUTPUT_PATH = join(ROOT_PATH, "output.prof")
OUTPUT_PATH = join(ROOT_PATH, "profiling_output.txt")

file = open(SCRIPT_PATH, mode="r")
script = file.read()
file.close()

if not exists(ROOT_PATH):
    makedirs(ROOT_PATH)

prof = cProfile.Profile()
prof.run(script)
prof.dump_stats(PROFILING_OUTPUT_PATH)

stream = open(OUTPUT_PATH, "w")
stats = pstats.Stats(PROFILING_OUTPUT_PATH, stream=stream)
stats.strip_dirs()
stats.sort_stats(SORT_BY)
stats.print_stats()
