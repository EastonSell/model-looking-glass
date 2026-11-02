"""Observe the reviewed direct supervisor for the antonym baseline screen."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import observe_distilbert_color_supervisor_v1 as observer

BOOTSTRAP = ("import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
             "sys.path.insert(1,sys.argv.pop(1));"
             "runpy.run_module('looking_glass.distilbert_antonym_runner_v1',run_name='__main__')")


def main():
    observer.BOOTSTRAP = BOOTSTRAP
    observer.main()


if __name__ == '__main__':
    main()
