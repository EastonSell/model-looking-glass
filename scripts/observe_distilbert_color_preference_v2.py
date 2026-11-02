"""Use the reviewed direct-supervisor observer with the v2 screen owner."""
from scripts import observe_distilbert_color_supervisor_v1 as observer

BOOTSTRAP = ("import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
             "sys.path.insert(1,sys.argv.pop(1));"
             "runpy.run_module('looking_glass.distilbert_color_preference_runner_v2',run_name='__main__')")


def main():
    observer.BOOTSTRAP = BOOTSTRAP
    observer.main()


if __name__ == '__main__':
    main()
