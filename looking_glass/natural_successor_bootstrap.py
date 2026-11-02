"""Source-first isolated bootstrap candidate for the successor owner.

The caller must independently pin this exact bootstrap, the approved site,
owner bytes and process identity. This is not a source lease or process audit.
"""

BOOTSTRAP = '''import hashlib,os,sys
if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or sys.flags.optimize
        or any(name in os.environ for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY"))):
    raise SystemExit(1)
p=sys.argv.pop(1); h=sys.argv.pop(1); requested=sys.argv.pop(1)
site=os.path.realpath(requested); selected=os.environ.get("MLG_NATURAL_SITE", "")
if (not os.path.isabs(p) or not os.path.isabs(requested) or not os.path.isabs(selected)
        or not os.path.isdir(site) or os.path.normcase(site)!=os.path.normcase(os.path.realpath(selected))
        or os.path.basename(site).lower()!="site-packages"
        or os.path.basename(os.path.dirname(site)).lower()!="lib"
        or any(os.path.normcase(os.path.realpath(x))==os.path.normcase(site) for x in sys.path if x)):
    raise SystemExit(1)
data=open(p,"rb").read(8388609)
if len(data)>8388608 or hashlib.sha256(data).hexdigest()!=h:
    raise SystemExit(1)
sys.path.append(site)
sys.argv[0]=p
exec(compile(data,p,"exec",dont_inherit=True),{"__name__":"__main__","__file__":p})
'''
