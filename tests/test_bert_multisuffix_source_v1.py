"""Exercise source binding using isolated, model-free subprocesses."""
from pathlib import Path
import os,shutil,subprocess,sys,sysconfig
from scripts import bert_multisuffix_agreement_v1 as owner


def test_isolated_bootstrap_ignores_checkout_shadow_and_rejects_tree(tmp_path):
    source=Path(__file__).resolve().parents[1]
    checkout=tmp_path/'checkout';checkout.mkdir()
    git=Path(shutil.which('git')).resolve()
    for name in owner.SOURCE_PATHS:
        dst=checkout/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,dst)
    def run_git(*args):
        return subprocess.run([str(git),'-C',str(checkout),*args],check=True,capture_output=True,text=True)
    run_git('init');run_git('config','core.autocrlf','false')
    run_git('add',*owner.SOURCE_PATHS)
    run_git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture')
    commit=run_git('rev-parse','HEAD').stdout.strip()
    marker=tmp_path/'checkout-shadow-ran'
    (checkout/'numbers.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("shadow")\n',encoding='utf-8')
    # Query the active interpreter's native venv scheme before isolated startup.
    site=Path(sysconfig.get_path('purelib')).resolve(strict=True)
    base=Path(sys._base_executable).resolve()
    positive=tmp_path/'positive-prefix';positive.mkdir()
    code=f'import sys;sys.path.insert(0,{str(checkout)!r});sys.path.append({str(site)!r});import numbers'
    subprocess.run([str(base),'-I','-S','-B','-X','pycache_prefix='+str(positive),'-c',code],check=True,capture_output=True)
    assert marker.read_text()=='shadow'
    marker.unlink()
    prefix=tmp_path/'owner-prefix';prefix.mkdir()
    model=tmp_path/'empty-model';model.mkdir()
    cmd=[str(base),'-I','-S','-B','-X','pycache_prefix='+str(prefix),
         str(checkout/'scripts/bert_multisuffix_agreement_v1.py'),'--site',str(site),'--git',str(git),
         '--model-dir',str(model),'--cases',str(checkout/owner.SOURCE_PATHS[-1]),
         '--output',str(tmp_path/'never-created'),'--device','cpu','--source-commit',commit]
    result=subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy(),timeout=90)
    assert result.returncode!=0 and 'model snapshot has missing or additional assets' in result.stderr, result.stderr
    assert not marker.exists() and not (tmp_path/'never-created').exists()
    tree=run_git('cat-file','-p',commit).stdout.splitlines()[0].split()[1]
    prefix2=tmp_path/'tree-prefix';prefix2.mkdir()
    cmd[cmd.index('-X')+1]='pycache_prefix='+str(prefix2);cmd[-1]=tree
    result=subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy(),timeout=30)
    assert result.returncode!=0 and 'source identifier does not name a commit' in result.stderr
    assert not marker.exists()


def test_cuda_guard_precedes_dependency_imports(tmp_path):
    base=Path(sys._base_executable).resolve()
    prefix=tmp_path/'prefix';prefix.mkdir()
    site=tmp_path/'empty-site';site.mkdir()
    owner_script=Path(owner.__file__).resolve()
    cmd=[str(base),'-I','-S','-B','-X','pycache_prefix='+str(prefix),str(owner_script),
         '--site',str(site),'--git',str(tmp_path/'missing-git'),'--cases',str(tmp_path/'missing-cases'),
         '--source-commit','a'*40,'--device','cuda']
    result=subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy(),timeout=30)
    assert result.returncode!=0 and 'physical-lease owner' in result.stderr
    assert 'No module named' not in result.stderr
