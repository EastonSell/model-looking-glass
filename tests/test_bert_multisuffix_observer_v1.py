"""No-model completion and ineligibility checks for the multisuffix observer."""
import hashlib,json
import pytest
from scripts import observe_bert_multisuffix_agreement_v1 as observer


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path,value): path.write_text(json.dumps(value),encoding='utf-8')


class Process:
    pid=1234
    def wait(self,timeout=None): return 0
    def poll(self): return 0


@pytest.mark.parametrize('eligible,events,passes', [(32,1088,True),(0,160,True),(31,1059,True),(31,1088,False)])
def test_observer_accepts_only_schedule_consistent_completion(tmp_path,monkeypatch,eligible,events,passes):
    attempt=tmp_path/'attempt';attempt.mkdir()
    controller=tmp_path/'controller';controller.mkdir()
    plan=attempt/'plan.json';dump(plan,{'synthetic':True})
    commit='a'*40; hashes={'source':'hash'}; git_sha='b'*64
    meta=attempt/'metadata.json';dump(meta,{'pid':1234,'source_commit':commit,'sources_sha256':hashes,'git_sha256':git_sha})
    dump(attempt/'complete.json',{'status':'complete','eligible_cases':eligible,'events':events,
                                  'plan_json_sha256':sha(plan),'metadata_json_sha256':sha(meta)})
    monkeypatch.setattr(observer.subprocess,'Popen',lambda *a,**kw:Process())
    monkeypatch.setattr(observer,'checked_sources',lambda *args:hashes)
    monkeypatch.setattr(observer,'digest',lambda p:git_sha if p==tmp_path/'git' else sha(p))
    receipt=observer.observe([],controller,attempt,commit,{},10,tmp_path,tmp_path/'git',hashes,git_sha)
    assert (receipt['status']=='complete') is passes
    assert (controller/observer.PENDING_NAME).exists() is (not passes)


def test_harmless_direct_child_completion_and_failure_sentinel(tmp_path,monkeypatch):
    import os,shutil,subprocess,sys
    commit='a'*40;hashes={'source':'c'*64}
    git=__import__('pathlib').Path(shutil.which('git')).resolve()
    git_sha=observer.digest(git)
    monkeypatch.setattr(observer,'checked_sources',lambda *_:hashes)
    for sentinel in (False,True):
        controller=tmp_path/('controller-'+str(sentinel)); controller.mkdir()
        attempt=tmp_path/('attempt-'+str(sentinel))
        code=(
            'import hashlib,json,os;from pathlib import Path;'
            f'p=Path({str(attempt)!r});p.mkdir();'
            f'meta={{"pid":os.getpid(),"source_commit":{commit!r},"sources_sha256":{hashes!r},"git_sha256":{git_sha!r}}};'
            '(p/"metadata.json").write_text(json.dumps(meta));'
            '(p/"plan.json").write_text(json.dumps({"synthetic":True}));'
            'mh=hashlib.sha256((p/"metadata.json").read_bytes()).hexdigest();'
            'ph=hashlib.sha256((p/"plan.json").read_bytes()).hexdigest();'
            '(p/"complete.json").write_text(json.dumps({"status":"complete","events":1059,"eligible_cases":31,"metadata_json_sha256":mh,"plan_json_sha256":ph}));'
            + ('(p/"failure.txt").write_text("failed after completion");' if sentinel else '')
        )
        receipt=observer.observe([sys._base_executable,'-B','-c',code],controller,attempt,commit,os.environ.copy(),30,tmp_path,git,hashes,git_sha)
        assert (receipt['status']=='complete') is (not sentinel)
        assert (controller/observer.PENDING_NAME).exists() is sentinel


def test_interrupted_receipt_publication_keeps_pending_marker(tmp_path,monkeypatch):
    import os,shutil,sys
    from pathlib import Path
    commit='a'*40;hashes={'source':'c'*64}
    git=Path(shutil.which('git')).resolve(); git_sha=observer.digest(git)
    monkeypatch.setattr(observer,'checked_sources',lambda *_:hashes)
    controller=tmp_path/'controller';controller.mkdir();attempt=tmp_path/'attempt'
    save=observer.save_json
    def interrupted(path,value):
        if path.name=='exit-observation.json':
            path.write_text(json.dumps(value),encoding='utf-8')
            raise KeyboardInterrupt('synthetic interruption after write')
        return save(path,value)
    monkeypatch.setattr(observer,'save_json',interrupted)
    code=(
        'import hashlib,json,os;from pathlib import Path;'
        f'p=Path({str(attempt)!r});p.mkdir();'
        f'meta={{"pid":os.getpid(),"source_commit":{commit!r},"sources_sha256":{hashes!r},"git_sha256":{git_sha!r}}};'
        '(p/"metadata.json").write_text(json.dumps(meta));'
        '(p/"plan.json").write_text(json.dumps({"synthetic":True}));'
        'mh=hashlib.sha256((p/"metadata.json").read_bytes()).hexdigest();'
        'ph=hashlib.sha256((p/"plan.json").read_bytes()).hexdigest();'
        '(p/"complete.json").write_text(json.dumps({"status":"complete","events":1088,"eligible_cases":32,"metadata_json_sha256":mh,"plan_json_sha256":ph}));'
    )
    with pytest.raises(KeyboardInterrupt):
        observer.observe([sys._base_executable,'-B','-c',code],controller,attempt,commit,os.environ.copy(),30,tmp_path,git,hashes,git_sha)
    assert (controller/observer.PENDING_NAME).is_file()


def test_owner_and_observer_bind_the_same_source_set():
    from scripts import bert_multisuffix_agreement_v1 as owner
    assert owner.SOURCE_PATHS == observer.SOURCE_PATHS
    assert 'docs/BERT_MULTISUFFIX_EXECUTION_V1.md' in owner.SOURCE_PATHS
    assert owner.SOURCE_PATHS[-1] == 'experiments/bert_multisuffix_agreement_cases_v1.json'
