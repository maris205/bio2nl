import copy
import pytest
import accept_scoring as acceptance


def fixture():
    checkpoint={'file':'/new/source/EP/pt0/ft0/best.pt','sha256':'a'*64,'state_sha256':'b'*64}
    expected={key:'c'*64 for key in ('execution_freeze_sha256','protocol_sha256','prepared_manifest_sha256','source_selection_sha256')}
    expected['checkpoint']=checkpoint
    prefixes={role:{'row_ids_sha256':'d'*64,'group_ids_sha256':'e'*64} for role in acceptance.ROLES}
    gates={'/protocol.json':'c'*64}
    audit={**expected,'status':'passed','complete_head_retained':True,'all_input_hashes_unchanged':True,
           'state_before_sha256':'b'*64,'state_after_sha256':'b'*64,'optimizer_updates_performed':0,
           'training_or_selection_performed':False,'target_calibration':False,'polarity_flipping':False,
           'fixed_absolute_tolerance':1e-5,'files':dict(gates),
           'roles':{role:{'rows':64,'all_decisions_equal':True,'maximum_log_probability_difference':0.,**prefixes[role]} for role in acceptance.ROLES}}
    return copy.deepcopy(audit),copy.deepcopy(expected),prefixes,gates


def test_complete_fixed_smoke_accepted():
    acceptance.validate_smoke_report(*fixture())


@pytest.mark.parametrize('case',['wrong_checkpoint','stale_manifest','changed_weights','updates','calibration',
                                'wrong_prefix','changed_decisions','rounding_above_threshold','nonfinite','partial_closure'])
def test_technical_smoke_mutations_rejected(case):
    audit,expected,prefixes,gates=copy.deepcopy(fixture())
    if case=='wrong_checkpoint':audit['checkpoint']['file']='/wrong/best.pt'
    elif case=='stale_manifest':audit['prepared_manifest_sha256']='f'*64
    elif case=='changed_weights':audit['state_after_sha256']='f'*64
    elif case=='updates':audit['optimizer_updates_performed']=1
    elif case=='calibration':audit['target_calibration']=True
    elif case=='wrong_prefix':audit['roles']['target']['row_ids_sha256']='f'*64
    elif case=='changed_decisions':audit['roles']['target']['all_decisions_equal']=False
    elif case=='rounding_above_threshold':audit['roles']['target']['maximum_log_probability_difference']=1.0001e-5
    elif case=='nonfinite':audit['roles']['target']['maximum_log_probability_difference']=float('nan')
    else:audit['files']={}
    with pytest.raises(ValueError):acceptance.validate_smoke_report(audit,expected,prefixes,gates)
