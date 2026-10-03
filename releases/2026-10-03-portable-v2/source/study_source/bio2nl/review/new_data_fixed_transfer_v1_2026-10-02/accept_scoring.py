"""Technical fixed-inference acceptance, no performance-based selection."""
import gzip
import json
import math
import shutil
from common import HERE, COUNTS, ROLES, check_gate, read, require, sha, atomic, now, check_hashes, path_inside, canonical, M2_RESULTS, gate_file_hashes


def prediction_space_bound(manifest):
    maxima={};prefixes={}
    for role in ROLES:
        rowfile=path_inside(HERE,manifest['roles'][role]['rows'])
        with gzip.open(rowfile,'rt') as stream:
            rows=[json.loads(line) for line in stream]
            ids=[(len(r['row_id']),len(r['group_id'])) for r in rows]
            require(len(rows)>=64,'Missing fixed first64 scoring rows')
            prefixes[role]={'row_ids_sha256':canonical([r['row_id'] for r in rows[:64]]),
                            'group_ids_sha256':canonical([r['group_id'] for r in rows[:64]])}
        require(len(ids)==COUNTS[role],'Prediction budget row count changed')
        maxima[role]={'row_id_chars':max(x[0] for x in ids),'group_id_chars':max(x[1] for x in ids)}
    predicted=sum(COUNTS[role]*(16+8+8+4*(v['row_id_chars']+v['group_id_chars']))*30 for role,v in maxima.items())
    return {'all_formal_prediction_files':60,'all_formal_prediction_rows':sum(COUNTS.values())*30,
            'maximum_id_lengths':maxima,'first64_identities':prefixes,'uncompressed_prediction_upper_bytes':predicted+60*4096,
            'metadata_and_logs_allowance_bytes':256*(1<<20),'future_allocation_bytes':predicted+60*4096+256*(1<<20)}


def validate_smoke_report(audit, expected, prefixes, gate_files):
    require(audit['status']=='passed','Smoke not accepted')
    for key in ('execution_freeze_sha256','protocol_sha256','prepared_manifest_sha256','source_selection_sha256','checkpoint'):
        require(audit[key]==expected[key],'Smoke identity differs: '+key)
    require(audit['complete_head_retained'] is True and audit['all_input_hashes_unchanged'] is True and
            audit['state_before_sha256']==audit['state_after_sha256']==expected['checkpoint']['state_sha256'],
            'Complete selected source state changed')
    require(type(audit['optimizer_updates_performed']) is int and audit['optimizer_updates_performed']==0 and
            audit['training_or_selection_performed'] is False and audit['target_calibration'] is False and
            audit['polarity_flipping'] is False,'Training, selection or calibration is forbidden')
    require(audit['fixed_absolute_tolerance']==1e-5 and set(audit['roles'])==set(ROLES),'Fixed replay contract differs')
    for role in ROLES:
        row=audit['roles'][role];difference=row['maximum_log_probability_difference']
        require(type(row['rows']) is int and row['rows']==64 and row['all_decisions_equal'] is True and
                type(difference) in (int,float) and math.isfinite(difference) and 0<=difference<=1e-5,
                'First64 replay or decisions failed: '+role)
        require(all(row[key]==value for key,value in prefixes[role].items()),'First64 row/group identity differs: '+role)
    require(isinstance(audit['files'],dict) and all(audit['files'].get(path)==digest for path,digest in gate_files.items()),
            'Smoke input closure omits or changes a gate file')


def main():
    protocol=check_gate(HERE,mode='smoke')
    gate_files=gate_file_hashes(HERE,mode='smoke')
    require(not (HERE/'scoring_acceptance.json').exists(),'Acceptance already exists')
    p=HERE/'verification/smoke_audit.json';audit_sha=sha(p);audit=read(p)
    check_hashes(HERE,audit['files'])
    manifest=read(HERE/'prepared/manifest.json');space=prediction_space_bound(manifest)
    selected=[v for v in read(protocol['source_selection']['file'])['selected']
              if v['job']==dict(condition='EP',pt_seed=0,ft_seed=0,smoke=False)]
    require(len(selected)==1,'Prespecified EP pt0 ft0 source classifier missing')
    checkpoint=selected[0]['checkpoint']
    expected={'execution_freeze_sha256':sha(HERE/'execution_freeze.json'),'protocol_sha256':sha(HERE/'protocol.json'),
              'prepared_manifest_sha256':sha(HERE/'prepared/manifest.json'),'source_selection_sha256':protocol['source_selection']['sha256'],
              'checkpoint':{'file':str(path_inside(M2_RESULTS,checkpoint['file'])),'sha256':checkpoint['sha256'],'state_sha256':checkpoint['state_sha256']}}
    validate_smoke_report(audit,expected,space['first64_identities'],gate_files)
    free=shutil.disk_usage(HERE).free
    require(free-space['future_allocation_bytes']>=8*(1<<30),'Insufficient future output allocation plus8GiB reserve')
    files=dict(audit['files']);files[str(p)]=audit_sha
    check_hashes(HERE,files)
    atomic(HERE/'scoring_acceptance.json',{'status':'accepted_for_fixed_scoring','created_at_utc':now(),
        'execution_freeze_sha256':sha(HERE/'execution_freeze.json'),'protocol_sha256':sha(HERE/'protocol.json'),
        'prepared_manifest_sha256':sha(HERE/'prepared/manifest.json'),'files':files,
        'disk_free_bytes':free,'space_estimate':space,'projected_peak_free_gib':(free-space['future_allocation_bytes'])/(1<<30),
        'reserve_bytes':8<<30,'complete_models_reused_without_copy':True,
        'performance_used_for_acceptance':False,'training_enabled':False})
    print('All fixed scoring accepted; no performance criterion',flush=True)

if __name__=='__main__':main()
