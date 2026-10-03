"""Bind completed new source models before reading held-out example content."""
from pathlib import Path
import time
from common import (HERE, M2, M2_RESULTS, M1, NEW_RAW, NEW_PREPARED, ARCHIVE,
                    read, sha, atomic, require, now, check_hashes, check_source_gate, canonical)


def main():
    start=time.monotonic()
    require(not (HERE/'source_barrier.json').exists() and not (HERE/'protocol.json').exists(),'Source freeze already exists')
    require(not (HERE/'prepared').exists() and not (HERE/'results').exists(),'Held-out work preceded source freeze')
    review_path=M2/'verification/completion_integrity_review_2026-10-02.json'
    review=read(review_path);require(review['status']=='passed','Source completion review failed')
    check_hashes(M2,review['verified_files'])
    selection_path=M2_RESULTS/'source_selection.json';selection=read(selection_path)
    require(selection['status']=='frozen' and len(selection['selected'])==27 and not selection['conditions_filtered_by_score'],'Source selection incomplete')
    require(canonical({k:v for k,v in selection.items() if k!='selection_sha256'})==selection['selection_sha256'],'Source canonical digest differs')
    source_protocol=read(M2/'protocol.json')
    require(sha(M2/'protocol.json')==selection['protocol_sha256'],'Source protocol changed')
    for group in ('raw_release','prepared'):
        for role in ('manifest','acceptance'):
            ref=source_protocol[group][role];check_hashes(HERE,{ref['file']:ref})
    source_manifest=read(NEW_PREPARED/'manifest.json')
    spec=read(HERE/'data_input_spec.json')
    require(len(spec)==17 and {'source_test','target_raw','target_membership'}<=set(spec),'Incomplete fixed data specification')
    delayed={'source_test','target_raw','target_membership'}
    for name,ref in spec.items():
        if name not in delayed:check_hashes(HERE,{ref['file']:ref})
    for name in ('confirmation_acceptance','confirmation_audit','confirmation_manifest','confirmation_references'):
        ref=source_manifest['inputs'][name];require(Path(ref['path']).is_relative_to(M1),'Wrong historical qualification root')
        check_hashes(HERE,{ref['path']:ref})
    paths=[review_path,selection_path,M2/'execution_status.json',M2_RESULTS/'report/audit.json',
           M2_RESULTS/'report/summary.json',M2_RESULTS/'report/REPORT.md',M2/'protocol.json',M2/'STATISTICAL_PLAN.md',
           M2/'training_acceptance.json',M2/'code_snapshot/model.py',M2/'code_snapshot/surface_features.py',
           M2/'verification/surface_audit.json',M2_RESULTS/'surface/selection.json',M2_RESULTS/'surface/lambda0.json',
           NEW_PREPARED/'manifest.json',NEW_PREPARED.parent/'data_acceptance.json',
           NEW_RAW/'manifest.json',NEW_RAW/'portable_validation/acceptance.json',ARCHIVE/'archive_manifest.json']
    paths += [Path(v['file']) for k,v in spec.items() if k not in delayed]
    for item in selection['selected']:
        for key in ('checkpoint','source_result'):
            ref=item[key];p=(M2_RESULTS/ref['file']).resolve();require(p.is_relative_to(M2_RESULTS),'Source artifact path escape')
            check_hashes(HERE,{str(p):ref});paths.append(p)
    files={str(p):sha(p) for p in sorted(set(paths))}
    barrier={'status':'frozen_sources_for_heldout_evaluation','created_at_utc':now(),
             'completion_review':{'file':str(review_path),'sha256':sha(review_path)},
             'source_selection_file_sha256':sha(selection_path),'source_selection_canonical_sha256':selection['selection_sha256'],
             'source_classifiers':27,'source_conditions_filtered':False,'heldout_examples_prepared':False,
             'target_raw_or_membership_parsed':False,'source_test_pairs_parsed':False,'files':files}
    atomic(HERE/'source_barrier.json',barrier)
    def ref(p):return {'file':str(p),'sha256':sha(p)}
    old=read(HERE.parent/'joint_transfer_v1_2026-09-30/protocol.json')
    protocol={k:v for k,v in old.items() if k not in ('created_at_utc','source_barrier_sha256','source_selection','model_loader','feature_loader','surface_checkpoint','statistical_plan','roles','limitations')}
    protocol.update(created_at_utc=now(),scope='new_data_fixed_source_test_and_QQP_transfer',
        user_authorization='2026-10-02 user approved next planned fixed-classifier transfer evaluation after completed new-data source training',
        source_barrier_sha256=sha(HERE/'source_barrier.json'),source_selection=ref(selection_path),
        source_protocol=ref(M2/'protocol.json'),model_loader=ref(M2/'code_snapshot/model.py'),
        feature_loader=ref(M2/'code_snapshot/surface_features.py'),surface_checkpoint=ref(M2_RESULTS/'surface/lambda0.json'),
        statistical_plan=ref(HERE/'STATISTICAL_PLAN.md'),transfer_protocol=ref(HERE/'TRANSFER_PROTOCOL.md'),
        data_inputs=spec,source_prepared_manifest_sha256=selection['prepared_manifest_sha256'],
        source_test_scoring_enabled=True,target_scoring_enabled=True,new_blind_confirmation_claimed=False,
        target_membership_selection_changed=False,training_enabled=False,source_weights_updated=False,
        roles={'source_test':{'rows':20854,'meaning':'new reconstructed portable-v2 operational protein similarity test pairs; same fixed upstream proteins, not strict structural homology'},
               'target':{'rows':39893,'meaning':'all fixed historical M1-qualified public QQP validation rows reconstructed from raw and fixed membership, previously scored; exploratory evaluation'}},
        limitations=['Weak source competence accompanies target interpretation',
        'English-only EE has twice the English exposure of EP/ES under equal total-token budgets',
        'Natural/shuffled fixed-BPE budgets have unequal residue and parent-prefix exposure',
        'QQP exact cohort has been historically scored; not new blind confirmation',
        'The protein pairs are newly reconstructed from the same upstream sequence pool; no new biological acquisition',
        'Shared target endpoint components are retained; no row-independent significance claims',
        'Raw QQP and reversible encodings remain local; redistribution not cleared'])
    protocol['queue'].update(workspace_lock=str(HERE.parents[2]/'.bio2nl_gpu_training_queue.lock'),
                             gpu_memory_below_mib=500,gpu_utilization_at_most_percent=10,gpu_poll_max_attempts=16,
                             minimum_launch_free_gib=9)
    atomic(HERE/'protocol.json',protocol)
    check_source_gate(HERE)
    atomic(HERE/'verification/source_freeze_review.json',{'status':'passed','completed_at_utc':now(),
        'elapsed_seconds':time.monotonic()-start,'protocol_sha256':sha(HERE/'protocol.json'),
        'source_barrier_sha256':sha(HERE/'source_barrier.json'),'source_checkpoint_count':27,
        'fresh_completion_files_reverified':len(review['verified_files']),'source_barrier_files':len(files),
        'target_raw_or_membership_parsed':False,'source_test_pairs_parsed':False,'models_loaded':False,
        'files':{str(HERE/'source_barrier.json'):sha(HERE/'source_barrier.json'),str(HERE/'protocol.json'):sha(HERE/'protocol.json')}})
    print('New source barrier and transfer protocol frozen',sha(HERE/'protocol.json'),flush=True)

if __name__=='__main__':main()
