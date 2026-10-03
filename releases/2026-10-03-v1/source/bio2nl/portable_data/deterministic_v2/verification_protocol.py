"""Predeclared, closed comparison coverage for the new deterministic raw release."""
from pathlib import Path

SUCCESS = 'accepted_new_raw_release_training_disabled'
PAIR_FILES = tuple(f'data/pairs/protein_sequence_similarity_{s}.tsv.gz' for s in ('all', 'train', 'validation', 'test'))
PAIR_REFERENCE = (*PAIR_FILES, 'validation/protein_endpoint_degrees.tsv',
                  'validation/protein_pair_baselines.csv', 'metadata/protein_pair_construction.json',
                  'validation/protein_pair_acceptance.json')
EXCLUDED = ('README.md', 'metadata/code_snapshot_bio2nl.json', 'metadata/code_snapshot_biopaws.json',
            'metadata/nlp_synthetic_README.md', 'validation/integrated_release_acceptance.json')
IDENTITY_FIELDS = {
    'configs/training_matrix_v2.json': 'data_release_id',
    'configs/smallweb_gpt2_v2.json': 'data_release_id',
    'metadata/synthetic_longrange_manifest.json': 'release_id',
    'metadata/protein_pair_manifest.json': 'release_id',
    'metadata/remote_manifest.json': 'release_id',
}
SPECIAL = {
    'validation/nlp_clean_exclusions.jsonl': 'exact_deterministic_conflict_ledger',
    'metadata/protein_pair_manifest.json': 'new_pair_manifest_current_descriptors',
    'metadata/remote_manifest.json': 'new_remote_manifest_pinned_historical_erratum',
    'validation/cross_dataset_homology.json': 'independent_current_cross_dataset_recomputation',
}
EXTRA = ('metadata/protein_candidate_order.json',)
SORTED_CANDIDATES = 'work/protein/positive_candidates_sorted.jsonl'
CANONICAL_COLUMNS = ['accession', 'entry_name', 'taxon_id', 'sequence_version', 'sequence_sha256',
                     'sequence', 'length', 'pretrain_eligible', 'pretrain_exclusion_reason', 'cluster_id', 'split']
OLD_HELPER_SHA = '861367828e362e3b04108bd7a610ff1aac48cbffcef824018d2f2e71570f1696'
BOOTSTRAP_SHA = '0eba41c54cc07586869a3c0cbcbe6092e2a6300df1a9a0e93ea0ae67d003fc77'
PAIR_AUDITOR_SHA = 'ee0b567e9b1676568a9eb71d4ccefbac8d82eeadaa6e997eece2cfc2a5ba342e'
REPLAY_AUDIT_SHA = '7a0cadb16e38038dda2cc603646badeaa15a20a1a3ec7d6f4ad99f8b7144ab20'
REMOTE_MANIFEST_SHA = 'f75596d56a1b8a129bb292210b7ad3a2d6b21d20f2999aae2d469d5d4d0da8ec'
REMOTE_SOURCE_SHA = '16535a38bfec4057cc27d6799b3fb299d6b34eac9868f18c4a4fd1d06f0b50c0'
REMOTE_STALE_SHA = '513f71203094bbb7a0a08c7e7bfe5448f85c8d657da1535a651b120afb737232'


def make_policy(entries, reference_root, replay_root, replay_audit):
    """Create exact coverage from already authenticated reference inventories."""
    reference_root, replay_root = Path(reference_root), Path(replay_root)
    if len(entries) != 336 or not all(n in entries for n in EXCLUDED):
        raise ValueError('Accepted reference coverage changed')
    if replay_audit['status'] != 'pair_replay_determinism_passed':
        raise ValueError('Deterministic reference lacks completed independent audit')
    files = {}
    for name in sorted(set(entries) - set(EXCLUDED)):
        kind = 'accepted_v2'
        source = reference_root / name
        pin = entries[name]
        mode = SPECIAL.get(name, 'accepted_scientific_content_with_strict_current_metadata')
        if name in PAIR_REFERENCE:
            kind, source = 'audited_deterministic_pair_replay', replay_root / name
            pin = replay_audit['verified_files'][str(source)]
            mode = 'deterministic_pair_reference_exact_except_construction_runtime'
        files[name] = {'reference_kind': kind, 'reference_file': str(source),
                       'reference_sha256': pin['sha256'], 'reference_bytes': pin['bytes'], 'mode': mode}
    return {'schema_version': 1, 'files': files, 'file_count': 331,
            'required_new_files': list(EXTRA), 'required_execution_builder_files': 28,
            'excluded_historical_files': list(EXCLUDED), 'identity_fields': IDENTITY_FIELDS,
            'canonical_columns': CANONICAL_COLUMNS, 'scientific_tolerance': 0,
            'model_input_row_order': 'exact', 'tokenizer_ids_and_merges': 'exact',
            'required_token_cells': 33, 'success_status': SUCCESS,
            'training_enabled': False, 'target_scoring_enabled': False,
            'exact_historical_pair_reproduction_claimed': False}
