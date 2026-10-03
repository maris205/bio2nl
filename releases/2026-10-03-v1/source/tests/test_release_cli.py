import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("release_cli",ROOT/"release_cli.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
fetch_spec=importlib.util.spec_from_file_location("fetch_verified",ROOT/"fetch_verified.py")
fetch=importlib.util.module_from_spec(fetch_spec);fetch_spec.loader.exec_module(fetch)


class ReleaseTests(unittest.TestCase):
    def test_extra_unlisted_source_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/"MANIFEST.json").write_text(json.dumps({"files":[]}));(p/"private.txt").write_text("fixture")
            with self.assertRaisesRegex(ValueError,"closed set"):cli.verify(p)

    def test_extra_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/"MANIFEST.json").write_text(json.dumps({"files":[]}));(p/"private").symlink_to(ROOT,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,"Symlink"):cli.verify(p)

    def test_output_symlink_into_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/"alias").symlink_to(ROOT,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,"symlinks"):cli.safe_output(ROOT,p/"alias/new")

    def test_normalized_output_into_source_rejected(self):
        with self.assertRaisesRegex(ValueError,"immutable"):cli.safe_output(ROOT,ROOT.parent/"source/../source/new")

    def test_real_nested_statistics(self):
        value=cli.report(ROOT)
        self.assertEqual(value["neural_cells"],54)
        self.assertEqual(value["reference_cells"],6)
        self.assertTrue(value["primary_direction_consistency_flag"])
        self.assertAlmostEqual(value["primary_target_auc_contrast"]["mean_difference"],0.02955986363218314,14)
        self.assertAlmostEqual(value["primary_target_auc_contrast"]["pretraining_sample_sd"],0.018654733217295428,14)

    def test_traversal_refused(self):
        with self.assertRaisesRegex(ValueError,"Unsafe"):cli.member(ROOT,"../private")

    def test_absolute_path_refused(self):
        with self.assertRaisesRegex(ValueError,"Unsafe"):cli.member(ROOT,"/tmp/private")

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/"link").symlink_to(ROOT/"release_cli.py")
            with self.assertRaisesRegex(ValueError,"Symlinked"):cli.member(p,"link")

    def test_same_size_tamper_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/"data").write_bytes(b"first")
            record={"path":"data","bytes":5,"sha256":cli.digest(p/"data")}
            (p/"data").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError,"SHA256"):cli.verify_files(p,[record])

    def test_duplicate_member_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/"data").write_bytes(b"x");r={"path":"data","bytes":1,"sha256":cli.digest(p/"data")}
            with self.assertRaisesRegex(ValueError,"Duplicate"):cli.verify_files(p,[r,r])

    def test_missing_cell_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);shutil.copytree(ROOT/"results",p/"results")
            f=p/"results/oct2/fixed_transfer/per_job_metrics.csv";lines=f.read_text().splitlines();f.write_text("\n".join(lines[:-1])+"\n")
            with self.assertRaisesRegex(ValueError,"54 unique"):cli.report(p)

    def test_wrong_aggregation_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);shutil.copytree(ROOT/"results",p/"results")
            f=p/"results/oct2/fixed_transfer/nested_metrics.csv"
            with f.open(newline="") as stream:r=csv.DictReader(stream);names=r.fieldnames;rows=list(r)
            rows[0]["pretraining_sample_sd"]="0.1234"
            with f.open("w",newline="") as stream:w=csv.DictWriter(stream,fieldnames=names);w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,"Aggregate mismatch"):cli.report(p)

    def test_wrong_direction_flag_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);shutil.copytree(ROOT/"results",p/"results")
            f=p/"results/oct2/fixed_transfer/summary.json";d=json.loads(f.read_text());d["primary_direction_consistency_flag"]=False;f.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"direction flag"):cli.report(p)

    def test_size_only_is_explicitly_not_hash_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/"data").write_bytes(b"x")
            self.assertEqual(cli.verify_files(p,[{"path":"data","bytes":1,"sha256":"wrong"}],hash_content=False),1)

    def test_fetch_verified_bytes_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);payload=b"locked input";recipe=p/"recipe.json";destination=p/"input"
            recipe.write_text(json.dumps({"sources":[{"id":"fixture","url":"https://example.org/fixed","bytes":len(payload),"sha256":cli.hashlib.sha256(payload).hexdigest()}]}))
            response=io.BytesIO(payload);response.geturl=lambda:"https://example.org/fixed"
            with patch("sys.argv",["fetch_verified.py","--recipe",str(recipe),"--id","fixture","--destination",str(destination)]),patch.object(fetch.urllib.request,"urlopen",return_value=response),patch("sys.stdout",new_callable=io.StringIO):
                fetch.main()
            self.assertEqual(destination.read_bytes(),payload)
            self.assertFalse(list(p.glob("verified-input-*")))

    def test_fetch_changed_revision_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);recipe=p/"recipe.json";destination=p/"input"
            recipe.write_text(json.dumps({"sources":[{"id":"fixture","url":"https://example.org/fixed","bytes":3,"sha256":cli.hashlib.sha256(b"old").hexdigest()}]}))
            response=io.BytesIO(b"new");response.geturl=lambda:"https://example.org/fixed"
            with patch("sys.argv",["fetch_verified.py","--recipe",str(recipe),"--id","fixture","--destination",str(destination)]),patch.object(fetch.urllib.request,"urlopen",return_value=response):
                with self.assertRaisesRegex(ValueError,"differ from fixed"):fetch.main()
            self.assertFalse(destination.exists())
            self.assertFalse(list(p.glob("verified-input-*")))


if __name__=="__main__":unittest.main()
