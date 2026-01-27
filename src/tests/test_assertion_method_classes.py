import tempfile
import unittest
from pathlib import Path

import utils.assertion_method_classes as amc
import utils.dafny_read_assertions_xml as readxml
from utils.dataset_class import Dataset

SAMPLE_SOURCE = """\
// VERIFY USING DAFNY:
// /Applications/dafny/dafny /Users/apple/GaussianDP/Dafny/gaussian.dfy
method gaussian (size:int, q: array<real>, q_hat: array<real>) returns (out: array<real>)
requires q_hat.Length==size
requires q.Length==size
requires size > 0
requires arraySquaredSum(q_hat[..]) <= 1.0
{
 var i : int := 0;
  var alpha : real := arraySquaredSum(q_hat[..1]);
 var eta: real := 0.0;
 var eta_hat: real := 0.0;
 out := new real[size];
 while (i <size)
 invariant 0 < i <= size ==> alpha <= arraySquaredSum(q_hat[..i])
 invariant i<=size
 {
  eta := *;
  eta_hat := - q_hat[i];
  alpha := arraySquaredSum(q_hat[..i+1]);
  assert (q_hat[i] + eta_hat ==0.0);
  out[i] := q[i] + eta;
  i := i+1;
 }
 assert i==size;
 assert alpha <= arraySquaredSum(q_hat[..size]);
 assert q_hat[..size] == q_hat[..];
 assert alpha <= arraySquaredSum(q_hat[..]);
 assert alpha <= 1.0;
}


function arraySquaredSum(a: seq<real>): real
requires |a| > 0
{
  if |a| == 1 then 
    a[0]*a[0]
  else 
    (a[0]*a[0]) + arraySquaredSum(a[1..])
}


"""

ASSERT_XML = """<program>
  <name>tmpmc4ds1oe.dfy</name>
  <Method>
    <name> _module._default.gaussian </name>
    <start_pos> 95 </start_pos>
    <end_pos> 873 </end_pos>
    <assertion>
      <type>Regular_assertion </type>
      <start_pos>630 </start_pos>
      <end_pos>663 </end_pos>
    </assertion>
    <assertion>
      <type>Regular_assertion </type>
      <start_pos>705 </start_pos>
      <end_pos>719 </end_pos>
    </assertion>
    <assertion>
      <type>Regular_assertion </type>
      <start_pos>722 </start_pos>
      <end_pos>768 </end_pos>
    </assertion>
    <assertion>
      <type>Regular_assertion </type>
      <start_pos>771 </start_pos>
      <end_pos>804 </end_pos>
    </assertion>
    <assertion>
      <type>Regular_assertion </type>
      <start_pos>807 </start_pos>
      <end_pos>849 </end_pos>
    </assertion>
    <assertion>
      <type>Regular_assertion </type>
      <start_pos>852 </start_pos>
      <end_pos>871 </end_pos>
    </assertion>
  </Method>
  <Function>
    <name> _module._default.arraySquaredSum </name>
    <start_pos> 877 </start_pos>
    <end_pos> 1025 </end_pos>
  </Function>
</program>
"""

INFO_XML="""<?xml version="1.0" ?>
<method>
  <name>_module._default.gaussian</name>
  <start_pos>95</start_pos>
  <end_pos>873</end_pos>
  <assertion_group>
    <id>3</id>
    <number_assertions>2</number_assertions>
    <assertion>
      <type>Regular_assertion</type>
      <start_pos>722</start_pos>
      <end_pos>768</end_pos>
    </assertion>
    <assertion>
      <type>Regular_assertion</type>
      <start_pos>771</start_pos>
      <end_pos>804</end_pos>
    </assertion>
  </assertion_group>
</method>
"""

def write_file(path: Path, content: str) -> Path:
    path.write_bytes(content.encode("utf-8"))
    return path

class TestAssertMethodClass(unittest.TestCase):
    def setUp(self):
        # make a temporary directory for files and datasets
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_substitute_a_given_pos_by_text_basic(self):
        b = b"0123456789"
        new_bytes, new_str = amc.substitute_a_given_pos_by_text(2, 5, "XX", b)
        self.assertEqual(new_bytes, b"01XX6789")
        self.assertEqual(new_str, "01XX6789")

    def test_fileinfo_and_filesegment_reading(self):
        src = self.tmp_path / "original_program.dfy"
        content = "line1\nASSERT_A\nline3\nASSERT_B\n"
        write_file(src, content)
        file_info = amc.FileInfo(src)
        self.assertEqual(file_info.file_path, src)
        self.assertEqual(content, file_info.file_text)
        self.assertEqual(file_info.end_pos, len(file_info.file_bytes) + 1)

    def test_assertion_and_method_linking_and_group_helpers(self):
        src = self.tmp_path / "original_program.dfy"
        write_file(src, SAMPLE_SOURCE)

        # XML in method (info.xml) format
        info_xml = INFO_XML

        file_obj = readxml.extract_assertion(info_xml, src)
        self.assertEqual(len(file_obj.methods), 1)
        method = file_obj.methods[0]
        self.assertEqual(method.method_name, "_module._default.gaussian")
        self.assertEqual(len(method.assertion_groups), 1)
        ag = method.assertion_groups[0]
        self.assertEqual(len(ag), 2)

        self.assertIs(amc.get_method_from_assertion_group(ag), method)
        self.assertIs(amc.get_file_from_assertion_group(ag), file_obj)
        sid = amc.get_assertion_group_string_id(ag)
        self.assertIn("method_start_95", sid)
        self.assertIn("_as_start_722_end_768", sid)
        self.assertIn("_as_start_771_end_804", sid)

    def test_get_method_with_assertion_group_changed_remove_and_replace(self):
        content = "header\n" \
                  "code line 1\n" \
                  "ASSERT_A\n" \
                  "middle\n" \
                  "ASSERT_B\n" \
                  "footer\n"
        src = self.tmp_path / "original_program.dfy"
        write_file(src, content)
        file_info = amc.FileInfo(src)
        method = amc.MethodInfo(0, len(content) - 1, "m", file_info)

        a_pos = file_info.file_text.index("ASSERT_A")
        a_end = a_pos + len("ASSERT_A") - 1
        b_pos = file_info.file_text.index("ASSERT_B")
        b_end = b_pos + len("ASSERT_B") - 1

        a = amc.AssertionInfo(a_pos, a_end, "Regular_assertion", method)
        b = amc.AssertionInfo(b_pos, b_end, "Regular_assertion", method)

        group = [a, b]
        method.add_assertion_group(group)

        replaced = method.get_method_with_assertion_group_changed(group, remove_empty_lines=True, change_text="/*REMOVED*/")
        self.assertNotIn("ASSERT_A", replaced)
        self.assertNotIn("ASSERT_B", replaced)
        self.assertIn("/*REMOVED*/", replaced)

        replaced_no_strip = method.get_method_with_assertion_group_changed(group, remove_empty_lines=False, change_text="X")
        self.assertNotIn("ASSERT_A", replaced_no_strip)
        self.assertNotIn("ASSERT_B", replaced_no_strip)
        self.assertIn("X", replaced_no_strip)

    def test_dataset_factories_and_get_all_assertion_groups(self):
        # dataset_all layout
        base = self.tmp_path / "dataset_all"
        base.mkdir()
        foo = base / "foo"
        foo.mkdir()
        src = foo / "original_program.dfy"
        write_file(src, SAMPLE_SOURCE)
        assert_xml = foo / "assert.xml"
        assert_xml.write_text(ASSERT_XML, encoding="utf-8")

        # dataset_assertion_groups layout
        base2 = self.tmp_path / "dataset_groups"
        base2.mkdir()
        bar = base2 / "bar"
        bar.mkdir()
        (bar / "original_program.dfy").write_text(SAMPLE_SOURCE, encoding="utf-8")
        ms = bar / "method_start_1"
        ms.mkdir()
        info = ms / "info.xml"
        info.write_text(INFO_XML, encoding="utf-8")

        ds_all = Dataset.from_dataset_all(base)
        self.assertIsInstance(ds_all, Dataset)
        self.assertEqual(len(ds_all.files), 1)
        self.assertEqual(len(ds_all.files[0].methods), 2)

        ds_groups = Dataset.from_dataset_assertion_groups(base2)
        self.assertIsInstance(ds_groups, Dataset)
        self.assertEqual(len(ds_groups.files), 1)
        groups = ds_groups.get_all_assertion_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)


if __name__ == "__main__":
    unittest.main()