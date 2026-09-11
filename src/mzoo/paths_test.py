from datetime import date

from mzoo.paths import run_dir


def test_first_call_creates_layout(tmp_path):
    out = run_dir("myproj", "exp1", "run1", root=tmp_path)
    today = date.today().strftime("%y%m%d")
    assert out.is_dir()
    assert out.parent.parent.name == "0001-myproj"
    assert out.parent.name == f"{today}-0001-exp1"
    assert out.name == "0001-run1"


def test_same_proj_reused_new_proj_incremented(tmp_path):
    out1 = run_dir("myproj", "exp1", "run1", root=tmp_path)
    out2 = run_dir("myproj", "exp2", "run1", root=tmp_path)
    out3 = run_dir("other", "exp1", "run1", root=tmp_path)
    assert out1.parent.parent == out2.parent.parent
    assert out3.parent.parent.name == "0002-other"


def test_same_exp_name_gets_new_number(tmp_path):
    out1 = run_dir("myproj", "exp1", "run1", root=tmp_path)
    out2 = run_dir("myproj", "exp1", "run2", root=tmp_path)
    assert out1.parent != out2.parent
    assert out2.parent.name.endswith("-0002-exp1")


def test_existing_exp_dir_name_reused(tmp_path):
    out1 = run_dir("myproj", "exp1", "run1", root=tmp_path)
    exp_dir_name = out1.parent.name
    out2 = run_dir("myproj", exp_dir_name, "run2", root=tmp_path)
    assert out2.parent == out1.parent
    assert out2.name == "0002-run2"
