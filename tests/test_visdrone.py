import json
import sys

import pytest
from PIL import Image

from edgevision.cli import main
from edgevision.data_audit import audit_image
from edgevision.visdrone import read_annotations


def test_native_classes_ignore_flags_and_geometry_survive_parsing(tmp_path):
    labels = tmp_path / "scene.txt"
    labels.write_text(
        "100,80,60,40,1,4,0,1,\n"  # trailing comma is accepted
        "0,0,20,20,0,0,-1,-1\n"
        "50,50,10,10,1,11,0,0\n"
        "60,60,10,10,0,4,0,2\n"
    )
    car, region, other, ignored = read_annotations(labels)
    assert car.xyxy == (100, 80, 160, 120)
    assert car.class_id == 3 and car.category_id == 4
    assert car.occlusion == 1 and car.role == "target"
    assert region.class_id is None and region.role == "ignored_region"
    assert region.truncation == -1
    assert other.class_id is None and other.role == "other"
    assert ignored.class_id == 3 and ignored.role == "ignored_object"
    assert ignored.score == 0 and ignored.occlusion == 2


@pytest.mark.parametrize(
    "row",
    [
        "0,0,10,10,1,4,0",  # missing field
        "0,0,10,10,1,4,0,0,8",  # wrong task/extra column
        "0,0,0,10,1,4,0,0",  # zero area
        "0,0,10,10,0.7,4,0,0",  # prediction rather than ground truth
        "0,0,10,10,2,4,0,0",  # invalid flag
        "0,0,10,10,1,12,0,0",  # unknown category
        "0,0,10,10,1,4,0,3",  # unknown occlusion
        "nan,0,10,10,1,4,0,0",  # invalid numeric value
    ],
)
def test_bad_rows_identify_source_line(tmp_path, row):
    labels = tmp_path / "bad.txt"
    labels.write_text("\n" + row + "\n")
    with pytest.raises(ValueError, match=r"bad.txt:2:"):
        read_annotations(labels)


def test_empty_file_is_valid_but_missing_file_is_not(tmp_path):
    labels = tmp_path / "empty.txt"
    labels.touch()
    assert read_annotations(labels) == ()
    with pytest.raises(FileNotFoundError):
        read_annotations(tmp_path / "missing.txt")


def test_audit_preserves_issues_and_refuses_to_overwrite(tmp_path):
    image = tmp_path / "scene.png"
    Image.new("RGB", (64, 48)).save(image)
    labels = tmp_path / "scene.txt"
    original = "50,30,30,30,1,4,0,0\n50,30,30,30,1,4,0,0\n"
    labels.write_text(original)
    output = tmp_path / "audit"
    report = audit_image(image, labels, "synthetic", output)
    assert report["status"] == "needs_review"
    assert report["issues"] == [
        {"line": 1, "kind": "out_of_bounds"},
        {"line": 2, "kind": "out_of_bounds"},
        {"line": 2, "kind": "duplicate_row"},
    ]
    records = json.loads((output / "annotations.json").read_text())["annotations"]
    assert records[0]["xyxy"] == [50, 30, 80, 60]  # No silent clipping.
    assert labels.read_text() == original
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["target_class_map"]["3"] == "car"
    assert len(manifest["image_sha256"]) == 64
    with pytest.raises(FileExistsError):
        audit_image(image, labels, "synthetic", output)


@pytest.mark.parametrize("row,exit_code", [("", None), ("60,0,10,10,1,4,0,0\n", 1)])
def test_cli_audit_reports_empty_image_or_review_needed(
    tmp_path, monkeypatch, capsys, row, exit_code
):
    image = tmp_path / "scene.png"
    Image.new("RGB", (64, 48)).save(image)
    labels = tmp_path / "scene.txt"
    labels.write_text(row)
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edgevision",
            "audit-image",
            "--image",
            str(image),
            "--annotations",
            str(labels),
            "--split",
            "synthetic",
            "--output-dir",
            str(output),
        ],
    )
    if exit_code:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == exit_code
    else:
        main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == ("needs_review" if row else "ok")
    assert (output / "overlay.png").is_file()
