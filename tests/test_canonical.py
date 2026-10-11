from __future__ import annotations

import itertools

import pytest

from cv_pipeline.canonical import canonical_path_key, looks_like_copy


def pick(*paths: str) -> str:
    return min(paths, key=canonical_path_key)


@pytest.mark.parametrize(
    "path",
    [
        "images/duplicates/scene.png",
        "images/Duplicate/scene.png",
        "dups/scene.png",
        "images/backup/2024/scene.png",
        "images/scene copy.jpg",
        "images/renamed_copy_of_scene.png",
        "images/scene-COPY-2.jpg",
        "images/scene (1).jpg",
        "images/copies/scene.jpg",
    ],
)
def test_copy_like_paths(path: str) -> None:
    assert looks_like_copy(path)


@pytest.mark.parametrize(
    "path",
    [
        "images/scene.png",
        "images/site_a/panel_3.jpg",
        "images/copyright_notice_panel.jpg",  # 'copy' only as a whole word
        "images/duplex_outlet.jpg",
        "videos/walkthrough.mp4",
        "images/scene_1.jpg",
    ],
)
def test_ordinary_paths(path: str) -> None:
    assert not looks_like_copy(path)


def test_duplicates_folder_never_wins_even_if_it_sorts_first() -> None:
    assert "images/duplicates/a.png" < "images/scene_001.png"  # the old failure mode
    assert pick("images/duplicates/a.png", "images/scene_001.png") == "images/scene_001.png"


def test_copy_marker_beats_depth() -> None:
    assert pick("copy.png", "site/a/b/photo.png") == "site/a/b/photo.png"


def test_shallower_path_preferred() -> None:
    assert pick("images/nested/deeper/x.png", "images/x.png") == "images/x.png"


def test_normalised_path_tiebreak_is_case_insensitive() -> None:
    assert pick("images/B.png", "images/a.png") == "images/a.png"


def test_unicode_normalisation() -> None:
    nfc, nfd = "images/café.png", "images/café.png"
    assert canonical_path_key(nfc)[2] == canonical_path_key(nfd)[2]
    assert canonical_path_key(nfc) != canonical_path_key(nfd)  # still a total order


def test_choice_is_independent_of_input_order() -> None:
    paths = [
        "images/duplicates/scene.png",
        "images/scene (1).png",
        "images/archive/scene.png",
        "images/Scene.png",
        "images/scene.png",
    ]
    winners = {min(perm, key=canonical_path_key) for perm in itertools.permutations(paths)}
    # NFC+casefold ties between Scene/scene; the raw path breaks it ('S' < 's')
    assert winners == {"images/Scene.png"}
