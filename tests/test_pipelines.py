"""Checks that need no GPU or ComfyUI: screenplay parsing for both pipelines, sprite keying/patching maths, lip-sync.
Run from the repo root:  venv\\Scripts\\python -m unittest discover tests
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

REPO = Path(__file__).parent.parent
sys.path[:0] = [str(REPO / "cartoon"), str(REPO / "stickman"), str(REPO / "shared")]
import director as cartoon_director  # noqa: E402  (cartoon/ comes first on the path)
import make_cartoon  # noqa: E402
import sprites  # noqa: E402

_spec = importlib.util.spec_from_file_location("stickman_director", REPO / "stickman" / "director.py")
stickman_director = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stickman_director)

CAST = json.loads((REPO / "cartoon" / "cast.json").read_text(encoding="utf-8"))


def direct(text):
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "screenplay.txt"
        src.write_text(text, encoding="utf-8")
        return cartoon_director.direct(src, Path(d) / "script.json", CAST)


HEADER = "TITLE: Test\nNARRATOR: nia\n@room: An empty room\n"


class CartoonDirector(unittest.TestCase):
    def test_narration_lines_are_shots_not_the_narrator_header(self):
        s = direct(HEADER + "== room | nia (calm)\nNARRATOR: First line.\nNARRATOR: Second line.\n")
        self.assertEqual(s["narrator"], "nia")
        self.assertEqual([sh["text"] for sh in s["shots"]], ["First line.", "Second line."])

    def test_speaker_lip_syncs_and_listener_does_not(self):
        s = direct(HEADER + "== room | nia, omar (calm)\nOMAR: Hello there. [smile]\n")
        talk = {a["id"]: a["talk"] for a in s["shots"][0]["actors"]}
        self.assertEqual(talk, {"nia": False, "omar": True})
        self.assertIn(["stand", "smile", True], s["needs"]["omar"])

    def test_characters_face_each_other(self):
        s = direct(HEADER + "== room | nia, omar (calm)\nNIA: Hi.\n")
        flips = {a["id"]: a["flip"] for a in s["shots"][0]["actors"]}
        self.assertEqual(flips, {"nia": False, "omar": True})

    def test_gesture_lasts_one_line_but_held_prop_persists(self):
        s = direct(HEADER + "POSE: mug = Holding a mug\n== room | nia (calm)\n"
                   "NIA: Look. [point angry]\nNIA: Again.\nNIA: Coffee. [mug]\nNIA: Still coffee.\n")
        poses = [sh["actors"][0]["pose"] for sh in s["shots"]]
        self.assertEqual(poses, ["point", "stand", "mug", "mug"])
        self.assertEqual(s["shots"][1]["actors"][0]["expr"], "angry")

    def test_expression_change_mid_line(self):
        s = direct(HEADER + "== room | nia (calm)\nNIA: Oh no. [smile -> shocked]\n")
        a = s["shots"][0]["actors"][0]
        self.assertEqual((a["expr"], a["expr2"]), ("smile", "shocked"))

    def test_later_character_enters_with_a_walk_pose(self):
        s = direct(HEADER + "== room | nia, leo@later (tense)\nNARRATOR: Then he came. [leo: enter right]\nLEO: Hi.\n")
        self.assertEqual([a["id"] for a in s["shots"][0]["actors"]], ["nia", "leo"])
        self.assertEqual(s["shots"][0]["actors"][1]["move"], {"kind": "enter", "side": "right"})
        self.assertEqual(s["shots"][0]["cam"]["type"], "wide")
        self.assertIn(["walk", "neutral", False], s["needs"]["leo"])

    def test_every_character_gets_the_standing_drawing_that_sets_scale(self):
        s = direct(HEADER + "POSE: mug = Holding a mug\n== room | nia (calm)\nNIA: Hi. [mug]\n")
        self.assertIn(["stand", "neutral", False], s["needs"]["nia"])

    def test_chapter_lines_mark_the_next_shot_and_comments_are_ignored(self):
        s = direct(HEADER + "# a note to self\n## The Key\n== room | nia (calm)\nNIA: Hi. # trailing note\n")
        self.assertEqual(s["shots"][0]["chapter"], "The Key")
        self.assertEqual(s["shots"][0]["text"], "Hi.")

    def test_cards_take_the_next_location(self):
        s = direct(HEADER + "@hall: A hall\n-- title 3 {pop: TITLE}\n== hall | nia (calm)\nNIA: Hi.\n")
        self.assertEqual(s["shots"][0]["plate"], "hall")

    def test_unknown_direction_word_names_the_line(self):
        with self.assertRaises(SystemExit) as e:
            direct(HEADER + "== room | nia (calm)\nNIA: Hi. [dancing]\n")
        self.assertIn("line 5", str(e.exception))

    def test_committed_cartoon_screenplays_parse(self):
        for src in (REPO / "cartoon" / "episodes").glob("*/screenplay.txt"):
            with self.subTest(src.parent.name), tempfile.TemporaryDirectory() as d:
                s = cartoon_director.direct(src, Path(d) / "script.json", CAST)
                self.assertGreater(len(s["shots"]), 0)
                for sh in s["shots"]:
                    for a in sh.get("actors", []):
                        self.assertIn(a["pose"], set(sprites.POSES) | set(s["poses"]))


class StickmanDirector(unittest.TestCase):
    def test_committed_stickman_screenplays_parse(self):
        cast = json.loads((REPO / "stickman" / "cast.json").read_text(encoding="utf-8"))
        for src in (REPO / "stickman" / "episodes").glob("*/screenplay.txt"):
            with self.subTest(src.parent.name), tempfile.TemporaryDirectory() as d:
                out = stickman_director.direct(src, Path(d) / "script.json", cast)
                self.assertGreater(len(out["shots"]), 50)


def keyed_disc(key_rgb, vignette):
    """A dark-outlined disc with pale pink 'eye whites' on a key background that darkens toward the corners."""
    h, w = 160, 120
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2) / np.hypot(h / 2, w / 2)
    img = np.array(key_rgb, np.float32) * (1 - vignette * r[..., None] ** 2)
    disc = np.hypot(yy - 80, xx - 60)
    img[disc < 34] = (20, 20, 20)
    img[disc < 30] = (180, 120, 90)
    img[(np.abs(yy - 70) < 3) & (np.abs(xx - 52) < 4)] = (245, 180, 240)
    return img


class Keying(unittest.TestCase):
    def test_magenta_key_with_vignette_and_pink_eye_whites(self):
        rgb, alpha = sprites._alpha(keyed_disc((235, 8, 135), 0.3), "magenta")
        self.assertLess(alpha[2, 2], 0.05)        # dark corner of the background is still keyed out
        self.assertEqual(alpha[80, 60], 1.0)      # skin
        self.assertEqual(alpha[70, 52], 1.0)      # key-tinted eye white stays opaque
        np.testing.assert_allclose(rgb[70, 52], (245, 180, 240), atol=1)

    def test_green_key(self):
        _, alpha = sprites._alpha(keyed_disc((12, 240, 6), 0.25), "green")
        self.assertLess(alpha[-3, -3], 0.05)
        self.assertEqual(alpha[80, 60], 1.0)


class GhostCheck(unittest.TestCase):
    def test_half_blended_patch_is_flagged_and_clean_figure_is_not(self):
        clean = keyed_disc((12, 240, 6), 0.0)
        ghost = clean.copy()
        ghost[20:50, 10:40] = 0.5 * np.array((12, 240, 6)) + 0.5 * np.array((230, 160, 30))
        with tempfile.TemporaryDirectory() as d:
            for name, img in (("clean", clean), ("ghost", ghost)):
                Image.fromarray(img.astype(np.uint8)).save(Path(d) / f"{name}.png")
            self.assertEqual(sprites._ghost_area(Path(d) / "clean.png", "green"), 0)
            self.assertGreater(sprites._ghost_area(Path(d) / "ghost.png", "green"), 100)


class FacePatch(unittest.TestCase):
    def test_only_dense_change_inside_the_head_box_is_kept(self):
        base = np.full((200, 100, 3), 100, np.float32)
        var = base.copy()
        var[20:40, 30:60] = 200          # the mouth change, inside the head box
        var[150:152, :] = 180            # a redrawn line on the body, outside the box
        out, area = sprites._patch(base, var, (0, 60, 0, 100))
        self.assertGreater(out[30, 45].mean(), 190)
        np.testing.assert_allclose(out[151], base[151])
        self.assertGreater(area, 0.01)

    def test_sparse_line_jitter_is_ignored(self):
        base = np.full((200, 100, 3), 100, np.float32)
        var = base.copy()
        var[30, :] = 200                 # a single 1-px line redrawn: not a face change
        _, area = sprites._patch(base, var, (0, 60, 0, 100))
        self.assertEqual(area, 0.0)


class LipSync(unittest.TestCase):
    def test_mouth_closed_in_silence_and_flaps_on_speech(self):
        sr = 24000
        t = np.arange(sr * 2) / sr
        x = np.where(t < 1.0, 0.0, 0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            sf.write(Path(d) / "line.wav", x, sr)
            mouth = make_cartoon._envelope(Path(d) / "line.wav")
        fps = make_cartoon.FPS
        self.assertFalse(any(mouth[:fps - 1]))
        speech = mouth[fps + 1:]
        self.assertTrue(any(speech))
        longest = max(len(run) for run in "".join("1" if m else "0" for m in speech).split("0"))
        self.assertLessEqual(longest, 4)        # long vowels still flap


class Redraw(unittest.TestCase):
    def test_redraw_forgets_one_expression_and_bumps_its_seed(self):
        with tempfile.TemporaryDirectory() as d:
            old, sprites.CHARS = sprites.CHARS, Path(d)
            try:
                c = Path(d) / "nia"
                (c / "raw").mkdir(parents=True)
                names = ["point_neutral", "point_angry", "point_angry_talk", "stand_neutral"]
                for n in names:
                    (c / f"{n}.png").write_bytes(b"x")
                    (c / "raw" / f"{n}.png").write_bytes(b"x")
                (c / "sprites.json").write_text(json.dumps({n: {} for n in names}))
                gone = sprites.redraw("nia", "point_angry")
                self.assertEqual(sorted(gone), ["point_angry", "point_angry_talk"])
                self.assertTrue((c / "point_neutral.png").exists())
                self.assertEqual(json.loads((c / "redraws.json").read_text()), {"point_angry": 1})
                gone = sprites.redraw("nia", "point")
                self.assertEqual(gone, ["point_neutral"])
                self.assertFalse((c / "raw" / "point_neutral.png").exists())
                with self.assertRaises(SystemExit):
                    sprites.redraw("nia", "stand")
            finally:
                sprites.CHARS = old


class SpriteIndex(unittest.TestCase):
    def test_saving_keeps_entries_another_run_wrote_meanwhile(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sprites.json"
            path.write_text(json.dumps({"walk_worried": {"top": 1}}))   # written by another run after this one started
            meta = sprites._merge_meta(path, {"point_angry": {"top": 2}})
            self.assertEqual(set(meta), {"walk_worried", "point_angry"})
            self.assertEqual(json.loads(path.read_text()), meta)


class TalkChunks(unittest.TestCase):
    def test_chunks_cover_the_clip_without_generating_much_extra(self):
        import comfy_client as cc
        self.assertEqual(cc.talk_chunks(56), [57])                 # a 2.2 s line: one short chunk
        self.assertEqual(cc.talk_chunks(91), [81, 21])             # 3.6 s: a full chunk, then only what is left
        for frames in range(10, 400, 7):
            sizes = cc.talk_chunks(frames)
            made = sizes[0] + sum(n - cc.TALK_MOTION for n in sizes[1:])
            self.assertGreaterEqual(made, frames)
            self.assertLess(made - frames, 24)
            self.assertTrue(all((n - 1) % 4 == 0 and n <= cc.TALK_CHUNK for n in sizes))


class ComfyNotRunning(unittest.TestCase):
    def test_clear_message_instead_of_a_crash(self):
        import comfy_client
        old, comfy_client.COMFY_URL = comfy_client.COMFY_URL, "http://127.0.0.1:9"
        try:
            with self.assertRaises(SystemExit) as e:
                comfy_client.queue({})
        finally:
            comfy_client.COMFY_URL = old
        self.assertIn("start_comfyui.bat", str(e.exception))


class MusicBed(unittest.TestCase):
    def test_short_track_loops_to_length_in_stereo_at_a_fixed_level(self):
        import music
        import sfx
        sr = 44100
        t = np.arange(sr * 5) / sr
        tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            sf.write(Path(d) / "calm_1.flac", tone, sr)
            old, music.LIB = music.LIB, Path(d)
            try:
                bed = music.bed("calm", 12.0, 0)
            finally:
                music.LIB = old
        self.assertEqual(bed.shape, (12 * sfx.SR, 2))
        self.assertAlmostEqual(float(np.sqrt(np.mean(bed[: 3 * sfx.SR] ** 2))), music.LEVEL, delta=0.01)


class Animate(unittest.TestCase):
    def setUp(self):
        import animate
        self.animate = animate

    def test_prompt_names_who_talks_who_listens_and_how_they_feel(self):
        script = {"poses": {"mug": "Holding a white coffee mug in front of {p} chest"}, "plates": {}}
        shot = {"cam": {"type": "two"}, "actors": [
            {"id": "leo", "pose": "mug", "expr": "smile", "talk": True},
            {"id": "nia", "pose": "stand", "expr": "neutral", "expr2": "shocked", "talk": False}]}
        p = self.animate.prompt(CAST, script, shot, "leo", ["leo", "nia"])
        self.assertIn("grey blazer", p)
        self.assertIn("stands holding a white coffee mug in front of his chest", p)
        self.assertIn("talks expressively", p)
        self.assertIn("then becomes shocked", p)
        self.assertIn("listens and reacts", p)

    def test_prompt_leaves_out_characters_who_are_not_in_the_picture(self):
        shot = {"cam": {"type": "close"}, "actors": [
            {"id": "nia", "pose": "stand", "expr": "worried", "talk": False},
            {"id": "leo", "pose": "stand", "expr": "smile", "talk": False}]}
        p = self.animate.prompt(CAST, {"poses": {}, "plates": {}}, shot, None, ["nia"])
        self.assertIn("hair puff", p)
        self.assertNotIn("blazer", p)
        self.assertIn("Nobody else is in the shot.", p)

    def test_masks_split_the_speaker_from_everyone_else(self):
        boxes = {"leo": (1100, 300, 1350, 1060), "nia": (560, 340, 780, 1060)}
        with tempfile.TemporaryDirectory() as d:
            speaker, others = self.animate.masks(boxes, "leo", Path(d), 3)
            sp, ot = np.asarray(Image.open(speaker))[..., 0], np.asarray(Image.open(others))[..., 0]
        x_leo, x_nia, y = int(1225 * 848 / 1920), int(670 * 848 / 1920), int(700 * 480 / 1080)
        self.assertEqual((sp[y, x_leo], sp[y, x_nia]), (255, 0))
        self.assertEqual((ot[y, x_leo], ot[y, x_nia]), (0, 255))

    def test_short_shots_get_short_clips_and_long_ones_one_full_clip(self):
        self.assertEqual(self.animate.i2v_frames(2.2), 37)       # covers 2.2 s at 16 fps, 4n+1
        self.assertEqual(self.animate.i2v_frames(0.5), 17)
        self.assertEqual(self.animate.i2v_frames(7.3), 81)
        for d in (0.5, 2.2, 3.0, 4.9, 7.3):
            n = self.animate.i2v_frames(d)
            self.assertEqual((n - 1) % 4, 0)
            self.assertGreaterEqual((n - 1) / 16, min(d, 5.0) - 0.07)

    def test_talk_audio_places_the_line_after_the_lead_and_pads_the_shot(self):
        sr = 24000
        with tempfile.TemporaryDirectory() as d:
            sf.write(Path(d) / "line.wav", np.ones(sr, np.float32) * 0.5, sr)
            out = self.animate.talk_audio(Path(d) / "line.wav", 0.15, 3.5, Path(d) / "talk.wav")
            x, _ = sf.read(out)
        self.assertEqual(len(x), int(3.5 * sr) + 1)
        self.assertEqual(x[int(0.1 * sr)], 0.0)
        self.assertAlmostEqual(x[int(0.5 * sr)], 0.5, places=3)
        self.assertEqual(x[int(1.3 * sr)], 0.0)


class Text(unittest.TestCase):
    def test_title_wraps_into_balanced_lines(self):
        img = make_cartoon._text_image("MY PERFECT NEIGHBOR HAD A KEY", 150, 1500)
        lines = round(img.height / (150 * 1.08))
        self.assertEqual(lines, 2)


if __name__ == "__main__":
    unittest.main()
