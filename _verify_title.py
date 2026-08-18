# Временная проверка фикса: после завершения прегенерации миниатюр заголовок
# должен вернуться на "Pixel", даже если язык интерфейса сменился во время работы.
import asyncio
import sys
import types

# Подменяем тяжёлые ML-модули (torch/faiss/tensorflow/...) и сам пакет core
# заглушками, чтобы импорт app_flet был лёгким и не падал в этой среде.
for _name in (
    "torch", "faiss", "tensorflow", "hdbscan", "transformers",
    "sentence_transformers", "numpy",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))


def _fake_core_package():
    pkg = types.ModuleType("core")
    pkg.__path__ = []
    for sub in ("database", "scanner", "dedup", "embedder", "clustererhdb", "search", "thumbnail_cache"):
        mod = types.ModuleType(f"core.{sub}")
        mod.__dict__.setdefault("run", lambda *a, **k: None)
        mod.__dict__.setdefault("load_clusters", lambda *a, **k: {})
        mod.__dict__.setdefault("get_thumbnail", lambda *a, **k: b"fake")
        mod.__dict__.setdefault("get_thumbnails_for_paths", lambda *a, **k: {})
        setattr(pkg, sub, mod)
        sys.modules[f"core.{sub}"] = mod
    return pkg


sys.modules.setdefault("core", _fake_core_package())

sys.path.insert(0, r"c:\Users\Yarex\Documents\ss6")

import app_flet as m  # noqa: E402
from i18n import set_language  # noqa: E402


class FakePage:
    def __init__(self):
        self.title = None
        self.session = {}
        self.updates = 0

    def update(self):
        self.updates += 1


_orig_sleep = asyncio.sleep


async def _fast_sleep(duration):
    await _orig_sleep(0)


m.asyncio.sleep = _fast_sleep


def make_app():
    cls = m.ImageDedupApp
    app = object.__new__(cls)
    app.page = FakePage()
    app._win_progress_key = None
    app._win_progress_stage = None
    app._win_progress_current = 0
    app._win_progress_total = 0
    app._prewarm_title_owner = False
    app.clusters = {
        -2: [f"/img/{i}.jpg" for i in range(3)],
        0: [f"/img/{i}.jpg" for i in range(3, 6)],
    }

    async def fake_gen(paths, size=150, max_workers=12):
        # В последнем batch переключаем язык интерфейса: это и есть сценарий,
        # при котором старый код (сравнение tr("stage.thumbs")) ломал сброс.
        if "/img/5.jpg" in paths:
            set_language("en")
        await asyncio.sleep(0)
        return {p: b"fake" for p in paths}

    app._generate_thumbnails_parallel = fake_gen
    return app


async def main():
    set_language("ru")
    ok = []

    # 1. Обычный сценарий без смены языка
    app = make_app()
    await app._prewarm_thumbnails()
    ok.append((app.page.title == "Pixel") and (app._win_progress_key is None))
    print(f"[1] Без смены языка: title={app.page.title!r} key={app._win_progress_key!r}")

    # 2. Смена языка в ходе генерации (бывший баг)
    app = make_app()
    await app._prewarm_thumbnails()
    ok.append((app.page.title == "Pixel") and (app._win_progress_key is None))
    print(f"[2] Со сменой языка:   title={app.page.title!r} key={app._win_progress_key!r}")

    # 3. Переключение языка в момент установки заголовка
    app = make_app()
    orig_set = app._set_window_progress

    def _watch(label, current=0, total=0, key=None):
        orig_set(label, current, total, key=key)
        if key == "prewarm":
            set_language("ru")
    app._set_window_progress = _watch
    await app._prewarm_thumbnails()
    ok.append((app.page.title == "Pixel") and (app._win_progress_key is None))
    print(f"[3] Смена языка в момент установки: title={app.page.title!r} key={app._win_progress_key!r}")

    print(f"\nИтог: {sum(ok)}/{len(ok)}")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    m.asyncio.run(main())