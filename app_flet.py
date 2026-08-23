import flet as ft
import asyncio
import os
import shutil
import sys
import time
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image as PILImage
import concurrent.futures

import config

# Единая настройка окружения (OMP/предупреждения) ДО импорта torch/faiss,
# чтобы фоновый импорт/lazy-загрузка SigLIP не создавали лишние OMP-потоки
# и не падали с "OMP: Error #179" на Apple Silicon.
config.setup_environment()

from i18n import LANG, tr, set_language, is_all_disks
from core import database, scanner, dedup, embedder, clustererhdb, search, thumbnail_cache


def _format_size(size_bytes: int) -> str:
    """Форматирует размер в байтах в читаемый вид (КБ/МБ/ГБ)."""
    try:
        size_bytes = int(size_bytes)
    except (TypeError, ValueError):
        return "0 " + tr("unit.B")
    if size_bytes < 1024:
        return f"{size_bytes} {tr('unit.B')}"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} {tr('unit.KB')}"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} {tr('unit.MB')}"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} {tr('unit.GB')}"


def _dedupe_scan_paths(paths):
    """Возвращает список корневых путей без дублей и пересечений.

    Убирает пути, которые являются дубликатами или вложены в другой уже
    включённый путь (например, на macOS корень '/' уже проходит по всем
    точкам монтирования /Volumes/*, поэтому их не нужно сканировать и
    считать повторно). Это исключает двойной учёт одного и того же диска
    в общем счётчике прогресса 'Сканирование... X/Y'.
    """
    tmp = []
    for p in paths:
        try:
            n = os.path.normpath(p)
            if sys.platform == "win32":
                n = os.path.normcase(n)
        except Exception:
            n = p
        if not n:
            continue
        tmp.append((p, n))

    keep = []
    for p, n in tmp:
        # Новый путь — наследник уже сохранённого корня (уже покрыт) → пропускаем
        if any(n == kn or n.startswith(kn.rstrip(os.sep) + os.sep) for _, kn in keep):
            continue
        # Новый путь шире сохранённого → выкидываем сохранённых наследников
        keep = [(kp, kn) for kp, kn in keep
                if not (kn == n or kn.startswith(n.rstrip(os.sep) + os.sep))]
        keep.append((p, n))

    return [p for p, _ in keep]



class ImageDedupApp:
    def __init__(self, page: ft.Page):
        self.page = page
        
        # 1. Настраиваем окно, но пока держим его скрытым, чтобы оно не
        #    мелькало в левом верхнем углу. Размер выставляем сразу.
        self.page.title = tr("app.title")
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.window.width = 1150
        self.page.window.height = 700

        # 2. Показываем окно только после готовности: ждём клиента,
        #    центрируем и делаем видимым. (page.window.center() — async,
        #    поэтому его нельзя просто вызвать в __init__.)
        async def _open_window():
            try:
                await asyncio.wait_for(
                    self.page.window.wait_until_ready_to_show(), timeout=5
                )
            except Exception:
                pass
            try:
                await asyncio.wait_for(self.page.window.center(), timeout=5)
            except Exception:
                pass
            try:
                self.page.window.visible = True
            except Exception:
                pass
            self.page.update()

        self.page.run_task(_open_window)
        
        self.page.padding = 20
        self.page.spacing = 10
        
        # Состояние приложения
        self.images = []
        self.clusters = {}
        self.cluster_names = {}
        self.search_results = []
        self.model_loaded = False
        self.model_loading = False
        self.scanning = False
        self.scan_task = None
        self._scan_worker_tasks = []
        self._prewarm_task = None
        self._prewarm_title_owner = False
        self.exporting = False
        self.export_task = None
        self._export_stop_requested = False
        self._current_lang = LANG
        self._scan_phase = "idle"

        # Текущий режим: scan / search / export / cluster
        self.current_mode = "scan"

        # Последний прогресс для заголовка окна (нужен при смене языка)
        self._win_progress_stage = None
        self._win_progress_current = 0
        self._win_progress_total = 0
        # Стабильный идентификатор «владельца» заголовка (напр. "prewarm").
        # Не зависит от языка, поэтому сброс заголовка не ломается при
        # переключении языка во время фоновой генерации миниатюр.
        self._win_progress_key = None

        self._WARM_PRIMARY = "#C9A96E"
        self._WARM_ON_PRIMARY = "#1C1610"
        self._WARM_PRIMARY_CONTAINER = "#4A3F2F"
        self._WARM_ON_PRIMARY_CONTAINER = "#F0E6D6"
        self._WARM_SURFACE = "#4A4238"
        self._WARM_ON_SURFACE = "#E8DCC8"
        self._WARM_ON_SURFACE_VARIANT = "#A89880"
        self._WARM_ERROR = "#CF6679"

        
        # Состояние выбора изображений
        self.selected_images = set()
        try:
            saved_selection = database.load_selected_files(scope="global")
            if saved_selection:
                self.selected_images = saved_selection
        except Exception as e:
            print(f"Ошибка загрузки выделенных файлов: {e}")
        
        # Сохраняем путь экспорта между переключениями вкладок
        self._export_dest_folder_path = None
        
        # Текущий контекст галереи (для обновления счётчика и иконки)
        self.current_gallery_paths = []
        self.current_gallery_scope = "overview"
        
        # Кэш для path_to_cluster_map (оптимизация производительности)
        self._cached_path_to_cluster = None
        self._cached_path_to_cluster_scope = None
        # Кэш карты кластеров для активной галереи поиска (чтобы не перестраивать
        # её на каждом скролле в load_more — итерация по всем кластерам дорогая).
        self._gallery_path_to_cluster = None
        

        # Создаём UI
        self.create_layout()
        
        # Загружаем статистику асинхронно, чтобы тяжёлые обращения к файлам
        # (например, к выбранным изображениям на внешнем диске) не блокировали
        # отрисовку окна при запуске.
        self.page.run_task(self.load_stats)
        
        # Показываем первый кластер по умолчанию
        self.page.run_task(self.show_clusters_tab)
        
        # Фоновая загрузка модели
        self.page.run_task(self.preload_model)
        
        # Обработчик изменения размера окна
        self.page.on_resize = self.on_window_resize
        
        # Обработчик закрытия окна — останавливаем фоновые вычисления
        self.page.on_window_event = self.on_window_event
        
        # Состояние модального превью
        self._preview_dialog = None
        
        # Глобальные клавиатурные сокращения для превью
        self.page.on_keyboard_event = self._on_preview_keyboard
        self.page.update()

    def _pick_folder(self, title: str) -> str | None:
        """Открывает нативный диалог выбора папки через tkinter."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path if path else None
    
    def create_layout(self):
        """Создание основного layout"""
        self._current_lang = LANG
        header_row = ft.Row(
            [
                ft.Container(expand=True),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        
        # Sidebar
        self.sidebar_stats = ft.Container(
            content=self.create_stats_section(),
            height=120,
        )
        self.sidebar_clusters = ft.Container(
            content=ft.Column(
                [
                    self.create_clusters_section(),
                ],
                scroll=ft.ScrollMode.AUTO,
            ),
            expand=True,
        )
        sidebar_content = ft.Column(
            [
                self.sidebar_stats,
                ft.Divider(),
                self.sidebar_clusters,
            ],
            expand=True,
                )

        self.sidebar = ft.Container(
            image=ft.DecorationImage(
                src=os.path.join(config.BASE_DIR, "assets", "wallpaper_left.png"),
                fit=ft.BoxFit.COVER,
            ),
            content=ft.Container(
                content=sidebar_content,
                bgcolor=ft.Colors.with_opacity(0.1, self._WARM_SURFACE),
                padding=15,
                expand=True,
            ),
            width=280,
            border_radius=10,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        )
        
        # Основная область (с обойным фоном)
        self.tab_content = ft.Container(
            content=ft.Row(
                [ft.ProgressRing(width=32, height=32), ft.Text(tr("loading"), size=14)],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=10,
            ),
            expand=True,
        )
        self.current_tab = -1
        main_content_column = ft.Column(
            [
                self.create_controls_panel(),
                ft.Divider(),
                self.tab_content,
            ],
            expand=True,
            spacing=10,
        )
        self.main_content = ft.Container(
            image=ft.DecorationImage(
                src=os.path.join(config.BASE_DIR, "assets", "wallpaper_right.png"),
                fit=ft.BoxFit.COVER,
            ),
            content=ft.Container(
                content=main_content_column,
                bgcolor=ft.Colors.with_opacity(0.1, self._WARM_SURFACE),
                padding=15,
                border_radius=10,
                expand=True,
            ),
            expand=True,
            border_radius=10,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        )
        
        # Добавляем на страницу: вся разметка поверх отдельных фонов.
        self.page.add(
            ft.Column(
                [
                    header_row,
                    ft.Row(
                        [self.sidebar, self.main_content],
                        expand=True,
                        spacing=20,
                    ),
                ],
                expand=True,
                spacing=10,
            )
        )
    


    def _set_window_progress(self, label: str, current: int = 0, total: int = 0, key: str = None, disk_current: int = None, disk_total: int = None):
        """Показывает стадию и счётчик в заголовке окна.

        Если заданы disk_current/disk_total, отображает двойной прогресс:
        'Сканирование... 2/3 (500/1200)'.

        key — стабильный идентификатор владельца заголовка (напр. "prewarm",
        "scan", "export"). Если не задан, используется сам label; для проверки
        «чей сейчас заголовок» сравнивайте именно key, а не локализованную
        строку label (она меняется при переключении языка).
        """
        self._win_progress_stage = label
        self._win_progress_current = current
        self._win_progress_total = total
        self._win_progress_key = key if key is not None else label
        if disk_current is not None and disk_total is not None and disk_total > 0:
            if total and total > 0:
                self.page.title = f"{label}... {disk_current}/{disk_total} ({current}/{total})"
            else:
                self.page.title = f"{label}... {disk_current}/{disk_total}"
        elif total and total > 0:
            self.page.title = f"{label}... {current}/{total}"
        else:
            self.page.title = f"{label}..."
        self.page.update()

    def _reset_window_progress(self):
        """Возвращает стандартный заголовок окна."""
        self._win_progress_stage = None
        self._win_progress_current = 0
        self._win_progress_total = 0
        self._win_progress_key = None
        self.page.title = tr("app.title")
        self.page.update()

    def _switch_language(self, lang: str):
        """Переключает язык приложения."""
        set_language(lang)
        self._current_lang = lang
        self._apply_language()
        self.page.run_task(self.load_stats)
        # Перестраиваем текущую вкладку на новом языке
        if self.current_tab == -1:
            self._reset_gallery_lazy_loading("cluster")
            asyncio.create_task(self.show_clusters_tab())
        elif self.current_tab == 0:
            self._reset_gallery_lazy_loading("search_results")
            self.show_search_tab()
        elif self.current_tab == 1:
            self._reset_gallery_lazy_loading("export")
            asyncio.create_task(self.show_export_tab())
        self.page.update()

    def _apply_language(self):
        """Обновляет все статические тексты интерфейса на текущий язык."""
        # Шапка
        self.page.title = tr("app.title")
        if hasattr(self, "lang_rb"):
            self.lang_rb.text = tr("lang.ru")
        if hasattr(self, "lang_en_btn"):
            self.lang_en_btn.text = tr("lang.en")
        if hasattr(self, "lang_rb"):
            self._update_lang_button_styles(self.lang_rb, self._current_lang == "ru")
        if hasattr(self, "lang_en_btn"):
            self._update_lang_button_styles(self.lang_en_btn, self._current_lang == "en")

        # Статистика
        if hasattr(self, "stats_title_text"):
            self.stats_title_text.value = tr("stat.title")
            self.stat_total_label.value = tr("stat.total")
            self.stat_dupes_label.value = tr("stat.dupes")
            self.stat_unique_label.value = tr("stat.unique")

        # Категории
        if hasattr(self, "categories_header"):
            self.categories_header.value = tr("categories.title", count=len(self.clusters))

        # Сканирование
        if hasattr(self, "scan_path_input"):
            self.scan_path_input.label = tr("scan.path.label")
            self.scan_path_input.hint_text = tr("scan.path.hint")
            if is_all_disks(self.scan_path_input.value):
                self.scan_path_input.value = tr("scan.all_disks")
            self.search_input.value = ""
            self.export_dest_folder.value = ""
        if hasattr(self, "browse_scan_path_button"):
            self.browse_scan_path_button.tooltip = tr("scan.browse")
        if hasattr(self, "browse_export_button"):
            self.browse_export_button.tooltip = tr("pick.export")
        if hasattr(self, "mode_scan_btn"):
            self.mode_scan_btn.content = tr("scan.button.scan")
            self.mode_scan_btn.tooltip = tr("scan.button.scan")
        if hasattr(self, "reset_button"):
            self.reset_button.content = tr("scan.button.reset")
            self.reset_button.tooltip = tr("scan.button.reset")

        # Кнопки режимов
        if hasattr(self, "mode_search_btn"):
            self.mode_search_btn.content = tr("tab.search")
        if hasattr(self, "mode_export_btn"):
            self.mode_export_btn.content = tr("tab.export")

        # Поиск
        if hasattr(self, "search_input"):
            self.search_input.label = tr("search.label")
            self.search_input.hint_text = tr("search.hint")
            self.search_button.content = tr("search.button")
            self.search_button.tooltip = tr("search.button")

        # Экспорт
        if hasattr(self, "export_button"):
            if self.exporting:
                self.export_button.content = tr("export.button.stop")
                self.export_button.tooltip = tr("export.button.stop")
            else:
                self.export_button.content = tr("export.button")
                self.export_button.tooltip = tr("export.button")
        if hasattr(self, "export_dest_folder"):
            self.export_dest_folder.label = tr("export.folder")
            self.export_dest_folder.hint_text = tr("export.default_folder")

        # Заголовок окна: перерисовываем стадию прогресса на новом языке
        # или возвращаем стандартный, если прогресс не активен.
        if self._win_progress_key is not None:
            stage_labels = {
                "scan": tr("stage.scan"),
                "dedup": tr("stage.dedup"),
                "embed": tr("stage.embed"),
                "cluster": tr("stage.cluster"),
                "prewarm": tr("stage.thumbs"),
                "export": tr("stage.export"),
            }
            label = stage_labels.get(self._win_progress_key, self._win_progress_stage or tr("app.title"))
            if self._win_progress_total and self._win_progress_total > 0:
                self.page.title = f"{label}... {self._win_progress_current}/{self._win_progress_total}"
            else:
                self.page.title = f"{label}..."
        else:
            self.page.title = tr("app.title")

    def _update_lang_button_styles(self, btn: ft.TextButton, is_active: bool):
        """Обновляет стили кнопки языка."""
        if is_active:
            btn.style = ft.ButtonStyle(
                bgcolor=self._WARM_PRIMARY,
                color=self._WARM_ON_PRIMARY,
            )
        else:
            btn.style = ft.ButtonStyle(
                bgcolor=self._WARM_SURFACE,
                color=self._WARM_ON_SURFACE,
            )

    def create_stats_section(self):
        """Секция статистики"""
        self.stat_total = ft.Text("0", size=16, weight=ft.FontWeight.BOLD, color=self._WARM_PRIMARY)
        self.stat_total_size = ft.Text("0 " + tr("unit.B"), size=11, color=self._WARM_ON_SURFACE_VARIANT)
        self.stat_duplicates = ft.Text("0", size=16, weight=ft.FontWeight.BOLD, color=self._WARM_ERROR)
        self.stat_duplicates_size = ft.Text("0 " + tr("unit.B"), size=11, color=self._WARM_ON_SURFACE_VARIANT)
        self.stat_unique = ft.Text("0", size=16, weight=ft.FontWeight.BOLD, color="#81C784")
        self.stat_unique_size = ft.Text("0 " + tr("unit.B"), size=11, color=self._WARM_ON_SURFACE_VARIANT)
        self.stat_selected = ft.Text("0", size=16, weight=ft.FontWeight.BOLD, color="#64B5F6")
        self.stat_selected_size = ft.Text("0 " + tr("unit.B"), size=11, color=self._WARM_ON_SURFACE_VARIANT)
        
        self.stats_title_text = ft.Text(tr("stat.title"), size=16, weight=ft.FontWeight.BOLD)
        self.stat_total_label = ft.Text(tr("stat.total"), size=11)
        self.stat_dupes_label = ft.Text(tr("stat.dupes"), size=11)
        self.stat_unique_label = ft.Text(tr("stat.unique"), size=11)
        self.stat_selected_label = ft.Text(tr("stat.selected"), size=11)
        
        return ft.Column(
            [
                self.stats_title_text,
                ft.Row(
                    [
                        ft.Column(
                            [self.stat_total, self.stat_total_size, self.stat_total_label],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Column(
                            [self.stat_duplicates, self.stat_duplicates_size, self.stat_dupes_label],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Column(
                            [self.stat_unique, self.stat_unique_size, self.stat_unique_label],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Column(
                            [self.stat_selected, self.stat_selected_size, self.stat_selected_label],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                ),
            ],
            spacing=10,
        )
    
    def create_clusters_section(self):
        """Секция категорий - кнопки кластеров в 3 столбца"""
        self.clusters_grid = ft.Column([])
        
        # Заголовок с количеством категорий
        self.categories_header = ft.Text(tr("categories.title", count=0), size=16, weight=ft.FontWeight.BOLD)
        
        # Активный кластер для подсветки
        self.active_cluster_id = None
        
        return ft.Column(
            [
                self.categories_header,
                self.clusters_grid,
            ],
            spacing=10,
        )
    
    def update_clusters_list(self):
        """Обновление списка категорий в боковой панели (кнопки в 3 столбца)"""
        self.clusters_grid.controls.clear()
        
        if not self.clusters:
            self.categories_header.value = tr("categories.title", count=0)
            return
        
        self.categories_header.value = tr("categories.title", count=len(self.clusters))
        
        self._cluster_display_map = {
            cid: i + 1 for i, (cid, _) in enumerate(sorted(self.clusters.items()))
        }
        
        clusters_list = sorted(self.clusters.items())
        columns = [[] for _ in range(3)]
        
        for i, (cluster_id, members) in enumerate(clusters_list):
            col_idx = i % 3
            columns[col_idx].append((cluster_id, members))
        
        # Создаём строки с кнопками
        max_rows = max(len(col) for col in columns)
        
        for row_idx in range(max_rows):
            row_buttons = []
            for col_idx in range(3):
                if row_idx < len(columns[col_idx]):
                    cluster_id, members = columns[col_idx][row_idx]
                    is_active = self.active_cluster_id == cluster_id
                    has_selected = any(path in self.selected_images for path in members)
                    if is_active:
                        bgcolor = self._WARM_PRIMARY
                        color = self._WARM_ON_PRIMARY
                    elif has_selected:
                        bgcolor = "#5D4037"
                        color = "#EFEBE9"
                    else:
                        bgcolor = self._WARM_SURFACE
                        color = self._WARM_ON_SURFACE
                    button = ft.ElevatedButton(
                        f"{self._cluster_display_map[cluster_id]} ({len(members)})",
                        data=cluster_id,
                        width=70,
                        height=50,
                        bgcolor=bgcolor,
                        color=color,
                        on_click=self.on_cluster_button_click,
                        style=ft.ButtonStyle(
                            text_style=ft.TextStyle(size=13, weight=ft.FontWeight.BOLD),
                            padding=ft.Padding(6, 4, 6, 4),
                            shape=ft.RoundedRectangleBorder(radius=6),
                        ),
                    )
                    row_buttons.append(button)
                else:
                    row_buttons.append(ft.Container(width=70))
            
            self.clusters_grid.controls.append(
                ft.Row(row_buttons, spacing=6, alignment=ft.MainAxisAlignment.CENTER)
            )
    
    def on_cluster_button_click(self, e):
        """Обработчик клика на кнопку кластера"""
        cluster_id = e.control.data
        self._save_current_gallery_scroll()
        self.active_cluster_id = cluster_id
        self.current_tab = -1
        self.current_mode = "cluster"
        self._update_mode_buttons()
        self._update_context_row()
        asyncio.create_task(self.show_clusters_tab())
        # Обновляем подсветку кнопок
        self.update_clusters_list()
    
    async def on_window_event(self, e):
        """Обработчик событий окна — при закрытии останавливаем фоновые вычисления."""
        try:
            if e is not None and getattr(e, "type", None) == ft.WindowEventType.CLOSE:
                await self._cancel_scan_workers()
                print("Приложение закрыто.")
                import sys as _sys
                _sys.exit(0)
        except Exception as ex:
            print(f"Ошибка в обработчике окна: {ex}")

    def on_window_resize(self, e):
        """Обработчик изменения размера окна - пересоздаём галерею"""
        self._save_current_gallery_scroll()
        current_tab = self.current_tab
        if current_tab == -1:
            scope = f"cluster_{self.active_cluster_id}"
        elif current_tab == 0:
            scope = "search_results"
        elif current_tab == 1:
            scope = "export"
        else:
            return
        
        if current_tab == -1:
            asyncio.create_task(self.show_clusters_tab())
        elif current_tab == 0:
            self.show_search_tab()
        elif current_tab == 1:
            asyncio.create_task(self.show_export_tab())
        
        if self._preview_dialog is not None and self._preview_dialog.open:
            self._update_preview_size()
    
    def create_controls_panel(self):
        """Верхняя панель управления: Row 1 (режимы) + Row 2 (контекстные контролы)."""
        # --- Row 1: кнопки режимов и сброс ---
        self.mode_scan_btn = ft.ElevatedButton(
            tr("scan.button.scan"),
            icon=ft.Icons.SEARCH,
            bgcolor=self._WARM_SURFACE,
            color=self._WARM_ON_SURFACE,
            tooltip=tr("scan.button.scan"),
            on_click=lambda e: self.switch_mode("scan"),
        )
        self.mode_search_btn = ft.ElevatedButton(
            tr("tab.search"),
            icon=ft.Icons.SEARCH,
            bgcolor=self._WARM_SURFACE,
            color=self._WARM_ON_SURFACE,
            on_click=lambda e: self.switch_mode("search"),
        )
        self.mode_export_btn = ft.ElevatedButton(
            tr("tab.export"),
            icon=ft.Icons.FOLDER_OPEN,
            bgcolor=self._WARM_SURFACE,
            color=self._WARM_ON_SURFACE,
            on_click=lambda e: self.switch_mode("export"),
        )
        self.reset_button = ft.ElevatedButton(
            tr("scan.button.reset"),
            icon=ft.Icons.DELETE_FOREVER,
            bgcolor=self._WARM_ERROR,
            color="#000000",
            tooltip=tr("scan.button.reset"),
            on_click=self._show_reset_dialog,
        )
        row1 = ft.Row(
            [
                self.mode_scan_btn,
                self.mode_search_btn,
                self.mode_export_btn,
                ft.Container(expand=True),
                self.reset_button,
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.START,
            alignment=ft.MainAxisAlignment.START,
        )

        # --- Row 2: контекстные контролы текущего режима ---
        # Сканирование
        self.scan_path_input = ft.TextField(
            label=tr("scan.path.label"),
            value=tr("scan.all_disks"),
            expand=True,
            height=48,
            border_radius=8,
            hint_text=tr("scan.path.hint"),
        )
        self.browse_scan_path_button = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            height=48,
            tooltip=tr("scan.browse"),
            on_click=self.browse_scan_path,
        )
        self.start_scan_btn = ft.ElevatedButton(
            tr("scan.button.start"),
            icon=ft.Icons.PLAY_ARROW,
            height=48,
            bgcolor=self._WARM_PRIMARY,
            color=self._WARM_ON_PRIMARY,
            on_click=lambda e: asyncio.create_task(self.toggle_scan(e)),
        )
        self.scan_context_row = ft.Row(
            [self.scan_path_input, self.browse_scan_path_button, self.start_scan_btn],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.END,
        )

        # Поиск
        self.search_input = ft.TextField(
            label=tr("search.label"),
            hint_text=tr("search.hint"),
            expand=True,
            height=48,
            on_submit=self.do_search,
        )
        self.search_button = ft.ElevatedButton(
            tr("search.button"),
            icon=ft.Icons.SEARCH,
            height=48,
            bgcolor=self._WARM_PRIMARY,
            color=self._WARM_ON_PRIMARY,
            tooltip=tr("search.button"),
            on_click=self.do_search,
        )
        self.search_results_container = ft.Column(
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.search_context_row = ft.Row(
            [self.search_input, self.search_button],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.END,
        )

        # Экспорт
        default_path = getattr(self, "_export_dest_folder_path", None) or \
            getattr(self.page.session, "export_dest_folder", None) or tr("export.default_folder")
        self.export_dest_folder = ft.TextField(
            label=tr("export.folder"),
            value=default_path,
            expand=True,
            height=48,
            on_change=self._on_export_path_changed,
        )
        self.browse_export_button = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            height=48,
            tooltip=tr("pick.export"),
            on_click=lambda e: self.browse_destination_folder(self.export_dest_folder),
        )
        self.export_button = ft.ElevatedButton(
            tr("export.button"),
            icon=ft.Icons.DOWNLOAD,
            height=48,
            bgcolor=self._WARM_PRIMARY,
            color=self._WARM_ON_PRIMARY,
            tooltip=tr("export.button"),
            on_click=self.toggle_export,
        )
        self.export_context_row = ft.Row(
            [self.export_dest_folder, self.browse_export_button, self.export_button],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # Категории (из боковой панели)
        self.cluster_select_all_icon_button = ft.IconButton(
            icon=ft.Icons.CHECK_BOX_OUTLINE_BLANK,
            tooltip=tr("select_all.toggle"),
        )
        self.selected_count_text = ft.Text(tr("selected.count", count=0), size=14)
        self.cluster_context_row = ft.Row(
            [self.cluster_select_all_icon_button, self.selected_count_text],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self.context_container = ft.Container(
            content=self.scan_context_row,
        )

        return ft.Column(
            [row1, ft.Container(expand=True), self.context_container],
            spacing=12,
            height=120,
            alignment=ft.MainAxisAlignment.START,
        )
    
    def _update_mode_buttons(self):
        """Подсветка активной кнопки режима."""
        for btn in (self.mode_scan_btn, self.mode_search_btn, self.mode_export_btn):
            is_active = (
                (btn is self.mode_scan_btn and self.current_mode == "scan") or
                (btn is self.mode_search_btn and self.current_mode == "search") or
                (btn is self.mode_export_btn and self.current_mode == "export")
            )
            if is_active and not self.scanning:
                btn.bgcolor = self._WARM_PRIMARY
                btn.color = self._WARM_ON_PRIMARY
                btn.style = ft.ButtonStyle(
                    bgcolor=self._WARM_PRIMARY,
                    color=self._WARM_ON_PRIMARY,
                )
            else:
                btn.bgcolor = self._WARM_SURFACE
                btn.color = self._WARM_ON_SURFACE
                btn.style = ft.ButtonStyle(
                    bgcolor=self._WARM_SURFACE,
                    color=self._WARM_ON_SURFACE,
                )

    def _update_context_row(self):
        """Показывает контекстные контролы текущего режима."""
        if self.current_mode == "search":
            self.context_container.content = self.search_context_row
        elif self.current_mode == "export":
            self.context_container.content = self.export_context_row
        elif self.current_mode == "cluster":
            self.context_container.content = self.cluster_context_row
        else:
            self.context_container.content = self.scan_context_row

    def switch_mode(self, mode: str):
        """Переключение режима: scan / search / export / cluster."""
        self._save_current_gallery_scroll()
        self.current_mode = mode
        self._update_mode_buttons()
        self._update_context_row()
        self.page.update()

        if mode == "scan":
            # Только переключаемся на режим сканирования.
            # Запуск сканирования выполняется кнопкой "Start scan" (start_scan_btn).
            pass
        elif mode == "search":
            self.current_tab = 0
            self.show_search_tab()
        elif mode == "export":
            self.current_tab = 1
            asyncio.create_task(self.show_export_tab())
        elif mode == "cluster":
            self.current_tab = -1
            asyncio.create_task(self.show_clusters_tab())

    async def preload_model(self):
        """Фоновая загрузка модели.

        Загрузка/скачивание SigLIP выполняется в отдельном потоке
        (asyncio.to_thread), чтобы не блокировать UI-поток Flet.
        """
        self.model_loading = True
        try:
            from core.embedder import get_embedder
            await asyncio.to_thread(get_embedder)
            self.model_loaded = True
            self._model_status_error = None

            # Фоновая предзагрузка FAISS-индекса: строим/читаем с диска заранее,
            # чтобы первый поиск не ждал ни модель, ни построение индекса.
            # Выполняем в отдельном потоке, чтобы не блокировать UI event loop.
            try:
                from core import search as search_mod
                await asyncio.to_thread(search_mod.prefetch_index)
            except Exception as _ie:
                print(f"Prefetch FAISS index error: {_ie}")
        except Exception as e:
            self._model_status_error = str(e)
        finally:
            self.model_loading = False
            self.page.update()
    
    async def load_stats(self):
        """Загрузка статистики из БД (асинхронно, чтобы не блокировать UI)."""
        try:
            stats = await asyncio.to_thread(database.load_db_stats)
            
            self.stat_total.value = str(stats["total"])
            
            # Обновляем размеры
            total_size = stats.get("total_size", 0)
            self.stat_total_size.value = _format_size(total_size)
            
            # Дедупликация для только что найденных файлов ещё не выполнена:
            # до её завершения строки в dedup_groups отсутствуют, поэтому
            # из БД получили бы уникальные == общему числу файлов, что неверно.
            # Также на старте приложения, если файлы есть, но дедупликация
            # не была завершена (нет pHash), показываем 0, а не обманчивое N.
            dedup_pending = getattr(self, '_scan_phase', 'idle') in ("scan", "dedup")
            if not dedup_pending:
                dedup_pending = await asyncio.to_thread(database.has_pending_dedup) and stats["total"] > 0
            if dedup_pending:
                self.stat_duplicates.value = "0"
                self.stat_unique.value = "0"
                self.stat_duplicates_size.value = "0 " + tr("unit.B")
                self.stat_unique_size.value = "0 " + tr("unit.B")
            else:
                self.stat_duplicates.value = str(stats["duplicates"])
                self.stat_unique.value = str(stats["unique"])
                # Размер дубликатов и уникальных — вычисляем из БД
                try:
                    dup_size = await asyncio.to_thread(database.get_duplicates_size)
                    self.stat_duplicates_size.value = _format_size(dup_size)
                    self.stat_unique_size.value = _format_size(max(total_size - dup_size, 0))
                except Exception:
                    self.stat_duplicates_size.value = "—"
                    self.stat_unique_size.value = "—"
            
            # Обновляем статистику выбранных файлов (мгновенно из БД,
            # без обращения к файловой системе — файлы могут быть на внешнем диске)
            selected_count, total_size_bytes = database.get_selected_files_stats()
            self.stat_selected.value = str(selected_count)
            self.stat_selected_size.value = _format_size(total_size_bytes)
            
            # Загрузить кластеры и их имена
            clusters, cluster_names = await asyncio.to_thread(database.load_clusters_with_names)
            old_clusters_id = id(self.clusters) if self.clusters else None
            self.clusters = clusters or {}
            self.cluster_names = cluster_names or {}
            
            # Инвалидировать кэш если кластеры изменились
            if old_clusters_id != id(self.clusters):
                self._invalidate_cluster_cache()
            
            # Обновить список категорий в боковой панели
            self.update_clusters_list()

            self.page.update()

            # Если это обычный старт без активного сканирования/других стадий,
            # проверяем миниатюры и догенерируем недостающие.
            # ВАЖНО: если дедупликация ещё не завершена (есть файлы без pHash),
            # миниатюры не генерируем — иначе они будут созданы и для
            # дубликатов, которые после дедупликации будут отфильтрованы.
            try:
                dedup_pending = await asyncio.to_thread(database.has_pending_dedup)
                if (not self.scanning and getattr(self, '_scan_phase', 'idle') == 'idle'
                        and stats.get('total', 0) > 0 and not dedup_pending):
                    await self.ensure_thumbnails()
            except Exception as e:
                print(f"Ошибка проверки миниатюр: {e}")
        except Exception as e:
            print(f"Ошибка загрузки статистики: {e}")

    def _update_selected_stats_async(self):
        """Обновляет статистику выбранных файлов в боковой панели из БД.

        Использует уже сохранённые в БД размеры (не ходит в файловую систему),
        поэтому работает мгновенно даже для файлов на внешнем диске.
        """
        try:
            selected_count, total_size_bytes = database.get_selected_files_stats()
            if hasattr(self, 'stat_selected'):
                self.stat_selected.value = str(selected_count)
            if hasattr(self, 'stat_selected_size'):
                self.stat_selected_size.value = _format_size(total_size_bytes)
            self.page.update()
        except Exception as e:
            print(f"Ошибка обновления статистики выбранных: {e}")
    
    def _find_gallery(self, control):
        """Рекурсивно ищет GridView в дереве контролов."""
        if isinstance(control, ft.GridView):
            return control
        if hasattr(control, 'controls'):
            for child in control.controls:
                result = self._find_gallery(child)
                if result:
                    return result
        return None

    def _save_current_gallery_scroll(self):
        """Сохранить scroll offset текущей галереи"""
        # Flet GridView не предоставляет прямой доступ к scroll offset.
        # Scroll position уже сохраняется в обработчике on_scroll
        # (create_scroll_handler), поэтому здесь ничего не делаем.
        pass

    def _restore_gallery_scroll(self, gallery, scope, paths=None, page_size=None):
        """Восстановить scroll offset галереи (упрощённая версия для Flet 0.86)"""
        try:
            scroll_key = f"gallery_scroll_{scope}"
            offset = getattr(self.page.session, scroll_key, None)
            if offset is not None and isinstance(offset, (int, float)) and offset > 0:
                async def do_restore():
                    # Ждём пока галерея загрузится и отобразится
                    for _ in range(20):
                        await asyncio.sleep(0.1)
                        try:
                            gallery.scroll_to(offset=offset, duration=0)
                            self.page.update()
                            return
                        except Exception:
                            pass
                asyncio.create_task(do_restore())
        except Exception:
            pass

    def _reset_gallery_lazy_loading(self, scope: str):
        """Сброс lazy loading state для указанного scope при смене вкладки."""

        # Сбрасываем offset до 0
        offset_key = f"gallery_offset_{scope}"
        setattr(self.page.session, offset_key, 0)

        # Сбрасываем флаг загрузки
        loading_key = f"gallery_loading_{scope}"
        setattr(self.page.session, loading_key, False)

        # Сбрасываем позицию скролла
        scroll_key = f"gallery_scroll_{scope}"
        setattr(self.page.session, scroll_key, 0)

    def _progress_callback(self, stage: str, current: int, total: int, message: str):
        """Callback для обновления прогресса из фоновых потоков.

        Прогресс показывается в заголовке окна как 'Стадия... счётчик/всего'
        (например 'Сканирование... 3/100'). Длинное сообщение (message) в UI
        больше не выводится — только стадия и счётчик.

page.update() в Flet НЕ потокобезопасен — на Windows вызов из чужого потока
        даёт WinError 1. Используем page.run_task() для переноса обновления
        в главный (UI) поток.
        """
        now = time.monotonic()
        if now - getattr(self, '_last_progress_update', 0) < 0.05:
            return
        self._last_progress_update = now
        try:
            stage_keys = {
                "scan": "scan",
                "dedup": "dedup",
                "embed": "embed",
                "cluster": "cluster",
            }
            stage_labels = {
                "scan": tr("stage.scan"),
                "dedup": tr("stage.dedup"),
                "embed": tr("stage.embed"),
                "cluster": tr("stage.cluster"),
            }
            key = stage_keys.get(stage, stage)
            label = stage_labels.get(stage, stage)

            async def update_ui():
                if not self.scanning:
                    return
                # Only update if this stage still owns the window progress
                if self._win_progress_key != key:
                    return
                self._set_window_progress(label, current, total, key=key)

            self.page.run_task(update_ui)
        except Exception as ex:
            print(f"Progress callback error: {ex}")
            import traceback
            traceback.print_exc()
    
    def _get_available_disks(self):
        """Возвращает список доступных дисков/точек монтирования.

        На Windows проверяет буквы дисков (C:\\, D:\\, ...).
        На macOS/Linux возвращает корневые точки монтирования.
        """
        if sys.platform == "win32":
            disks = []
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {}
                    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        path = f"{letter}:\\"
                        futures[path] = executor.submit(os.path.exists, path)
                    
                    for path, future in futures.items():
                        try:
                            if future.result(timeout=2):
                                disks.append(path)
                        except Exception:
                            continue
            except Exception:
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    path = f"{letter}:\\"
                    if os.path.exists(path):
                        disks.append(path)
            return disks
        else:
            # macOS/Linux: / и /Volumes/* (macOS дополнительно монтирует
            # внешние диски в /Volumes)
            disks = ["/"]
            root_dev = os.stat("/").st_dev
            volumes = "/Volumes"
            if os.path.isdir(volumes):
                for name in os.listdir(volumes):
                    full = os.path.join(volumes, name)
                    if os.path.isdir(full):
                        try:
                            if os.stat(full).st_dev != root_dev:
                                disks.append(full)
                        except OSError:
                            continue
            return disks

    async def _cancel_scan_workers(self):
        """Отменяет все фоновые задачи сканирования и дожидается завершения."""
        self.scanning = False
        scanner.STOP_REQUESTED = True

        if self.scan_task is not None:
            self.scan_task.cancel()
            self.scan_task = None

        tasks = list(getattr(self, "_scan_worker_tasks", []))
        self._scan_worker_tasks = []
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass

    async def _run_scan_workflow(self, scan_paths):
        try:
            files = []
            total_disks = len(scan_paths)
            global_total = getattr(self, '_global_scan_total', 0)
            per_disk_totals = getattr(self, '_per_disk_totals', {})
            # Дубли в БД не прибавляем: incremental_scan пересканирует все файлы
            # на диске, поэтому счётчик прогресса должен идти от 0 до количества
            # файлов на диске (иначе уже проиндексированные файлы посчитаются дважды).
            cumulative_base = 0
            
            for i, path in enumerate(scan_paths):
                if scanner.STOP_REQUESTED:
                    break
                disk_idx = i + 1
                disk_total = per_disk_totals.get(path, 0)
                
                if total_disks == 1:
                    def _single_disk_progress(stage, current, total, message):
                        if stage == "scan":
                            now = time.monotonic()
                            last = getattr(self, '_last_progress_update', 0)
                            if now - last < 0.05:
                                return
                            self._last_progress_update = now
                            async def _update():
                                if not self.scanning:
                                    return
                                if self._win_progress_key != "scan":
                                    return
                                self._set_window_progress(tr("stage.scan"), current, total, key="scan")
                            self.page.run_task(_update)
                        else:
                            self._progress_callback(stage, current, total, message)
                    disk_progress_cb = _single_disk_progress
                else:
                    def _disk_progress(stage, current, total, message, _cumulative_base=cumulative_base, _disk_total=disk_total, _global_total=global_total, _disk_idx=disk_idx, _total_disks=total_disks):
                        if stage == "scan":
                            now = time.monotonic()
                            last = getattr(self, '_last_disk_progress_update', 0)
                            if now - last < 0.15:
                                return
                            self._last_disk_progress_update = now
                            async def _update_disk_progress():
                                if not self.scanning:
                                    return
                                if self._win_progress_key != "scan":
                                    return
                                self._set_window_progress(tr("stage.scan"), _cumulative_base + current, _global_total, key="scan", disk_current=_disk_idx, disk_total=_total_disks)
                            self.page.run_task(_update_disk_progress)
                        else:
                            self._progress_callback(stage, current, total, message)
                    disk_progress_cb = _disk_progress
                
                self._set_window_progress(tr("stage.scan"), cumulative_base, global_total, key="scan", disk_current=disk_idx, disk_total=total_disks)
                
                task = asyncio.create_task(asyncio.to_thread(
                    scanner.run,
                    path,
                    None, None, None,
                    incremental=True,
                    progress_callback=disk_progress_cb,
                    total=disk_total,
                ))
                self._scan_worker_tasks.append(task)
                try:
                    result = await task
                finally:
                    if task in self._scan_worker_tasks:
                        self._scan_worker_tasks.remove(task)
                if result:
                    files.extend(result)
                    await self.load_stats()
                
                cumulative_base += disk_total
            
            if scanner.STOP_REQUESTED:
                return
            
            if not files:
                self.show_snackbar(tr("scan.no_files"), "#FFB74D")
                return
            
            self._scan_phase = "dedup"
            
            # 2. Дедупликация
            self._set_window_progress(tr("stage.dedup"), key="dedup")
            
            task = asyncio.create_task(asyncio.to_thread(dedup.run, incremental=True, progress_callback=self._progress_callback))
            self._scan_worker_tasks.append(task)
            try:
                await task
            finally:
                if task in self._scan_worker_tasks:
                    self._scan_worker_tasks.remove(task)

            if scanner.STOP_REQUESTED:
                return

            # Дедупликация завершена (группы сохранены в БД) — переводим фазу
            # ДО load_stats, чтобы статистика показала реальные значения
            # дубликатов/уникальных, а не нулевую «до дедупликации».
            self._scan_phase = "embed"

            # Обновляем статистику после дедупликации (дубликаты уже сохранены в БД)
            await self.load_stats()
            
            # 3. Эмбеддинги
            self._set_window_progress(tr("stage.embed"), key="embed")
            
            task = asyncio.create_task(asyncio.to_thread(embedder.run, incremental=True, progress_callback=self._progress_callback))
            self._scan_worker_tasks.append(task)
            try:
                result = await task
            finally:
                if task in self._scan_worker_tasks:
                    self._scan_worker_tasks.remove(task)

            # Эмбеддинги могли измениться — сбрасываем кэш семантического поиска,
            # чтобы последующие запросы использовали свежие данные.
            search.clear_cache()

            if scanner.STOP_REQUESTED:
                return
            
            self._scan_phase = "cluster"
            
            # 4. Кластеризация (только если включено в конфиге)
            if config.AUTO_CLUSTER_AFTER_SCAN:
                self._set_window_progress(tr("stage.cluster"), key="cluster")
                
                task = asyncio.create_task(asyncio.to_thread(clustererhdb.run, progress_callback=self._progress_callback))
                self._scan_worker_tasks.append(task)
                try:
                    await task
                finally:
                    if task in self._scan_worker_tasks:
                        self._scan_worker_tasks.remove(task)
                
                if scanner.STOP_REQUESTED:
                    return
                
                # Обновить статистику после кластеризации
                await self.load_stats()
            
            self._scan_phase = "done"
            
            # 5. По умолчанию открываем первый кластер (как в боковой панели).
            if self.clusters:
                self.switch_mode("cluster")
            
            # 6. Фоновая прегенерация миниатюр (WebP) в БД.
            # Порядок обхода — строго как кнопки кластеров в боковой панели.
            # Прогресс виден в заголовке окна, UI не блокируется.
            self._start_prewarm_thumbnails()

            self.show_snackbar(tr("ready"), "#81C784")
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            if self.scanning:
                self.show_snackbar(tr("error", error=e), self._WARM_ERROR)
        finally:
            self._scan_phase = "idle"
            print("Процесс завершён.")
            self.scanning = False
            self.start_scan_btn.content = tr("scan.button.start")
            self.start_scan_btn.icon = ft.Icons.PLAY_ARROW
            self.start_scan_btn.bgcolor = self._WARM_PRIMARY
            self.start_scan_btn.color = self._WARM_ON_PRIMARY
            if not (self._prewarm_task is not None and not self._prewarm_task.done()):
                self._reset_window_progress()

    def _start_prewarm_thumbnails(self):
        """Запускает фоновую прегенерацию миниатюр после скана (идемпотентно).

        Генерация идёт в фоне (asyncio-задача) и не блокирует UI; результат —
        готовые WebP BLOB в таблице thumbnails. Повторный запуск игнорируется,
        пока предыдущая задача ещё работает.
        """
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return
        self._prewarm_task = asyncio.create_task(self._prewarm_thumbnails())

    def _cancel_prewarm(self):
        """Отменяет фоновую прегенерацию миниатюр (сброс БД / новый скан)."""
        task = self._prewarm_task
        self._prewarm_task = None
        if task is not None and not task.done():
            task.cancel()

    async def ensure_thumbnails(self):
        """Проверяет миниатюры при старте и догенерирует недостающие.

        Прогресс показывается в заголовке окна. Не перезаписывает заголовок,
        если там уже активна другая стадия. Не запускается, если дедупликация
        ещё не завершена (has_pending_dedup) — иначе миниатюры были бы созданы
        и для дубликатов, которые будут отфильтрованы.
        """
        total = await asyncio.to_thread(database.count_missing_thumbnails)
        if total <= 0:
            return

        # Пока мы считали миниатюры в фоне, могла начаться дедупликация —
        # тогда генерацию откладываем до её завершения.
        if await asyncio.to_thread(database.has_pending_dedup):
            return

        # Защита от гонки: пока мы считали миниатюры в фоне, мог начаться
        # скан/дедупликация или заголовок мог занять другая стадия.
        # В этом случае не трогаем заголовок вовсе, иначе он «перескочит»
        # на «Миниатюры...» посреди другой стадии (напр. «Дедупликация...»).
        if getattr(self, '_win_progress_key', None) not in (None, "prewarm") or self.scanning or getattr(self, '_scan_phase', 'idle') != 'idle':
            return

        done = 0
        last_update = time.monotonic()
        title_owner = True
        setattr(self.page.session, "gallery_prewarm_active", True)
        try:
            self._set_window_progress(tr("stage.thumbs"), 0, total, key="prewarm")
        except Exception:
            pass
        try:
            while done < total:
                try:
                    if self._win_progress_key != "prewarm":
                        break
                except Exception:
                    break
                batch = await asyncio.to_thread(database.missing_thumbnail_paths, limit=config.PREWARM_CHUNK_SIZE)
                if not batch:
                    break
                await self._generate_thumbnails_parallel(batch, 150, max_workers=config.PREWARM_THUMB_WORKERS)
                done += len(batch)
                now = time.monotonic()
                if now - last_update >= 0.15:
                    last_update = now
                    try:
                        self._set_window_progress(tr("stage.thumbs"), done, total, key="prewarm")
                    except Exception:
                        pass
        finally:
            setattr(self.page.session, "gallery_prewarm_active", False)
            if title_owner and getattr(self, '_win_progress_key', None) == "prewarm":
                title_owner = False
                try:
                    self._reset_window_progress()
                except Exception:
                    pass
        if getattr(self, '_win_progress_key', None) == "prewarm":
            try:
                await asyncio.sleep(0.4)
                if getattr(self, '_win_progress_key', None) == "prewarm":
                    self._reset_window_progress()
            except Exception:
                pass

    async def _prewarm_thumbnails(self):
        """Фоновая генерация миниатюр для всех файлов кластеров.

        Порядок обхода — в точности как кнопки в боковой панели:
        sorted(self.clusters.items()) (cluster_id: -2, -1, 0, 1, ...).
        Кластеры содержат только каноничные (уникальные) файлы, поэтому
        дубликаты и некластеризованные пути здесь не трогаются — они
        покрываются ленивой генерацией при открытии галереи.

        Прогресс выводится в заголовок окна через _set_window_progress
        ('Миниатюры... 120/1048'); после завершения/отмены заголовок
        возвращается к стандартному (если не запущена другая стадия).
        """
        try:
            # Даём finally у scan-workflow вызвать _reset_window_progress(),
            # иначе он сотрёт установленный здесь заголовок.
            await asyncio.sleep(0.25)

            clusters = self.clusters or {}
            if not clusters:
                # Например, при отключённой авто-кластеризации — читаем из БД.
                clusters = await asyncio.to_thread(database.load_clusters) or {}

            order = []
            seen = set()
            for cid, members in sorted(clusters.items()):
                for p in members:
                    if p not in seen:
                        seen.add(p)
                        order.append(p)

            total = len(order)
            if total == 0:
                # Нечего прегенерировать — заголовок всё ещё принадлежит
                # предыдущей стадии (напр. «Кластеризация...»). Возвращаем
                # стандартный заголовок, иначе он навсегда останется висеть.
                try:
                    self._reset_window_progress()
                except Exception:
                    pass
                return

            done = 0
            last_update = time.monotonic()
            self._prewarm_title_owner = True
            setattr(self.page.session, "gallery_prewarm_active", True)
            # Сразу занимаем заголовок, не дожидаясь throttle: иначе при быстрой
            # генерации _win_progress_key не установится и finally не сбросит
            # заголовок, оставиvis прошлую стадию (напр. «Кластеризация...»).
            try:
                self._set_window_progress(tr("stage.thumbs"), 0, total, key="prewarm")
            except Exception:
                pass
            try:
                while done < total:
                    batch = order[done:done + config.PREWARM_CHUNK_SIZE]
                    await self._generate_thumbnails_parallel(
                        batch, 150, max_workers=config.PREWARM_THUMB_WORKERS
                    )
                    done += len(batch)
                    now = time.monotonic()
                    # Троттлинг обновления заголовка, чтобы не спамить page.update()
                    if now - last_update >= 0.15:
                        last_update = now
                        try:
                            self._set_window_progress(tr("stage.thumbs"), done, total, key="prewarm")
                        except Exception:
                            pass
            finally:
                setattr(self.page.session, "gallery_prewarm_active", False)
                # Сбрасываем заголовок на 'Pixel' ТОЛЬКО если он всё ещё принадлежит
                # прегенерации и не был перезаписан новым скан/стадией.
                # Сравниваем стабильный ключ, а не tr("stage.thumbs"): при смене
                # языка перевод меняется, иначе заголовок навсегда остался бы
                # «Миниатюры...» (см. также _set_window_progress).
                if self._prewarm_title_owner and self._win_progress_key == "prewarm":
                    self._prewarm_title_owner = False
                    try:
                        self._reset_window_progress()
                    except Exception:
                        pass

            # Страховка: если последнее обновление заголовка потерялось (Flet
            # может схлопнуть быстрые последовательные update), повторно приводим
            # заголовок к «Pixel». Не выполняется, если запущена новая стадия/скан.
            if self._win_progress_key == "prewarm":
                await asyncio.sleep(0.4)
                if self._win_progress_key == "prewarm":
                    try:
                        self._reset_window_progress()
                    except Exception:
                        pass
        except asyncio.CancelledError:
            if self._prewarm_title_owner and self._win_progress_key == "prewarm":
                self._prewarm_title_owner = False
                try:
                    self._reset_window_progress()
                except Exception:
                    pass
        except Exception:
            import traceback
            traceback.print_exc()
            if self._prewarm_title_owner and self._win_progress_key == "prewarm":
                self._prewarm_title_owner = False
                try:
                    self._reset_window_progress()
                except Exception:
                    pass

    async def toggle_scan(self, e):
        """Запуск/остановка сканирования"""
        if self.scanning:
            # Останавливаем сканирование
            await self._cancel_scan_workers()
            self.start_scan_btn.content = tr("scan.button.start")
            self.start_scan_btn.icon = ft.Icons.PLAY_ARROW
            self.start_scan_btn.bgcolor = self._WARM_PRIMARY
            self.start_scan_btn.color = self._WARM_ON_PRIMARY
            self._reset_window_progress()
            self.show_snackbar(tr("scan.stopped"), "#FFB74D")
            print("Сканирование остановлено пользователем.")
            return
        
        scan_path = self.scan_path_input.value
        
        # Проверка пути
        if is_all_disks(scan_path):
            # Сканируем все доступные диски
            disks = self._get_available_disks()
            if not disks:
                self.show_snackbar(tr("scan.no_disks"), self._WARM_ERROR)
                return
            scan_paths = disks
        elif not os.path.exists(scan_path):
            self.show_snackbar(tr("scan.bad_path", path=scan_path), self._WARM_ERROR)
            return
        else:
            scan_paths = [scan_path]

        # Убираем дубликаты и пересекающиеся пути (например, '/' + '/Volumes/*'
        # на macOS), чтобы один и тот же диск не сканировался и не считался дважды.
        scan_paths = _dedupe_scan_paths(scan_paths)
        
        self._cancel_prewarm()
        self._reset_window_progress()
        self.scanning = True
        scanner.STOP_REQUESTED = False
        self._scan_phase = "scan"

        # Сразу меняем состояние кнопки, чтобы UI отреагировал мгновенно,
        # а не после долгого предподсчёта файлов на диске.
        self.start_scan_btn.content = tr("scan.button.stop")
        self.start_scan_btn.icon = ft.Icons.STOP
        self.start_scan_btn.bgcolor = self._WARM_ERROR
        self.page.update()

        if len(scan_paths) > 1:
            self._set_window_progress(tr("progress.prepare"), key="scan")
            def _pre_count():
                totals = {}
                for p in scan_paths:
                    try:
                        totals[p] = scanner._count_matching_files(p)
                    except Exception:
                        totals[p] = 0
                return totals
            self._per_disk_totals = await asyncio.to_thread(_pre_count)
            self._global_scan_total = sum(self._per_disk_totals.values())
        else:
            self._per_disk_totals = {}
            def _pre_count_single():
                try:
                    return scanner._count_matching_files(scan_paths[0])
                except Exception:
                    return 0
            disk_total = await asyncio.to_thread(_pre_count_single)
            self._global_scan_total = disk_total
            self._per_disk_totals = {scan_paths[0]: disk_total}

        # Показываем прогресс в заголовке окна
        self._set_window_progress(tr("stage.scan"), key="scan")
        
        self.scan_task = asyncio.create_task(self._run_scan_workflow(scan_paths))
    
    async def toggle_export(self, e):
        """Запуск/остановка экспорта"""
        if self.exporting:
            self._export_stop_requested = True
            self.exporting = False
            self.export_button.content = tr("export.button")
            self.export_button.icon = ft.Icons.DOWNLOAD
            self.export_button.bgcolor = self._WARM_PRIMARY
            self.export_button.color = self._WARM_ON_PRIMARY
            if not self.scanning:
                self._reset_window_progress()
            self.page.update()
            self.show_snackbar(tr("export.stopped"), "#FFB74D")
            print("Экспорт остановлен пользователем.")
            return
        
        await self._run_export()
    
    def browse_scan_path(self, e):
        """Открыть нативный проводник для выбора пути сканирования."""
        path = self._pick_folder(tr("pick.scan"))
        if path is not None:
            self.scan_path_input.value = path
            self.page.update()
    
    def show_snackbar(self, message: str, color: str):
        """Показать уведомление"""
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(message),
            bgcolor=color,
        )
        self.page.snack_bar.open = True
        self.page.update()
    
    def _show_reset_dialog(self, e):
        """Показать диалог подтверждения сброса"""
        self._reset_dialog_title = ft.Text(tr("reset.title"))
        self._reset_dialog_content = ft.Text(tr("reset.prompt"))
        self._reset_dialog_cancel = ft.TextButton(tr("cancel"), on_click=lambda e: self._dismiss_reset_dialog())
        self._reset_dialog_confirm = ft.TextButton(tr("reset.confirm"), on_click=self._on_reset_confirm)
        self._reset_dialog = ft.AlertDialog(
            modal=True,
            title=self._reset_dialog_title,
            content=self._reset_dialog_content,
            actions=[
                self._reset_dialog_cancel,
                self._reset_dialog_confirm,
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        
        if self._reset_dialog not in self.page.overlay:
            self.page.overlay.append(self._reset_dialog)
        self._reset_dialog.open = True
        self.page.update()
    
    def _dismiss_reset_dialog(self):
        """Закрыть диалог сброса"""
        if self._reset_dialog is not None:
            self._reset_dialog.open = False
            self.page.update()
    
    async def _on_reset_confirm(self, e):
        """Обработчик подтверждения сброса"""
        self._dismiss_reset_dialog()
        await self._do_reset()
    
    async def _do_reset(self):
        """Полный сброс базы данных, кэша и состояния приложения"""
        if self.scanning:
            await self._cancel_scan_workers()
            self.start_scan_btn.content = tr("scan.button.start")
            self.start_scan_btn.icon = ft.Icons.PLAY_ARROW
            self.start_scan_btn.bgcolor = self._WARM_PRIMARY
            self.start_scan_btn.color = self._WARM_ON_PRIMARY
        self._cancel_prewarm()
        self._reset_window_progress()
        
        database.clear_all()
        thumbnail_cache.clear()
        search.clear_cache()
        
        self.images = []
        self.clusters = {}
        self.cluster_names = {}
        self.search_results = []
        self.selected_images = set()
        self.current_gallery_paths = []
        self.current_gallery_scope = "overview"
        self.current_tab = -1
        self.active_cluster_id = None
        self._invalidate_cluster_cache()
        
        self.scan_path_input.value = tr("scan.all_disks")
        self.search_input.value = ""
        self.export_dest_folder.value = tr("export.default_folder")
        self._export_dest_folder_path = None
        setattr(self.page.session, "export_dest_folder", None)
        
        self.stat_total.value = "0"
        self.stat_total_size.value = "0 " + tr("unit.B")
        self.stat_duplicates.value = "0"
        self.stat_duplicates_size.value = "0 " + tr("unit.B")
        self.stat_unique.value = "0"
        self.stat_unique_size.value = "0 " + tr("unit.B")
        self.stat_selected.value = "0"
        self.stat_selected_size.value = "0 " + tr("unit.B")

        # Reset selection-related UI controls to unselected state
        if hasattr(self, 'select_all_icon_button'):
            self.select_all_icon_button.icon = ft.Icons.CHECK_BOX_OUTLINE_BLANK
        if hasattr(self, 'cluster_select_all_icon_button'):
            self.cluster_select_all_icon_button.icon = ft.Icons.CHECK_BOX_OUTLINE_BLANK
        if hasattr(self, 'selected_count_text'):
            self.selected_count_text.value = tr("selected.count", count=0)
        
        self.update_clusters_list()
        
        self.tab_content.content = ft.Text(tr("categories.empty"))
        
        self.switch_mode("scan")
        self.page.update()
        self.show_snackbar(tr("reset.done"), "#81C784")

    def get_selection_state(self, paths: list) -> str:
        """Определить состояние выделения для иконки чекбокса.
        
        Returns:
            "all" - все выбраны
            "none" - ни один не выбран
            "partial" - частично выбраны
        """
        if not paths:
            return "none"
        
        selected_count = sum(1 for p in paths if p in self.selected_images)
        if selected_count == 0:
            return "none"
        elif selected_count == len(paths):
            return "all"
        else:
            return "partial"
    
    def get_checkbox_icon(self, state: str) -> str:
        """Получить иконку чекбокса в зависимости от состояния."""
        if state == "all":
            return ft.Icons.CHECK_BOX
        elif state == "none":
            return ft.Icons.CHECK_BOX_OUTLINE_BLANK
        else:  # partial
            return ft.Icons.INDETERMINATE_CHECK_BOX
    
    async def show_export_tab(self):
        """Показать режим 'Экспорт' — выбранные файлы в галерее (контролы на панели управления)."""
        # Сброс lazy loading state для export scope при переключении вкладки
        self._reset_gallery_lazy_loading("export")
        
        # Выбранные файлы — пути берём из БД (без обращения к файловой системе,
        # чтобы переключение на экспорт не зависало на внешнем диске).
        selected_paths = sorted(self.selected_images)

        # Сохраняем текущий контекст галереи
        self.current_gallery_paths = selected_paths
        self.current_gallery_scope = "export"

        if selected_paths:
            # Показываем индикатор загрузки пока генерируются миниатюры
            self.tab_content.content = ft.Row(
                [ft.ProgressRing(width=32, height=32)],
                alignment=ft.MainAxisAlignment.CENTER,
            )
            self.page.update()
            gallery = await self.create_gallery(selected_paths, "export")
            content = ft.Column([gallery], expand=True)
        else:
            content = ft.Column(
                [ft.Text(tr("export.none"), size=14, color=self._WARM_ON_SURFACE_VARIANT)],
                expand=True,
            )

        self.tab_content.content = content
        self.page.update()
        # Восстанавливаем позицию скролла при смене вкладки на экспорт
        if selected_paths:
            page_size = self._calc_grid_size(len(selected_paths))
            self._restore_gallery_scroll(gallery, "export", selected_paths, page_size)
    async def _run_export(self):
        """Экспорт выбранных файлов в одну папку без разделения по категориям"""
        selected_paths = sorted(self.selected_images)
        if not selected_paths:
            self.show_snackbar(tr("export.none_selected"), "#FFB74D")
            return
        
        self.exporting = True
        self._export_stop_requested = False
        self.export_button.content = tr("export.button.stop")
        self.export_button.icon = ft.Icons.STOP
        self.export_button.bgcolor = self._WARM_ERROR
        
        dest_folder = self.export_dest_folder.value if hasattr(self, 'export_dest_folder') else tr("export.default_folder")
        self._export_dest_folder_path = dest_folder
        setattr(self.page.session, 'export_dest_folder', dest_folder)
        dest_path = Path(dest_folder).expanduser()
        dest_path.mkdir(parents=True, exist_ok=True)
        
        total_files = len(selected_paths)
        total_copied = 0
        
        self._set_window_progress(tr("stage.export"))

        used_names = set()
        
        def copy_files():
            nonlocal total_copied
            for i, file_path in enumerate(selected_paths):
                if self._export_stop_requested:
                    break
                try:
                    dst = _unique_path(dest_path, Path(file_path).name, used_names)
                    shutil.copy2(file_path, dst)
                    used_names.add(dst.name)
                    total_copied += 1
                except Exception as ex:
                    print(f"Ошибка копирования {file_path}: {ex}")
                
                if (i + 1) % 10 == 0 or (i + 1) == total_files:
                    progress = (i + 1) / total_files
                    async def _update_progress(cur=i+1, tot=total_files, p=progress):
                        self._update_export_progress(cur, tot, p)
                    self.page.run_task(_update_progress)
        
        await asyncio.to_thread(copy_files)
        
        # Сбрасываем UI только если экспорт не был остановлен
        if self.exporting:
            self.exporting = False
            self.export_button.content = tr("export.button")
            self.export_button.icon = ft.Icons.DOWNLOAD
            self.export_button.bgcolor = self._WARM_PRIMARY
            self.export_button.color = self._WARM_ON_PRIMARY
            if not self.scanning:
                self._reset_window_progress()
            self.page.update()
            self.show_snackbar(tr("export.copied", count=total_copied, dest=dest_path), "#81C784")

    def _update_export_progress(self, current: int, total: int, progress: float):
        self._set_window_progress(tr("stage.export"), current, total)

    async def show_clusters_tab(self):
        """Показать вкладку 'Категории'"""
        # Сброс lazy loading state для cluster scope при переключении вкладки
        self._reset_gallery_lazy_loading("cluster")
        
        if not self.clusters:
            self.tab_content.content = ft.Text(tr("categories.empty"))
            self.page.update()
            return
        
        # Получить выбранную категорию
        # Проверяем, что active_cluster_id всё ещё существует в clusters
        # (после повторной кластеризации старые ID могут исчезнуть)
        if self.active_cluster_id is not None and self.active_cluster_id in self.clusters:
            cluster_id = self.active_cluster_id
        else:
            cluster_id = sorted(self.clusters.keys())[0]
            self.active_cluster_id = cluster_id
        
        # Обновляем подсветку кнопок
        self.update_clusters_list()
        
        members = self.clusters.get(int(cluster_id), [])
        # Сортируем для стабильного порядка
        members.sort()
        
        # Кнопка "Выбрать/Снять все" с интерактивной иконкой
        selection_state = self.get_selection_state(members)
        self.cluster_select_all_icon_button.icon = self.get_checkbox_icon(selection_state)
        self.cluster_select_all_icon_button.tooltip = tr("select_all.toggle")
        self.cluster_select_all_icon_button.on_click = lambda e: self.toggle_select_all(members)

        # Счётчик выбранных
        self.selected_count_text.value = tr("selected.count", count=len([p for p in members if p in self.selected_images]))

        # Сохраняем текущий контекст галереи
        self.current_gallery_paths = members
        self.current_gallery_scope = f"cluster_{cluster_id}"

        # Показываем индикатор загрузки пока генерируются миниатюры
        self.tab_content.content = ft.Row(
            [ft.ProgressRing(width=32, height=32)],
            alignment=ft.MainAxisAlignment.CENTER,
        )
        self.page.update()

        gallery = await self.create_gallery(members, f"cluster_{cluster_id}")

        self.tab_content.content = ft.Column(
            [gallery],
            expand=True,
        )
        self.page.update()
        # Восстанавливаем позицию скролла при смене вкладки на категории
        self._restore_gallery_scroll(gallery, f"cluster_{cluster_id}", members, 100)
    
    def show_search_tab(self):
        """Показать режим 'Поиск' (контролы поиска на панели управления, здесь только результаты)."""
        # Сброс lazy loading state для search_results scope при переключении вкладки
        self._reset_gallery_lazy_loading("search_results")
        
        # Сохраняем текущий контекст галереи
        self.current_gallery_scope = "search_results"
        self.current_gallery_paths = getattr(self, 'search_result_paths', [])

        self.tab_content.content = self.search_results_container
        self.page.update()
    
    async def do_search(self, e):
        """Выполнить поиск"""
        query = self.search_input.value
        
        if not query:
            self.show_snackbar(tr("search.empty_query"), "#FFB74D")
            return
        
        try:
            # Показываем индикатор загрузки
            self.search_results_container.controls = [
                ft.Container(
                    content=ft.ProgressRing(width=32, height=32),
                    alignment=ft.Alignment.CENTER,
                    expand=True,
                ),
            ]
            self.page.update()
            
            # Сбрасываем offset, scroll и lazy offset для результатов поиска
            session_key = "gallery_offset_search_results"
            setattr(self.page.session, session_key, 0)
            setattr(self.page.session, "gallery_scroll_search_results", 0)
            
            # Поиск в фоне. top_k (SEARCH_TOP_K) — защитный максимум кандидатов,
            # а не жёсткое ограничение результата: количество решает порог
            # SEARCH_THRESHOLD (все, кто прошёл порог, попадают в галерею).
            results = await asyncio.to_thread(
                search.run,
                query,
                top_k=config.SEARCH_TOP_K
            )
            
            if results:
                # Сортируем результаты по близости (от наиболее похожих к менее)
                paths = [p for p, _ in results]  # results уже отсортированы по score в core/search.py
                self.search_result_paths = paths
                self.current_gallery_paths = paths
                gallery = await self.create_gallery(paths, "search_results")
                
                self.search_results_container.controls = [
                    gallery,
                ]
            else:
                self.search_result_paths = []
                self.current_gallery_paths = []
                self.search_results_container.controls = [
                    ft.Text(tr("search.none"), size=14),
                ]
            
            self.page.update()
            if results:
                self._restore_gallery_scroll(gallery, "search_results", paths, 100)
        except Exception as e:
            self.search_results_container.controls = [
                ft.Text(tr("search.error", error=e), size=14, color=self._WARM_ERROR),
            ]
            self.page.update()
            self.show_snackbar(tr("search.error", error=e), self._WARM_ERROR)
    
    def _get_path_to_cluster_map(self, paths: list = None):
        """Возвращает dict {path: cluster_id} для изображений в кластерах.
        
        Оптимизировано с кэшированием - перестраивается только при изменении кластеров.
        
        Args:
            paths: если указан, возвращает map только для этих путей (дешевле,
                   особенно для export scope, где paths = только выбранные файлы).
        """
        # Быстрый путь: кластеры уже в памяти — фильтруем по нужным путям
        if self.clusters and paths is not None:
            target = set(str(p) for p in paths)
            path_to_cluster = {}
            for cluster_id, members in self.clusters.items():
                for p in members:
                    if p in target:
                        path_to_cluster[p] = cluster_id
            return path_to_cluster
        
        # Проверяем, нужно ли обновлять кэш (только для полной перестройки)
        if (self._cached_path_to_cluster is not None and 
            self._cached_path_to_cluster_scope == id(self.clusters)):
            if paths is None:
                return self._cached_path_to_cluster
            target = set(str(p) for p in paths)
            return {p: cid for p, cid in self._cached_path_to_cluster.items() if p in target}
        
        # Перестраиваем кэш из БД
        if paths is not None:
            target = set(str(p) for p in paths)
            placeholders = ",".join("?" for _ in target)
            rows = database.conn.execute(
                f"SELECT i.path, c.cluster_id FROM clusters c JOIN images i ON c.image_id = i.id WHERE i.path IN ({placeholders})",
                list(target),
            ).fetchall()
            path_to_cluster = {r["path"]: r["cluster_id"] for r in rows}
        else:
            clusters = self.clusters or database.load_clusters() or {}
            path_to_cluster = {}
            for cluster_id, members in clusters.items():
                for path in members:
                    path_to_cluster[path] = cluster_id
        
        self._cached_path_to_cluster = path_to_cluster
        self._cached_path_to_cluster_scope = id(self.clusters)
        return path_to_cluster
    
    def _invalidate_cluster_cache(self):
        """Инвалидировать кэш кластеров (вызывать при изменении кластеров)."""
        self._cached_path_to_cluster = None
        self._cached_path_to_cluster_scope = None

    def _make_gallery_item(self, path: str, scope: str, gallery: ft.GridView, path_to_cluster: dict = None, thumb_data: bytes = None, placeholder_mode: bool = False):
        """Создаёт элемент галереи (изображение + бейдж категории для поиска).

        thumb_data — bytes WebP-миниатюры из БД (см. thumbnail_cache.get_thumbnail);
        Flet 0.86.5 принимает raw bytes в src, поэтому temp-файлы не нужны.

        placeholder_mode — если True и миниатюры ещё нет, рисует серую плитку
        вместо ожидания; реальное изображение подставит _fill_page_thumbnails.
        """
        is_selected = path in self.selected_images
        
        # Используем предварительно сгенерированную миниатюру.
        # ВАЖНО: никакой синхронной генерации здесь быть не должно — этот метод
        # вызывается из UI-потока (load_more/_fill_page_thumbnails), и PIL-декод
        # оригинала на 50–150 мс за фото блокирует интерфейс. Отсутствующие
        # миниатюры всегда догружаются фоном через _fill_page_thumbnails/
        # _generate_thumbnails_parallel, а здесь просто рисуем плейсхолдер.
        if thumb_data is None and not placeholder_mode:
            placeholder_mode = True
        if thumb_data is not None:
            image = ft.Image(
                src=thumb_data,
                fit="contain",
                width=150,
                height=150,
                cache_width=150,
                cache_height=150,
                gapless_playback=True,
            )
        elif placeholder_mode:
            # Плейсхолдер — Container с возможностью показа фото через .image.
            # ВАЖНО: НЕ заменяем этот контрол в галерее позже. Замена
            # gallery.controls[i] меняет список детей GridView, и клиент видит
            # её только после полного обновления родителя (клик). Вместо этого
            # _fill_page_thumbnails присваивает holder.image = DecorationImage(...)
            # и патчит один этот контрол — список не трогается, миниатюра
            # появляется сразу.
            holder = ft.Container(
                width=150,
                height=150,
                bgcolor=ft.Colors.with_opacity(0.08, self._WARM_SURFACE),
            )
            image = holder
        else:
            # Fallback: плейсхолдер вместо загрузки оригинала.
            # Раньше здесь был src=path (полное декодирование JPEG 3648×2736
            # ради плитки 150×150) — это давало секунды фриза при скролле.
            image = ft.Container(
                width=150,
                height=150,
                bgcolor=ft.Colors.with_opacity(0.08, self._WARM_SURFACE),
            )
        
        # Для поиска показываем номер категории в углу
        if scope in ("search_results", "export") and path_to_cluster:
            cluster_id = path_to_cluster.get(path)
            if cluster_id is not None:
                display_num = getattr(self, '_cluster_display_map', {}).get(cluster_id, cluster_id)
                badge = ft.Container(
                    content=ft.Text(
                        str(display_num),
                        size=10,
                        weight=ft.FontWeight.NORMAL,
                        color=ft.Colors.with_opacity(0.7, self._WARM_ON_SURFACE_VARIANT),
                    ),
                    bgcolor=ft.Colors.with_opacity(0.4, self._WARM_SURFACE),
                    padding=ft.Padding(4, 1, 4, 1),
                    border_radius=6,
                )
                content = ft.Stack(
                    [
                        image,
                        ft.Container(
                            content=badge,
                            alignment=ft.Alignment.TOP_LEFT,
                            padding=ft.Padding(4, 4, 0, 0),
                        ),
                    ],
                    width=150,
                    height=150,
                )
            else:
                content = image
        else:
            content = image
        
        # Оборачиваем в GestureDetector для поддержки жестов
        # on_tap_down - левый клик (выделение + рамка)
        # on_secondary_tap - правый клик (preview/zoom)
        def on_tap_down(e):
            self.toggle_image_selection(path, gallery)
        
        def on_secondary_tap(e):
            self.show_preview(path)
        
        gesture = ft.GestureDetector(
            content=ft.Container(
                content=content,
                border_radius=8,
                border=ft.Border.all(4, "#64B5F6") if is_selected else None,
                data=path,  # <-- важно: храним оригинальный путь для поиска контейнера
            ),
            on_tap_down=on_tap_down,
            on_secondary_tap=on_secondary_tap,
        )

        # Для плейсхолдера запоминаем "держатель" изображения, чтобы
        # _fill_page_thumbnails мог подставить миниатюру, не заменяя контрол.
        if placeholder_mode:
            gesture._thumb_holder = image
        return gesture

    def _make_loading_indicator(self) -> ft.Container:
        """Индикатор загрузки в конце галереи (пока подгружаются следующие фото).

        Отображается как ячейка сетки с ProgressRing по центру. Помечаем
        через data, чтобы отличать от обычных изображений и правильно
        управлять им при lazy loading.
        """
        return ft.Container(
            content=ft.ProgressRing(width=40, height=40, stroke_width=4),
            width=150,
            height=150,
            alignment=ft.Alignment.CENTER,
            border_radius=8,
            data="_loading_indicator",
        )

    def _find_loading_indicator(self, gallery: ft.GridView):
        """Возвращает индекс ячейки-индикатора загрузки в controls, либо None."""
        if gallery is None:
            return None
        for i, control in enumerate(gallery.controls):
            if isinstance(control, ft.Container) and control.data == "_loading_indicator":
                return i
        return None

    def _calc_grid_columns(self) -> int:
        """Вычисляет количество столбцов сетки (runs_count) под ширину окна."""
        window_w = getattr(self.page, 'width', 1200) or 1200
        sidebar_w = 280
        page_padding = 20 * 2      # горизонтальный padding страницы
        spacing_w = 20             # spacing между sidebar и main
        available_w = max(200, window_w - sidebar_w - page_padding - spacing_w)

        # Размер миниатюры + отступ
        item_size = 150 + 5        # max_extent + spacing

        cols = max(1, int(available_w / item_size))
        return min(cols, 8)        # ограничиваем максимум 8 столбцами

    def _calc_grid_size(self, path_count: int = None) -> int:
        """Вычисляет количество изображений, помещающихся в видимой области окна.

        Учитывает ширину боковой панели, отступы страницы и высоту
        заголовка/контролов. Возвращает количество миниатюр, которые
        заполняют окно целиком (с запасом для плавной прокрутки).
        """
        window_w = getattr(self.page, 'width', 1200) or 1200
        window_h = getattr(self.page, 'height', 800) or 800

        # Ширина, доступная под галерею: окно - sidebar - padding страницы - spacing
        sidebar_w = 280
        page_padding = 20 * 2      # горизонтальный padding страницы
        spacing_w = 20             # spacing между sidebar и main
        available_w = max(200, window_w - sidebar_w - page_padding - spacing_w)

        # Высота, доступная под галерею: окно - padding - заголовок/контролы (~200px)
        header_h = 200             # scan section + tabs + controls
        available_h = max(200, window_h - page_padding - header_h)

        # Размер миниатюры + отступы
        item_size = 150 + 5        # max_extent + spacing
        run_size = 150 + 5         # child height + run_spacing

        cols = max(1, int(available_w / item_size))
        rows = max(1, int(available_h / run_size))

        per_page = cols * rows

        # Загружаем с запасом (в 3 раза), чтобы следующая порция была
        # уже сгенерирована до того, как пользователь докрутит до конца
        # (объёмная маленькая подгрузка по текущему viewport).
        if path_count is not None:
            per_page = min(per_page * 3, path_count)

        return max(per_page, 10)

    def _calc_lazy_window(self, path_count: int = None) -> int:
        """Возвращает число изображений для буфера прогноза предзагрузки.

        Для бесконечной галереи viewport-окно берётся из доступной области,
        а запоминание порога перекрывает беспокойство на окончание списка
        тем, что следующая страница подготавливается заранее.
        """
        visible_count = self._calc_grid_size(path_count)
        # Ставим предзагрузку на 2 viewport-а вперёд, но не больше
        # фактического числа элементов в наборе.
        if path_count is not None:
            return min(max(visible_count * 2, 10), path_count)
        return max(visible_count * 2, 10)

    async def _generate_thumbnails_parallel(self, paths: list, size: int = 150, max_workers: int = 12) -> dict:
        """Пакетное получение миниатюр (BLOB WebP) для списка путей.

        Оптимизация по сравнению с чистой per-path генерацией:
          1. все os.stat выполняются пакетно и дёшево;
          2. готовые миниатюры читаются из БД ОДНИМ SQL-запросом;
          3. в потоках (asyncio.to_thread, ограничено semaphore) генерируются
             только отсутствующие WebP — с сохранением в БД на будущее.

        Returns:
            dict {path: bytes | None}
        """
        if not paths:
            return {}

        # 1) Метаданные файлов. os.stat кэшируется: одни и те же файлы статятся
        # на КАЖДОЙ странице галереи, а на macOS stat может внезапно блокироваться
        # (Spotlight, iCloud, сброс кэша inode) — в логах до 5 с на 30 файлов.
        # mtime/size не меняются между страницами, поэтому TTL-кэш безопасен:
        # устаревшая запись максимум META_TTL секунд приведёт лишь к лишней
        # перегенерации миниатюры позже.
        now_ts = __import__('time').monotonic()
        cache = getattr(self, "_meta_cache", None)
        if cache is None:
            cache = self._meta_cache = {}

        def _collect_meta():
            meta = {}
            for p in paths:
                hit = cache.get(p)
                if hit is not None and now_ts - hit[2] < config.META_CACHE_TTL:
                    meta[p] = (hit[0], hit[1])
                    continue
                try:
                    st = os.stat(p)
                    meta[p] = (st.st_mtime, st.st_size)
                    cache[p] = (st.st_mtime, st.st_size, __import__('time').monotonic())
                except OSError:
                    meta[p] = None
            return meta

        meta = await asyncio.to_thread(_collect_meta)

        existing = [p for p in paths if meta.get(p) is not None]
        meta_map = {p: meta[p] for p in existing}
        result = {p: None for p in paths}

        if not existing:
            return result

        # 2) Батч-чтение уже готовых миниатюр из БД (1 запрос вместо N).
        db_thumbs = await asyncio.to_thread(
            thumbnail_cache.get_thumbnails_for_paths, existing, meta_map
        )
        result.update(db_thumbs)

        # 3) Генерация только отсутствующих (в потоках, с сохранением в БД).
        missing = [p for p in existing if p not in db_thumbs]
        if missing:
            semaphore = asyncio.Semaphore(max_workers)

            async def generate_one(path):
                async with semaphore:
                    blob = await asyncio.to_thread(thumbnail_cache.get_thumbnail, path, size)
                    return path, blob

            generated = await asyncio.gather(*[generate_one(p) for p in missing])
            result.update(dict(generated))

        return result

    def _generate_thumbnails_batch(self, paths: list, size: int = 150) -> dict:
        """Синхронная пакетная генерация миниатюр (для фонового потока).

        Returns:
            dict {path: bytes | None}
        """
        result = {}
        for path in paths:
            result[path] = thumbnail_cache.get_thumbnail(path, size=size)
        return result

    def _preload_gallery_cache(self, paths: list, scope: str):
        """Предварительно прогревает кэш миниатюр для портянки фото.

        Важно: предзагрузка не создаёт новые контролы GridView, а только
        наполняет thumbnail_cache фоновыми генерациями миниатюр. Это даёт
        гладкую прокрутку без вставки тысяч виджетов за один раз.
        """
        loading_key = f"gallery_preloading_{scope}"
        if getattr(self.page.session, loading_key, False):
            return
        setattr(self.page.session, loading_key, True)

        async def preload():
            try:
                if not paths:
                    return
                for idx in range(0, len(paths), 100):
                    batch = paths[idx:idx + 100]
                    await self._generate_thumbnails_parallel(batch, 150, max_workers=config.PREWARM_THUMB_WORKERS)
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                setattr(self.page.session, loading_key, False)

        asyncio.create_task(preload())

    async def create_gallery(self, paths: list, scope: str):
        """Создание галереи с lazy loading.
        
        Генерация миниатюр выполняется в фоновом потоке через asyncio.to_thread,
        чтобы не блокировать UI при переключении категорий. Сразу после создания
        первой порции запускается фоновая предзагрузка всех остальных миниатюр.
        """
        # Validate paths
        if not paths or not isinstance(paths, list):
            return ft.Column([ft.Text(tr("gallery.empty"))])
        
        # Загружаем выделение из БД (всегда используем глобальный scope)
        saved_selection = database.load_selected_files(scope="global")
        if saved_selection:
            self.selected_images.update(saved_selection)
        
        # Обновляем статистику выбранных в боковой панели (в фоновом потоке)
        if hasattr(self, 'stat_selected'):
            self._update_selected_stats_async()
        
        # Для поиска строим map path -> cluster_id (один раз).
        # Сохраняем в self._gallery_path_to_cluster, чтобы load_more не
        # перестраивал эту дорогую карту на каждом скролле.
        if scope == "search_results":
            self._gallery_path_to_cluster = self._get_path_to_cluster_map(paths)
            path_to_cluster = self._gallery_path_to_cluster
        else:
            self._gallery_path_to_cluster = None
            path_to_cluster = None
        
        # Состояние для lazy loading.
        # Важный фикс: после открытия новой галереи (кластер, поиск, export)
        # смещение должно стартовать с нуля. Иначе повторное открытие scope
        # может взять старый offset и срезать список в произвольной точке.
        session_key = f"gallery_offset_{scope}"
        setattr(self.page.session, session_key, 0)
        setattr(self.page.session, f"gallery_loading_{scope}", False)
        setattr(self.page.session, f"gallery_filling_{scope}", False)

        # Динамический page_size: заполняем видимую область окна целиком
        # (с запасом ×3, чтобы следующая порция была готова заранее)
        page_size = self._calc_grid_size(len(paths))
        offset = 0

        try:
            page_paths = paths[offset:offset + page_size]
        except (TypeError, ValueError):
            page_size = self._calc_grid_size(len(paths))
            page_paths = paths[:page_size]
            offset = 0
        
        # Мгновенный показ: строим галерею сразу, без ожидания генерации.
        # Реальные миниатюры подставятся фоновой задачей _fill_page_thumbnails.
        # Создаём GridView с динамическим числом столбцов под ширину окна
        gallery = ft.GridView(
            expand=True,
            runs_count=self._calc_grid_columns(),
            child_aspect_ratio=1.0,
            max_extent=150,
            spacing=5,
            run_spacing=5,
            scroll=ft.ScrollMode.AUTO,
        )
        
        # Передаём gallery в обработчик скролла
        gallery.on_scroll = self.create_scroll_handler(paths, scope, page_size, gallery)
        
        # Добавляем плейсхолдеры (минимальная работа в UI-потоке — только создание контролов)
        for path in page_paths:
            gallery.controls.append(
                self._make_gallery_item(path, scope, gallery, path_to_cluster, placeholder_mode=True)
            )
        
        # Обновляем offset строго на количество реально загруженных путей.
        # Если этого не сделать, offset останется 0, и первый же скролл
        # в load_more заново подгрузит paths[0:page_size] → дубли путей в галерее.
        setattr(self.page.session, session_key, len(page_paths))

        # Показываем индикатор загрузки в конце галереи, если остались
        # ещё не загруженные фото. Он будет заменён/передвинут при подгрузке.
        if offset + len(page_paths) < len(paths):
            gallery.controls.append(self._make_loading_indicator())

        # Фоновая подстановка миниатюр на место плейсхолдеров первого экрана.
        # Дальнейшие миниатюры подгружаются лениво при скролле.
        asyncio.create_task(self._fill_page_thumbnails(page_paths, gallery, scope))

        return gallery

    async def _fill_page_thumbnails(self, page_paths: list, gallery: ft.GridView, scope: str, max_workers: int = 12):
        """Подменяет серые плейсхолдеры первого экрана реальными миниатюрами.

        Выполняется в фоне после мгновенного открытия галереи: батч-чтение из
        БД + генерация только недостающих WebP, затем замена controls по индексам
        (плейсхолдеры занимают первые len(page_paths) позиций в GridView).
        """
        if not page_paths or gallery is None:
            return
        filling_key = f"gallery_filling_{scope}"
        setattr(self.page.session, filling_key, True)
        try:
            thumbs = await self._generate_thumbnails_parallel(page_paths, 150, max_workers=max_workers)
            # path_to_cluster не нужен: патчим только существующий плейсхолдер
            pending = []
            for i, path in enumerate(page_paths):
                thumb = thumbs.get(path)
                if thumb is None:
                    continue
                gesture = gallery.controls[i] if i < len(gallery.controls) else None
                holder = getattr(gesture, "_thumb_holder", None)
                if holder is None:
                    continue
                # Подставляем фото в СУЩЕСТВУЮЩИЙ контрол, не заменяя его в списке.
                # Так не меняется список детей GridView → клиент видит миниатюру
                # сразу (иначе — только после полного обновления родителя/клика).
                holder.image = ft.DecorationImage(src=thumb, fit="contain")
                holder.bgcolor = None
                pending.append(gesture)
            if pending:
                # Патчим только изменённые контролы — O(порция), список не трогаем
                self.page.update(*pending)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            setattr(self.page.session, filling_key, False)
    
    def create_scroll_handler(self, paths: list, scope: str, page_size: int, gallery: ft.GridView):
        """Создать обработчик скролла для lazy loading (инкрементальная автозагрузка).

        Подгружает следующую порцию только когда пользователь реально подошёл
        к нижней границе уже видимой части галереи. Признак: расстояние до
        конца меньше примерно одной порции плюс запас для плавности.
        """
        loading_key = f"gallery_loading_{scope}"
        filling_key = f"gallery_filling_{scope}"
        # Троттлинг: GridView генерирует десятки ScrollEvent в секунду.
        # Каждый event — отдельная задача в event-loop; без троттлинга они
        # накапливаются и блокируют loop на секунды во время прокрутки.
        throttle_state = {"last": 0.0}

        def on_scroll(e: ft.ScrollEvent):
            # КЛЮЧЕВЫЕ ФИКС: Flet после КАЖДОГО обработанного события вызывает
            # автообновление ближайшего изолированного предка — у нас GridView
            # не изолирован, поэтому диффится вся Page (Page id=1 took 65–110 ms
            # на каждое срабатывание; события скролла идут десятками в секунду →
            # event-loop работает вхолостую). Отключаем автообновление для хендлера:
            # страницу/галерею мы патчим сами (patch_control новых контролов).
            try:
                ft.context.disable_auto_update()
            except Exception:
                pass
            try:
                import time as _time
                now = _time.monotonic()
                if now - throttle_state["last"] < 0.15:
                    return
                throttle_state["last"] = now
                # Пока идёт первичная заливка первого экрана (_fill_page_thumbnails),
                # скролл-подгрузку не стартуем: две тяжёлые задачи одновременно
                # конкурируют за event-loop и дают фризы интерфейса.
                if getattr(self.page.session, filling_key, False):
                    return
                max_scroll = getattr(e, 'max_scroll_extent', 0) or 0
                pixels = getattr(e, 'pixels', 0) or 0
                if max_scroll <= 0:
                    return

                # Нормализуем загрузочное окно под текущее окно.
                # На первом вызове page_size из create_gallery уже решён,
                # но при resize окна он устаревает, поэтому пересчитываем.
                dynamic_page_size = self._calc_grid_size(len(paths))
                prefetch_items = self._calc_lazy_window(len(paths))
                # Ранний prefetch: стартуем за ~2 страницы до конца списка,
                # чтобы следующая порция была готова ДО того, как пользователь
                # докрутит до спиннера (блобы из БД читаются за мс, генерация
                # отсутствующих идёт в фоне).
                prefetch_px = max(dynamic_page_size * 150 * 2, 1200)
                remaining = max_scroll - pixels

                # Если пользователь ещё находится далеко от нижней границы,
                # ничего не делаем.
                if remaining > prefetch_px:
                    return

                session_key = f"gallery_offset_{scope}"
                offset = getattr(self.page.session, session_key, 0)
                if offset is None or not isinstance(offset, (int, float)):
                    offset = 0
                offset_int = int(offset)

                paths_len = int(len(paths)) if paths is not None else 0

                # Все элементы уже загружены.
                # Важно: НЕ сравниваем offset + page_size, иначе последняя
                # неполная порция (меньше одной страницы) никогда не подгрузится:
                # спиннер в конце галереи останется крутиться вечно.
                if offset_int >= paths_len:
                    return

                # Защита от повторной загрузки: при наличии уже запущенной
                # задачки мы не начинаем новую и не перетаскиваем UI.
                is_loading = getattr(self.page.session, loading_key, False)
                if is_loading:
                    return

                # Отслеживаем, что буфер позволяет не ждать на последней "кадре".
                # if current progress from page_count to prefetch_items? no block.
                setattr(self.page.session, f"gallery_prefetch_window_{scope}", prefetch_items)

                # Загружаем следующую порцию
                setattr(self.page.session, loading_key, True)
                asyncio.create_task(self._load_more_with_cleanup(paths, scope, gallery, loading_key))

            except Exception as ex:
                print(f"Scroll error: {ex}")
                import traceback
                traceback.print_exc()

        return on_scroll
    
    async def _load_more_with_cleanup(self, paths: list, scope: str, gallery: ft.GridView, loading_key: str):
        try:
            await self.load_more(paths, scope, gallery)
        finally:
            setattr(self.page.session, loading_key, False)
    
    async def load_more(self, paths: list, scope: str, gallery: ft.GridView):
        """Загрузка следующей порции (Infinite Scroll) - упрощённая версия для Flet 0.86.
        
        Генерация миниатюр выполняется в фоновом потоке через asyncio.to_thread,
        чтобы не блокировать UI при прокрутке.
        """
        try:
            # Validate inputs
            if not paths or not isinstance(paths, list):
                print("[load_more] invalid paths")
                return
            if gallery is None:
                print("[load_more] gallery is None")
                return
            
            # Используем тот же динамический page_size что и в create_gallery
            # (заполняет видимую область окна, пересчитывается при resize)
            page_size = self._calc_grid_size(len(paths))

            session_key = f"gallery_offset_{scope}"
            offset = getattr(self.page.session, session_key, 0)

            # Ensure offset is a valid integer
            if offset is None or not isinstance(offset, (int, float)):
                offset = 0
            offset = int(offset)

            if offset < 0 or offset > len(paths):
                print(f"[load_more] invalid offset={offset}, len={len(paths)}")
                offset = 0
                setattr(self.page.session, session_key, 0)

            if offset >= len(paths):
                idx = self._find_loading_indicator(gallery)
                if idx is not None:
                    gallery.controls.pop(idx)
                    self.page.update()
                return

            # Грузим только следующую часть от актуального offset, а не перескакиваем
            # на +page_size в индексе, иначе при возврате к галерее open/restore
            # можно пропустить начало или получить дырки в списке.
            page_paths = paths[offset:offset + page_size]
            if not page_paths:
                return

            # Для поиска используем карту кластеров, построенную один раз
            # в create_gallery (не перестраиваем её на каждом скролле).
            path_to_cluster = self._gallery_path_to_cluster if scope == "search_results" else None

            # Генерируем миниатюры параллельно в фоновом режиме (12 потоков)
            thumbs = await self._generate_thumbnails_parallel(page_paths, 150, max_workers=12)

            # Находим индикатор загрузки (он всегда в конце галереи).
            # Новые элементы вставляем ПЕРЕД ним, чтобы индикатор оставался
            # внизу, пока есть ещё не догруженные фото.
            indicator_idx = self._find_loading_indicator(gallery)
            if indicator_idx is None:
                indicator_idx = len(gallery.controls)

            # ВАЖНО (производительность): gallery.update()/page.update() без
            # аргументов вычисляют дифф рекурсивно по всему поддереву GridView —
            # O(все загруженные элементы) на каждый чанк. На глубоком скролле
            # это секунды Python-CPU в event-loop. Поэтому патчим ТОЛЬКО новые
            # контролы через page.update(*new_items) — O(порция), константа на
            # любой глубине.
            new_items = []
            insert_at = indicator_idx
            for path in page_paths:
                item = self._make_gallery_item(path, scope, gallery, path_to_cluster, thumbs.get(path))
                gallery.controls.insert(insert_at, item)
                new_items.append(item)
                insert_at += 1

            self.page.update(*new_items)
            # Обновляем offset строго по фактически загруженному количеству.
            loaded_count = len(page_paths)
            new_offset = offset + loaded_count
            setattr(self.page.session, f"gallery_offset_{scope}", new_offset)

            # Если загружены уже все фото — убираем индикатор загрузки из конца.
            # Иначе оставляем его на месте: следующая порция будет подгружена
            # при очередном скролле.
            if new_offset >= len(paths):
                idx = self._find_loading_indicator(gallery)
                if idx is not None:
                    gallery.controls.pop(idx)
                    gallery.update()
        except Exception as ex:
            print(f"Load more error: {ex}")
            import traceback
            traceback.print_exc()
            try:
                setattr(self.page.session, f"gallery_offset_{scope}", 0)
            except:
                pass
    
    def toggle_image_selection(self, path: str, gallery: ft.GridView = None):
        """Переключение выбора изображения"""
        if path in self.selected_images:
            self.selected_images.remove(path)
        else:
            self.selected_images.add(path)
        
        # Сохраняем выделение в БД
        database.save_selected_files(list(self.selected_images), scope="global")
        
        # Обновляем визуальную рамку у конкретного контейнера
        updated = False
        if gallery:
            for control in gallery.controls:
                # Элементы галереи обёрнуты в GestureDetector
                container = control.content if isinstance(control, ft.GestureDetector) else control
                if isinstance(container, ft.Container) and container.data == path:
                    is_selected = path in self.selected_images
                    container.border = ft.Border.all(4, "#64B5F6") if is_selected else None
                    updated = True
                    break
        
        # Обновляем счётчик выбранных
        if hasattr(self, 'selected_count_text'):
            selected_in_view = len([p for p in self.current_gallery_paths if p in self.selected_images])
            self.selected_count_text.value = tr("selected.count", count=selected_in_view)
        
        # Обновляем статистику выбранных в боковой панели (из БД, мгновенно)
        if hasattr(self, 'stat_selected'):
            self._update_selected_stats_async()
        
        # Обновляем иконку "Выбрать/Снять все"
        if self.current_gallery_scope == "overview" and hasattr(self, 'select_all_icon_button'):
            state = self.get_selection_state(self.current_gallery_paths)
            self.select_all_icon_button.icon = self.get_checkbox_icon(state)
        elif self.current_gallery_scope.startswith("cluster_") and hasattr(self, 'cluster_select_all_icon_button'):
            state = self.get_selection_state(self.current_gallery_paths)
            self.cluster_select_all_icon_button.icon = self.get_checkbox_icon(state)
        
        self.update_clusters_list()
        self.page.update()
    
    def select_all(self, paths: list, value: bool):
        """Выбрать все / снять все"""
        if value:
            self.selected_images.update(paths)
        else:
            self.selected_images.difference_update(paths)
        
        # Сохраняем выделение в БД
        database.save_selected_files(list(self.selected_images), scope="global")
        
        self.update_clusters_list()
        
        self.show_snackbar(tr("selected.files", count=len(self.selected_images)), self._WARM_PRIMARY)
        
        # Обновляем статистику выбранных в боковой панели (в фоновом потоке)
        if hasattr(self, 'stat_selected'):
            self._update_selected_stats_async()
        
        # Обновляем текущую вкладку
        if self.current_tab == -1:
            asyncio.create_task(self.show_clusters_tab())
        elif self.current_tab == 0:
            self.show_search_tab()
        elif self.current_tab == 1:
            asyncio.create_task(self.show_export_tab())
    
    def _clear_preview_state(self):
        self._preview_paths = []
        self._preview_idx = 0
        self._preview_image = None
        self._preview_image_container = None
        self._preview_stack = None
        self._preview_scale = 1.0
        self._preview_pan_x = 0.0
        self._preview_pan_y = 0.0
        self._preview_base_w = 800
        self._preview_base_h = 600
        self._preview_path_text = None
        self._preview_counter_text = None
        self._preview_click_start_x = 0.0
        self._preview_panning = False
        self._preview_pan_distance = 0.0
        self._preview_last_pan_x = 0.0
        self._preview_last_pan_y = 0.0
        self._preview_current_path = None

    def _get_image_size(self, path: str):
        """Возвращает (width, height) изображения или None.

        Сначала читает размер из БД, чтобы не открывать файл на каждом
        вызове preview. Если в БД нет — делает fallback на PIL и сохраняет
        размер обратно в БД для последующих открытий.
        """
        try:
            size = database.get_image_size(path)
            if size:
                return size
        except Exception:
            pass
        try:
            from PIL import Image as PILImage
            with PILImage.open(path) as img:
                w, h = img.size
                try:
                    database.update_image_size(path, w, h)
                except Exception:
                    pass
                return w, h
        except Exception:
            return None

    def _calc_preview_size(self, image_size=None):
        w = getattr(self.page, 'width', 1200) or 1200
        h = getattr(self.page, 'height', 800) or 800
        max_w = max(400, w - 80)
        max_h = max(300, h - 140)
        limit_w = min(1000, max_w)
        limit_h = min(800, max_h)
        if image_size is not None:
            img_w, img_h = image_size
            if img_w > 0 and img_h > 0:
                scale = min(limit_w / img_w, limit_h / img_h, 1.0)
                return max(300, int(img_w * scale)), max(220, int(img_h * scale))
        return limit_w, limit_h

    def _update_preview_size(self):
        if self._preview_dialog is None or not self._preview_dialog.open:
            return
        image_size = self._get_image_size(self._preview_current_path) if self._preview_current_path else None
        new_w, new_h = self._calc_preview_size(image_size)
        self._preview_base_w = new_w
        self._preview_base_h = new_h
        if self._preview_image is not None:
            self._preview_image.width = new_w * self._preview_scale
            self._preview_image.height = new_h * self._preview_scale
        if self._preview_stack is not None:
            self._preview_stack.width = new_w
            self._preview_stack.height = new_h
        if self._preview_dialog_content is not None:
            self._preview_dialog_content.content.width = new_w + 16
            self._preview_dialog_content.content.height = new_h + 52
        self.page.update()

    def _on_preview_dismissed(self, e):
        self._preview_dialog = None
        self._clear_preview_state()
        self.page.on_secondary_tap = None
        self.page.update()

    def _open_file_location(self, path: str):
        import subprocess
        import sys as _sys
        import threading
        import time
        folder = os.path.dirname(path)
        if _sys.platform == "win32":
            norm = os.path.normpath(os.path.abspath(path))
            try:
                import ctypes
                ctypes.windll.shell32.ShellExecuteW(
                    None,
                    "open",
                    "explorer.exe",
                    f"/select,{norm}",
                    None,
                    1,
                )
            except Exception:
                subprocess.Popen(["explorer.exe", f"/select,{norm}"])
            def _ensure_visible():
                time.sleep(1.0)
                try:
                    import win32gui
                    import win32con
                    import win32api
                    found = []
                    def cb(hwnd, extra):
                        if win32gui.IsWindowVisible(hwnd):
                            cls = win32gui.GetClassName(hwnd)
                            if cls in ("CabinetWClass", "ExploreWClass"):
                                found.append(hwnd)
                        return True
                    win32gui.EnumWindows(cb, None)
                    if found:
                        hwnd = found[-1]
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(hwnd)
                        win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_F5, 0)
                        win32api.SendMessage(hwnd, win32con.WM_KEYUP, win32con.VK_F5, 0)
                        win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_F6, 0)
                        win32api.SendMessage(hwnd, win32con.WM_KEYUP, win32con.VK_F6, 0)
                except Exception:
                    pass
            threading.Thread(target=_ensure_visible, daemon=True).start()
        elif _sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", folder])

    def _navigate_preview(self, delta):
        paths = self._preview_paths
        if not paths:
            return
        self._preview_idx = (self._preview_idx + delta) % len(paths)
        new_path = paths[self._preview_idx]
        self._preview_current_path = new_path
        self._preview_image.src = new_path
        self._preview_scale = 1.0
        self._preview_pan_x = 0.0
        self._preview_pan_y = 0.0
        self._preview_image.width = self._preview_base_w
        self._preview_image.height = self._preview_base_h
        self._preview_image_container.left = 0
        self._preview_image_container.top = 0
        if self._preview_path_text is not None:
            self._preview_path_text.value = new_path
            self._preview_path_text.tooltip = new_path
            self._preview_path_text.update()
        if self._preview_image_container is not None:
            self._preview_image_container.update()
        if self._preview_image is not None:
            self._preview_image.update()

    def _zoom_preview(self, factor, cx=None, cy=None):
        if cx is None:
            cx = self._preview_base_w / 2.0
        if cy is None:
            cy = self._preview_base_h / 2.0
        old_scale = self._preview_scale
        new_scale = max(0.5, min(5.0, old_scale * factor))
        if old_scale == new_scale:
            return
        self._preview_pan_x = cx - (cx - self._preview_pan_x) * (new_scale / old_scale)
        self._preview_pan_y = cy - (cy - self._preview_pan_y) * (new_scale / old_scale)
        self._preview_scale = new_scale
        self._preview_image.width = self._preview_base_w * new_scale
        self._preview_image.height = self._preview_base_h * new_scale
        self._preview_image_container.left = self._preview_pan_x
        self._preview_image_container.top = self._preview_pan_y
        now = time.perf_counter()
        if now - getattr(self, "_preview_last_zoom_update", 0.0) >= 0.016:
            self._preview_last_zoom_update = now
            self._preview_image.update()
            self._preview_image_container.update()

    def _reset_preview_zoom(self):
        self._preview_scale = 1.0
        self._preview_pan_x = 0.0
        self._preview_pan_y = 0.0
        self._preview_image.width = self._preview_base_w
        self._preview_image.height = self._preview_base_h
        self._preview_image_container.left = 0
        self._preview_image_container.top = 0
        self._preview_image.update()
        self._preview_image_container.update()

    def _on_preview_keyboard(self, e: ft.KeyboardEvent):
        dlg = self._preview_dialog
        if dlg is None or not dlg.open:
            return
        key = e.key
        if key == "Escape":
            dlg.open = False
            self._preview_dialog = None
            self._clear_preview_state()
            self.page.update()
        elif key == "Arrow Left":
            self._navigate_preview(-1)
        elif key == "Arrow Right":
            self._navigate_preview(1)
        elif key in ("+", "=", "Equal"):
            self._zoom_preview(1.2)
        elif key == "-" or key == "Minus":
            self._zoom_preview(1 / 1.2)
        elif key in ("0", "Digit0", "Numpad 0"):
            self._reset_preview_zoom()

    def show_preview(self, path: str):
        paths = self.current_gallery_paths or [path]
        if path not in paths:
            paths = [path] + paths
        current_idx = paths.index(path) if path in paths else 0

        image_size = self._get_image_size(path)
        base_w, base_h = self._calc_preview_size(image_size)

        preview_image = ft.Image(
            src=path,
            fit="contain",
            width=base_w,
            height=base_h,
        )

        image_container = ft.Container(
            content=preview_image,
            left=0,
            top=0,
        )

        stack = ft.Stack(
            [image_container],
            width=base_w,
            height=base_h,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

        self._preview_paths = paths
        self._preview_idx = current_idx
        self._preview_current_path = paths[current_idx] if paths else path
        self._preview_image = preview_image
        self._preview_image_container = image_container
        self._preview_stack = stack
        self._preview_base_w = base_w
        self._preview_base_h = base_h
        self._preview_scale = 1.0
        self._preview_pan_x = 0.0
        self._preview_pan_y = 0.0
        self._preview_click_start_x = 0.0
        self._preview_panning = False
        self._preview_pan_distance = 0.0
        self._preview_last_pan_x = 0.0
        self._preview_last_pan_y = 0.0

        path_text = ft.Text(
            path,
            size=12,
            color=ft.Colors.with_opacity(0.8, self._WARM_ON_SURFACE_VARIANT),
            text_align=ft.TextAlign.CENTER,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._preview_path_text = path_text

        def on_tap_down(e):
            lp = e.local_position
            self._preview_click_start_x = lp.x if lp else base_w / 2
            self._preview_panning = False
            self._preview_pan_distance = 0.0

        def on_pan_start(e):
            lp = e.local_position
            self._preview_last_pan_x = lp.x if lp else 0.0
            self._preview_last_pan_y = lp.y if lp else 0.0

        def on_pan_update(e):
            lp = e.local_position
            if lp is None:
                return
            dx = lp.x - self._preview_last_pan_x
            dy = lp.y - self._preview_last_pan_y
            self._preview_last_pan_x = lp.x
            self._preview_last_pan_y = lp.y
            if abs(dx) > 1 or abs(dy) > 1:
                self._preview_panning = True
            self._preview_pan_distance += abs(dx) + abs(dy)
            self._preview_pan_x += dx
            self._preview_pan_y += dy
            image_container.left = self._preview_pan_x
            image_container.top = self._preview_pan_y
            image_container.update()

        def on_pan_end(e):
            self._preview_panning = False

        def on_scroll(e):
            sd = e.scroll_delta
            if hasattr(sd, 'dy'):
                sd = sd.dy
            elif hasattr(sd, 'y'):
                sd = sd.y
            else:
                sd = float(sd)
            if sd == 0:
                return
            factor = 1.1 if sd > 0 else 1 / 1.1
            lp = e.local_position
            cx = lp.x if lp else base_w / 2.0
            cy = lp.y if lp else base_h / 2.0
            self._zoom_preview(factor, cx, cy)

        def on_tap(e):
            if not self._preview_panning or self._preview_pan_distance < 10:
                nav = -1 if self._preview_click_start_x < base_w / 2 else 1
                self._navigate_preview(nav)

        def on_secondary_tap(e):
            if self._preview_dialog is not None and self._preview_dialog.open:
                self._preview_dialog.open = False
                self._on_preview_dismissed(None)

        def on_dialog_secondary_tap(e):
            if self._preview_dialog is not None and self._preview_dialog.open:
                self._preview_dialog.open = False
                self._on_preview_dismissed(None)

        preview_gesture = ft.GestureDetector(
            content=stack,
            on_tap_down=on_tap_down,
            on_pan_start=on_pan_start,
            on_pan_update=on_pan_update,
            on_pan_end=on_pan_end,
            on_scroll=on_scroll,
            on_tap=on_tap,
        )

        dialog_w, dialog_h = self._calc_preview_size(image_size)
        path_text = ft.Text(
            path,
            size=11,
            color=ft.Colors.with_opacity(0.85, self._WARM_ON_SURFACE_VARIANT),
            text_align=ft.TextAlign.CENTER,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._preview_path_text = path_text
        path_text_container = ft.GestureDetector(
            content=ft.Container(
                content=ft.Row(
                    [
                        path_text,
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                padding=6,
                border_radius=6,
                on_click=lambda e: (
                    self._open_file_location(self._preview_current_path)
                    if self._preview_current_path and os.path.exists(self._preview_current_path)
                    else None
                ),
                tooltip=tr("preview.open_location"),
            ),
            on_secondary_tap=lambda e: (
                self._on_preview_dismissed(None)
                if self._preview_dialog is not None and self._preview_dialog.open
                else None
            ),
        )
        dialog_content = ft.GestureDetector(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.GestureDetector(
                            content=ft.Container(
                                content=preview_gesture,
                                alignment=ft.Alignment(0.0, 0.0),
                                padding=6,
                                expand=True,
                            ),
                            on_secondary_tap=on_secondary_tap,
                        ),
                        path_text_container,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=4,
                ),
                width=dialog_w + 40,
                height=dialog_h + 76,
                padding=12,
            ),
            on_secondary_tap=on_dialog_secondary_tap,
        )
        self._preview_dialog_content = dialog_content

        dialog = ft.AlertDialog(
            content=dialog_content,
            modal=False,
            title=None,
            actions=[],
            content_padding=ft.Padding(0),
            inset_padding=ft.Padding(0),
            title_padding=ft.Padding(0),
            actions_padding=ft.Padding(0),
            on_dismiss=self._on_preview_dismissed,
        )

        if self._preview_dialog is not None and self._preview_dialog in self.page.overlay:
            self._preview_dialog.open = False
            self._on_preview_dismissed(None)

        def on_page_secondary_tap(e):
            if self._preview_dialog is not None and self._preview_dialog.open:
                self._preview_dialog.open = False
                self._on_preview_dismissed(None)

        self.page.on_secondary_tap = on_page_secondary_tap
        self.page.overlay.append(dialog)
        self._preview_dialog = dialog
        dialog.open = True
        self.page.update()
    
    def show_context_menu(self, path: str, local_position=None):
        """Показать контекстное меню при правом клике на изображение"""
        
        def open_preview(e):
            self.page.update()
            self.show_preview(path)
        
        def open_location(e):
            self.page.update()
            folder = os.path.dirname(path)
            if os.path.exists(folder):
                import subprocess
                import sys as _sys
                if _sys.platform == "win32":
                    os.startfile(folder)
                elif _sys.platform == "darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
        
        # Создаём контейнер для позиционирования меню
        menu_container = ft.Container(
            content=ft.PopupMenuButton(
                items=[
                    ft.PopupMenuItem(
                        content=ft.Text(tr("context.zoom_in")),
                        icon=ft.Icons.ZOOM_IN,
                        on_click=open_preview,
                    ),
                    ft.PopupMenuItem(
                        content=ft.Text(tr("context.open_location")),
                        icon=ft.Icons.FOLDER_OPEN,
                        on_click=open_location,
                    ),
                ],
            ),
            # Позиционируем по координатам клика, если они переданы
            left=local_position.x if local_position and hasattr(local_position, 'x') else 100,
            top=local_position.y if local_position and hasattr(local_position, 'y') else 100,
        )
        
        # Добавляем в overlay для корректного отображения в десктопном режиме
        self.page.overlay.append(menu_container)
        
        # Открываем меню
        menu_container.content.open = True
        self.page.update()
    
    def browse_destination_folder(self, text_field: ft.TextField):
        """Открыть нативный проводник для выбора папки назначения."""
        path = self._pick_folder(tr("pick.export"))
        if path is not None:
            text_field.value = path
            self._export_dest_folder_path = path
            setattr(self.page.session, 'export_dest_folder', path)
            self.page.update()

    def _on_export_path_changed(self, e: ft.ControlEvent):
        """Сохраняем путь экспорта сразу при изменении текстового поля."""
        path = (e.control.value or "").strip()
        if path:
            self._export_dest_folder_path = path
            setattr(self.page.session, 'export_dest_folder', path)

    def toggle_select_all(self, paths: list):
        """Переключатель: если все выбраны - снимает все, иначе выбирает все"""
        all_selected = all(p in self.selected_images for p in paths)
        self.select_all(paths, not all_selected)
    
def _unique_path(folder, name, used_names):
    """Возвращает путь с уникальным именем, избегая коллизий."""
    if name not in used_names and not (folder / name).exists():
        return folder / name
    stem, suffix = os.path.splitext(name)
    i = 1
    while True:
        candidate = f"{stem}_{i}{suffix}"
        if candidate not in used_names and not (folder / candidate).exists():
            return folder / candidate
        i += 1


def main():
    import traceback
    try:
        ft.app(target=ImageDedupApp)
    except Exception as e:
        traceback.print_exc()
        print(f"Критическая ошибка при запуске: {e}")


if __name__ == "__main__":
    main()