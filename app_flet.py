import flet as ft
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path
import config
from core import database, scanner, dedup, embedder, clustererhdb, search, thumbnail_cache


def _format_size(size_bytes: int) -> str:
    """Форматирует размер в байтах в читаемый вид (КБ/МБ/ГБ)."""
    try:
        size_bytes = int(size_bytes)
    except (TypeError, ValueError):
        return "0 Б"
    if size_bytes < 1024:
        return f"{size_bytes} Б"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} КБ"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} МБ"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} ГБ"


class ImageDedupApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Image Deduplication"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.window_width = 1200
        self.page.window_height = 800
        self.page.padding = 20
        self.page.spacing = 10
        
        # Состояние приложения
        self.images = []
        self.clusters = {}
        self.cluster_names = {}
        self.search_results = []
        self.progress_visible = False
        self.progress_value = 0.0
        self.progress_label = ""
        self.model_loaded = False
        self.model_loading = False
        self.scanning = False
        self.scan_task = None
        
        # Состояние выбора изображений
        self.selected_images = set()
        
        # Текущий контекст галереи (для обновления счётчика и иконки)
        self.current_gallery_paths = []
        self.current_gallery_scope = "overview"
        
        # Кэш для path_to_cluster_map (оптимизация производительности)
        self._cached_path_to_cluster = None
        self._cached_path_to_cluster_scope = None
        
        # Создаём UI
        self.create_layout()
        
        # Загружаем статистику
        self.load_stats()
        
        # Показываем первый кластер по умолчанию
        asyncio.create_task(self.show_clusters_tab())
        
        # Фоновая загрузка модели
        asyncio.create_task(self.preload_model())
        
        # Обработчик изменения размера окна
        self.page.on_resize = self.on_window_resize

        # Обработчик закрытия окна — останавливаем фоновые вычисления
        self.page.window.on_event = self.on_window_event
    
    def create_layout(self):
        """Создание основного layout"""
        # Sidebar
        self.sidebar = ft.Container(
            content=ft.Column(
                [
                    self.create_stats_section(),
                    ft.Divider(),
                    self.create_clusters_section(),
                ],
                scroll=ft.ScrollMode.AUTO,
            ),
            width=280,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            padding=15,
            border_radius=10,
        )
        
        # Основная область
        self.main_content = ft.Column(
            [
                self.create_scan_section(),
                self.create_progress_section(),
                self.create_tabs(),
            ],
            expand=True,
            spacing=10,
        )
        
        # Добавляем на страницу
        self.page.add(
            ft.Row(
                [self.sidebar, self.main_content],
                expand=True,
                spacing=20,
            )
        )
    
    def create_stats_section(self):
        """Секция статистики"""
        self.stat_total = ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY)
        self.stat_total_size = ft.Text("0 Б", size=10, color=ft.Colors.ON_SURFACE_VARIANT)
        self.stat_duplicates = ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.RED)
        self.stat_duplicates_size = ft.Text("0 Б", size=10, color=ft.Colors.ON_SURFACE_VARIANT)
        self.stat_unique = ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN)
        self.stat_unique_size = ft.Text("0 Б", size=10, color=ft.Colors.ON_SURFACE_VARIANT)
        self.model_status_ring = ft.ProgressRing(width=16, height=16, stroke_width=2)
        self.model_status = ft.Text("Загрузка модели...", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.model_status_row = ft.Row(
            [self.model_status_ring, self.model_status],
            spacing=6,
        )
        
        return ft.Column(
            [
                ft.Text("Статистика", size=18, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [
                        ft.Column(
                            [self.stat_total, self.stat_total_size, ft.Text("Всего", size=12)],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Column(
                            [self.stat_duplicates, self.stat_duplicates_size, ft.Text("Дубли", size=12)],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Column(
                            [self.stat_unique, self.stat_unique_size, ft.Text("Уникальных", size=12)],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                ),
                self.model_status_row,
            ],
            spacing=10,
        )
    
    def create_clusters_section(self):
        """Секция категорий - кнопки кластеров в 3 столбца"""
        self.clusters_grid = ft.Column([])
        
        # Заголовок с количеством категорий
        self.categories_header = ft.Text("Категории (0)", size=18, weight=ft.FontWeight.BOLD)
        
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
            self.categories_header.value = "Категории (0)"
            return
        
        # Обновляем заголовок с количеством категорий
        self.categories_header.value = f"Категории ({len(self.clusters)})"
        
        # Создаём 3 столбца с кнопками
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
                    # Нумерация с 1, показываем количество изображений
                    button = ft.ElevatedButton(
                        f"{cluster_id + 2} ({len(members)})",
                        data=cluster_id,
                        width=70,
                        height=50,
                        bgcolor=ft.Colors.PRIMARY if is_active else ft.Colors.SURFACE_CONTAINER_HIGHEST,
                        color=ft.Colors.ON_PRIMARY if is_active else ft.Colors.ON_SURFACE,
                        on_click=self.on_cluster_button_click,
                        style=ft.ButtonStyle(
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
        asyncio.create_task(self.show_clusters_tab())
        # Обновляем подсветку кнопок
        self.update_clusters_list()
    
    def on_window_event(self, e):
        """Обработчик событий окна — при закрытии останавливаем фоновые вычисления."""
        if e.data == ft.WindowEventType.CLOSE:
            scanner.STOP_REQUESTED = True
            self.scanning = False
            # Даём фоновым потокам сигнал остановки и выходим.
            # os._exit может прервать запись в SQLite; sys.exit позволяет
            # Python-сборщику мусора корректно закрыть соединения.
            import sys as _sys
            _sys.exit(0)

    def on_window_resize(self, e):
        """Обработчик изменения размера окна - пересоздаём галерею"""
        # Сохраняем текущий offset
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
        
        # Пересоздаём текущую вкладку
        if current_tab == -1:
            asyncio.create_task(self.show_clusters_tab())
        elif current_tab == 0:
            self.show_search_tab()
        elif current_tab == 1:
            asyncio.create_task(self.show_export_tab())
    
    def create_scan_section(self):
        """Секция сканирования"""
        self.scan_path_input = ft.TextField(
            label="Путь для сканирования",
            value="Все диски",
            expand=True,
            border_radius=8,
            hint_text="Выберите папку для сканирования",
        )
        
        # Кнопка выбора пути через проводник
        browse_scan_path_button = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            tooltip="Выбрать папку",
            on_click=self.browse_scan_path,
        )
        
        self.scan_button = ft.ElevatedButton(
            "Сканировать",
            icon=ft.Icons.SEARCH,
            bgcolor=ft.Colors.PRIMARY,
            color=ft.Colors.ON_PRIMARY,
            on_click=self.toggle_scan,
        )
        
        return ft.Column(
            [
                ft.Row(
                    [self.scan_path_input, browse_scan_path_button, self.scan_button],
                    spacing=10,
                ),
            ],
            spacing=5,
        )
    
    def create_progress_section(self):
        """Секция прогресса"""
        self.progress_bar = ft.ProgressBar(
            visible=False,
            value=0,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
        )
        self.progress_text = ft.Text(
            "",
            size=14,
            color=ft.Colors.ON_SURFACE_VARIANT,
            visible=False,
        )
        
        self.progress_container = ft.Column(
            [self.progress_bar, self.progress_text],
            visible=False,
            spacing=5,
        )
        
        return self.progress_container
    
    def create_tabs(self):
        """Вкладки"""
        # Кнопки вкладок
        self.tab_buttons = ft.Row(
            [
                ft.ElevatedButton(
                    "Поиск",
                    icon=ft.Icons.SEARCH,
                    on_click=lambda e: self.switch_tab(0),
                ),
                ft.ElevatedButton(
                    "Экспорт",
                    icon=ft.Icons.FOLDER_OPEN,
                    on_click=lambda e: self.switch_tab(1),
                ),
            ],
            spacing=5,
        )
        
        # Контент вкладок
        self.tab_content = ft.Container(
            content=ft.Row(
                [ft.ProgressRing(), ft.Text("Загрузка...", size=14)],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=10,
            ),
            expand=True,
        )
        
        # Текущая вкладка (-1 = кластерный режим)
        self.current_tab = -1
        
        return ft.Column(
            [self.tab_buttons, self.tab_content],
            expand=True,
            spacing=10,
        )
    
    async def preload_model(self):
        """Фоновая загрузка модели"""
        self.model_loading = True
        try:
            from core.embedder import get_embedder
            get_embedder()
            self.model_loaded = True
            self.model_status_ring.visible = False
            self.model_status.value = "✅ Модель загружена"
            self.model_status.color = ft.Colors.GREEN
        except Exception as e:
            self.model_status_ring.visible = False
            self.model_status.value = f"⚠️ Ошибка: {e}"
            self.model_status.color = ft.Colors.RED
        finally:
            self.model_loading = False
            self.page.update()
    
    def load_stats(self):
        """Загрузка статистики из БД"""
        try:
            stats = database.load_db_stats()
            
            self.stat_total.value = str(stats["total"])
            self.stat_duplicates.value = str(stats["duplicates"])
            self.stat_unique.value = str(stats["unique"])
            
            # Обновляем размеры
            total_size = stats.get("total_size", 0)
            self.stat_total_size.value = _format_size(total_size)
            
            # Размер дубликатов и уникальных — вычисляем из БД
            try:
                dup_size = database.get_duplicates_size()
                self.stat_duplicates_size.value = _format_size(dup_size)
                self.stat_unique_size.value = _format_size(max(total_size - dup_size, 0))
            except Exception:
                self.stat_duplicates_size.value = "—"
                self.stat_unique_size.value = "—"
            
            # Загрузить кластеры и их имена
            clusters, cluster_names = database.load_clusters_with_names()
            old_clusters_id = id(self.clusters) if self.clusters else None
            self.clusters = clusters or {}
            self.cluster_names = cluster_names or {}
            
            # Инвалидировать кэш если кластеры изменились
            if old_clusters_id != id(self.clusters):
                self._invalidate_cluster_cache()
            
            # Обновить список категорий в боковой панели
            self.update_clusters_list()
            
            self.page.update()
        except Exception as e:
            print(f"Ошибка загрузки статистики: {e}")
    
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
        """Восстановить scroll offset галереи"""
        try:
            scroll_key = f"gallery_scroll_{scope}"
            offset = getattr(self.page.session, scroll_key, None)
            if offset is not None and isinstance(offset, (int, float)) and offset > 0:
                async def do_restore():
                    # Догружаем элементы до сохранённой lazy позиции,
                    # чтобы скролл мог физически добраться до сохранённого offset
                    if paths and page_size:
                        lazy_key = f"gallery_lazy_offset_{scope}"
                        saved_lazy = getattr(self.page.session, lazy_key, 0)
                        if saved_lazy and isinstance(saved_lazy, (int, float)):
                            saved_lazy = int(saved_lazy)
                            # Догружаем пока не достигнем сохранённого lazy offset
                            for _ in range(200):  # защита от бесконечного цикла
                                current_lazy = getattr(self.page.session, f"gallery_offset_{scope}", 0)
                                if current_lazy is None or not isinstance(current_lazy, (int, float)):
                                    current_lazy = 0
                                current_lazy = int(current_lazy)
                                if current_lazy >= saved_lazy or current_lazy >= len(paths):
                                    break
                                await self.load_more(paths, scope, gallery)
                                await asyncio.sleep(0.05)
                    
                    # Пробуем несколько раз с задержкой, чтобы галерея успела отрисоваться
                    for attempt in range(5):
                        await asyncio.sleep(0.3)
                        try:
                            gallery.scroll_to(offset=offset, duration=0)
                            self.page.update()
                            return
                        except Exception:
                            pass
                asyncio.create_task(do_restore())
        except Exception:
            pass

    def _progress_callback(self, stage: str, current: int, total: int, message: str):
        """Callback для обновления прогресс-бара из фоновых потоков.

        page.update() в Flet НЕ потокобезопасен — на Windows вызов из чужого потока
        даёт WinError 1. Используем page.run_task() для переноса обновления
        в главный (UI) поток.
        """
        now = time.monotonic()
        if now - getattr(self, '_last_progress_update', 0) < 0.05:
            return
        self._last_progress_update = now
        try:
            labels = {
                "scan": "Сканирование",
                "dedup": "Дедупликация",
                "embed": "Эмбеддинги",
                "cluster": "Кластеризация",
            }
            label = labels.get(stage, stage)

            # Безопасное обновление значений
            async def update_ui():
                if total and total > 0:
                    progress = min(current / total, 1.0)
                    self.progress_bar.value = progress
                    self.progress_text.value = f"{label}: {current}/{total} ({int(progress * 100)}%) · {message}"
                else:
                    # Индикатор неопределённого прогресса
                    self.progress_text.value = f"{label}: {message}"

                self.page.update()

            # Переносим обновление UI в главный поток
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
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                path = f"{letter}:\\"
                if os.path.exists(path):
                    disks.append(path)
            return disks
        else:
            # macOS/Linux: / и /Volumes/* (macOS дополнительно монтирует
            # внешние диски в /Volumes)
            disks = ["/"]
            volumes = "/Volumes"
            if os.path.isdir(volumes):
                for name in os.listdir(volumes):
                    full = os.path.join(volumes, name)
                    if os.path.isdir(full):
                        disks.append(full)
            return disks

    async def toggle_scan(self, e):
        """Запуск/остановка сканирования"""
        if self.scanning:
            # Останавливаем сканирование
            self.scanning = False
            scanner.STOP_REQUESTED = True
            self.scan_button.text = "Сканировать"
            self.scan_button.icon = ft.Icons.SEARCH
            self.scan_button.bgcolor = ft.Colors.PRIMARY
            self.progress_bar.visible = False
            self.progress_text.visible = False
            self.progress_container.visible = False
            self.page.update()
            self.show_snackbar("Сканирование остановлено", ft.Colors.ORANGE)
            return
        
        scan_path = self.scan_path_input.value
        
        # Проверка пути
        if scan_path.lower() in ["all_disks", "все диски"]:
            # Сканируем все доступные диски
            disks = self._get_available_disks()
            if not disks:
                self.show_snackbar("Не найдено доступных дисков", ft.Colors.RED)
                return
            scan_paths = disks
        elif not os.path.exists(scan_path):
            self.show_snackbar(f"Путь не существует: {scan_path}", ft.Colors.RED)
            return
        else:
            scan_paths = [scan_path]
        
        self.scanning = True
        scanner.STOP_REQUESTED = False
        self.scan_button.text = "Стоп"
        self.scan_button.icon = ft.Icons.STOP
        self.scan_button.bgcolor = ft.Colors.ERROR
        
        # Показываем прогресс
        self.progress_container.visible = True
        self.progress_bar.visible = True
        self.progress_text.visible = True
        self.progress_bar.value = 0
        self.progress_text.value = "Подготовка..."
        self.page.update()
        
        try:
            # 1. Сканирование (все пути последовательно)
            self.progress_text.value = "Сканирование..."
            self.progress_bar.value = 0
            self.page.update()
            
            files = []
            for i, path in enumerate(scan_paths):
                if scanner.STOP_REQUESTED:
                    break
                self.progress_text.value = f"Сканирование ({i + 1}/{len(scan_paths)}): {path}"
                self.page.update()
                
                result = await asyncio.to_thread(
                    scanner.run,
                    path,
                    None, None, None,
                    incremental=True,
                    progress_callback=self._progress_callback,
                )
                if result:
                    files.extend(result)
            
            if scanner.STOP_REQUESTED:
                return
            
            if not files:
                self.show_snackbar("Файлы не найдены", ft.Colors.ORANGE)
                return
            
            # 2. Дедупликация
            self.progress_text.value = "Дедупликация..."
            self.progress_bar.value = 0
            self.page.update()
            
            await asyncio.to_thread(dedup.run, incremental=True, progress_callback=self._progress_callback)

            if scanner.STOP_REQUESTED:
                return

            # Обновляем статистику после дедупликации (дубликаты уже сохранены в БД)
            self.load_stats()

            # 3. Эмбеддинги
            self.progress_text.value = "Эмбеддинги..."
            self.progress_bar.value = 0
            self.page.update()
            
            result = await asyncio.to_thread(embedder.run, incremental=True, progress_callback=self._progress_callback)

            # Эмбеддинги могли измениться — сбрасываем кэш семантического поиска,
            # чтобы последующие запросы использовали свежие данные.
            search.clear_cache()

            if scanner.STOP_REQUESTED:
                return
            
            # 4. Кластеризация (только если включено в конфиге)
            if config.AUTO_CLUSTER_AFTER_SCAN:
                self.progress_text.value = "Кластеризация..."
                self.progress_bar.value = 0
                self.page.update()
                
                await asyncio.to_thread(clustererhdb.run, progress_callback=self._progress_callback)
                
                if scanner.STOP_REQUESTED:
                    return
                
                # Обновить статистику после кластеризации
                self.load_stats()
            
            self.show_snackbar("Готово!", ft.Colors.GREEN)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if self.scanning:
                self.show_snackbar(f"Ошибка: {e}", ft.Colors.RED)
        finally:
            self.scanning = False
            scanner.STOP_REQUESTED = False
            self.scan_button.text = "Сканировать"
            self.scan_button.icon = ft.Icons.SEARCH
            self.scan_button.bgcolor = ft.Colors.PRIMARY
            self.progress_bar.visible = False
            self.progress_text.visible = False
            self.progress_container.visible = False
            self.page.update()
    
    async def browse_scan_path(self, e):
        """Открыть проводник для выбора пути сканирования"""
        def get_directory_path():
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            folder_path = filedialog.askdirectory(
                title="Выберите папку для сканирования",
                initialdir="/"
            )
            
            root.destroy()
            return folder_path
        
        # Запускаем в отдельном потоке, чтобы не блокировать UI
        path = await asyncio.to_thread(get_directory_path)
        if path:
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
    
    def switch_tab(self, index: int):
        """Переключение вкладки"""
        self._save_current_gallery_scroll()
        self.current_tab = index
        if index == 0:
            self.show_search_tab()
        elif index == 1:
            asyncio.create_task(self.show_export_tab())
    
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
        """Показать вкладку 'Экспорт' — выбранные файлы в галерее + статистика"""
        # Выбранные файлы (только существующие)
        selected_paths = [p for p in self.selected_images if os.path.exists(p)]
        selected_paths.sort()
        
        total_files = len(selected_paths)
        
        # Вычисляем общий размер выбранных файлов
        total_size_bytes = 0
        for path in selected_paths:
            try:
                total_size_bytes += os.path.getsize(path)
            except:
                pass
        
        # Конвертируем в читаемый формат
        if total_size_bytes < 1024 * 1024:
            size_str = f"{total_size_bytes / 1024:.1f} КБ"
        elif total_size_bytes < 1024 * 1024 * 1024:
            size_str = f"{total_size_bytes / (1024 * 1024):.1f} МБ"
        else:
            size_str = f"{total_size_bytes / (1024 * 1024 * 1024):.2f} ГБ"
        
        # Статистика выбранных файлов
        stats_row = ft.Row(
            [
                ft.Column(
                    [ft.Text(str(total_files), size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                     ft.Text("Выбрано файлов", size=12)],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Column(
                    [ft.Text(size_str, size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                     ft.Text("Общий размер", size=12)],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_AROUND,
        )
        
        # Поле папки назначения
        self.export_dest_folder = ft.TextField(
            label="Папка назначения",
            value="Фотоальбом",
            expand=True,
        )
        
        browse_button = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            tooltip="Выбрать папку",
            on_click=lambda e: self.browse_destination_folder(self.export_dest_folder),
        )
        
        # Кнопка экспорта
        export_button = ft.ElevatedButton(
            "Экспортировать выбранные",
            icon=ft.Icons.DOWNLOAD,
            bgcolor=ft.Colors.PRIMARY,
            color=ft.Colors.ON_PRIMARY,
            on_click=self.export_all_files,
        )
        
        # Кнопки управления
        controls_row = ft.Row(
            [
                self.export_dest_folder,
                browse_button,
                export_button,
            ],
            spacing=10,
        )
        
        # Сохраняем текущий контекст галереи
        self.current_gallery_paths = selected_paths
        self.current_gallery_scope = "export"
        
        # Галерея выбранных файлов
        if selected_paths:
            # Показываем индикатор загрузки пока генерируются миниатюры
            self.tab_content.content = ft.Column(
                [stats_row, ft.Divider(), controls_row, ft.Divider(), ft.ProgressRing()],
                expand=True,
            )
            self.page.update()
            
            gallery = await self.create_gallery(selected_paths, "export")
            content = ft.Column(
                [stats_row, ft.Divider(), controls_row, ft.Divider(), gallery],
                expand=True,
            )
        else:
            content = ft.Column(
                [stats_row, ft.Divider(), controls_row,
                 ft.Text("Ничего не выбрано. Выберите изображения в галерее.", size=14, color=ft.Colors.ON_SURFACE_VARIANT)],
                expand=True,
            )
        
        self.tab_content.content = content
        self.page.update()
        if selected_paths:
            self._restore_gallery_scroll(gallery, "export", selected_paths, 100)
    
    def export_all_files(self, e):
        """Экспорт выбранных файлов по категориям"""
        # Только выбранные файлы (существующие)
        selected_paths = [p for p in self.selected_images if os.path.exists(p)]
        if not selected_paths:
            self.show_snackbar("Нет выбранных файлов для экспорта", ft.Colors.ORANGE)
            return
        
        dest_folder = self.export_dest_folder.value if hasattr(self, 'export_dest_folder') else "Фотоальбом"
        dest_path = Path(dest_folder).expanduser()
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Загружаем кластеры
        clusters = self.clusters or database.load_clusters() or {}
        
        # Группируем выбранные файлы по категориям
        files_by_cluster = {}
        unclustered = []
        
        for path in selected_paths:
            found_cluster = False
            for cluster_id, members in clusters.items():
                if path in members:
                    if cluster_id not in files_by_cluster:
                        files_by_cluster[cluster_id] = []
                    files_by_cluster[cluster_id].append(path)
                    found_cluster = True
                    break
            if not found_cluster:
                unclustered.append(path)
        
        # Копируем файлы
        total_copied = 0
        
        # Копируем файлы по категориям
        for cluster_id, files in files_by_cluster.items():
            if not files:
                continue
            cluster_folder = dest_path / f"Категория_{cluster_id}"
            cluster_folder.mkdir(parents=True, exist_ok=True)
            used_names = set()
            for file_path in files:
                try:
                    dst = _unique_path(cluster_folder, Path(file_path).name, used_names)
                    shutil.copy2(file_path, dst)
                    used_names.add(dst.name)
                    total_copied += 1
                except Exception as ex:
                    print(f"Ошибка копирования {file_path}: {ex}")
        
        # Копируем некатегоризированные файлы
        if unclustered:
            unclustered_folder = dest_path / "Без_категории"
            unclustered_folder.mkdir(parents=True, exist_ok=True)
            used_names = set()
            for file_path in unclustered:
                try:
                    dst = _unique_path(unclustered_folder, Path(file_path).name, used_names)
                    shutil.copy2(file_path, dst)
                    used_names.add(dst.name)
                    total_copied += 1
                except Exception as ex:
                    print(f"Ошибка копирования {file_path}: {ex}")
        
        self.show_snackbar(f"Скопировано {total_copied} файлов в {dest_path}", ft.Colors.GREEN)

    async def show_clusters_tab(self):
        """Показать вкладку 'Категории'"""
        if not self.clusters:
            self.tab_content.content = ft.Text("Категории ещё не созданы. Запустите полный цикл.")
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
        self.cluster_select_all_icon_button = ft.IconButton(
            icon=self.get_checkbox_icon(selection_state),
            tooltip="Выбрать/Снять все",
            on_click=lambda e: self.toggle_select_all(members),
        )
        
        # Счётчик выбранных
        self.selected_count_text = ft.Text(f"Выбрано: {len([p for p in members if p in self.selected_images])}", size=14)
        
        # Кнопки управления (без копирования)
        controls_row = ft.Row(
            [
                self.cluster_select_all_icon_button,
                self.selected_count_text,
            ],
            spacing=10,
        )
        
        # Сохраняем текущий контекст галереи
        self.current_gallery_paths = members
        self.current_gallery_scope = f"cluster_{cluster_id}"
        
        # Показываем индикатор загрузки пока генерируются миниатюры
        self.tab_content.content = ft.Column(
            [controls_row, ft.Divider(), ft.ProgressRing()],
            expand=True,
        )
        self.page.update()
        
        # Галерея (генерация миниатюр в фоновом потоке)
        gallery = await self.create_gallery(members, f"cluster_{cluster_id}")
        
        self.tab_content.content = ft.Column(
            [controls_row, ft.Divider(), gallery],
            expand=True,
        )
        self.page.update()
        self._restore_gallery_scroll(gallery, f"cluster_{cluster_id}", members, 100)
    
    def show_search_tab(self):
        """Показать вкладку 'Поиск'"""
        # Создаём элементы только один раз, чтобы сохранять состояние
        # (введённый текст и результаты) при переключении вкладок
        if not hasattr(self, 'search_input'):
            self.search_input = ft.TextField(
                label="Поиск по описанию",
                hint_text="например: кот на окне",
                expand=True,
                on_submit=self.do_search,
            )
            
            self.search_button = ft.ElevatedButton(
                "Найти",
                icon=ft.Icons.SEARCH,
                on_click=self.do_search,
            )
            
            self.search_results_container = ft.Column(
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            )
        
        # Сохраняем текущий контекст галереи
        self.current_gallery_scope = "search_results"
        self.current_gallery_paths = getattr(self, 'search_result_paths', [])
        
        self.tab_content.content = ft.Column(
            [
                ft.Row([self.search_input, self.search_button], spacing=10),
                ft.Divider(),
                self.search_results_container,
            ],
            expand=True,
        )
        self.page.update()
    
    async def do_search(self, e):
        """Выполнить поиск"""
        query = self.search_input.value
        
        if not query:
            self.show_snackbar("Введите запрос", ft.Colors.ORANGE)
            return
        
        try:
            # Показываем индикатор загрузки
            self.search_results_container.controls = [
                ft.ProgressRing(),
                ft.Text("Поиск...", size=14),
            ]
            self.page.update()
            
            # Сбрасываем offset, scroll и lazy offset для результатов поиска
            session_key = "gallery_offset_search_results"
            setattr(self.page.session, session_key, 0)
            setattr(self.page.session, "gallery_scroll_search_results", 0)
            setattr(self.page.session, "gallery_lazy_offset_search_results", 0)
            
            # Поиск в фоне. top_k = 1000 — берём максимум, затем фильтруем
            # по порогу близости в core/search.py (динамическое количество).
            results = await asyncio.to_thread(
                search.run,
                query,
                top_k=1000
            )
            
            if results:
                paths = [p for p, _ in results]
                paths.sort()
                self.search_result_paths = paths
                self.current_gallery_paths = paths
                gallery = await self.create_gallery(paths, "search_results")
                
                self.search_results_container.controls = [
                    ft.Text(f"Найдено: {len(results)}", size=14),
                    ft.Divider(),
                    gallery,
                ]
            else:
                self.search_result_paths = []
                self.current_gallery_paths = []
                self.search_results_container.controls = [
                    ft.Text("Ничего не найдено", size=14),
                ]
            
            self.page.update()
            if results:
                self._restore_gallery_scroll(gallery, "search_results", paths, 100)
        except Exception as e:
            self.search_results_container.controls = [
                ft.Text(f"Ошибка поиска: {e}", size=14, color=ft.Colors.RED),
            ]
            self.page.update()
            self.show_snackbar(f"Ошибка поиска: {e}", ft.Colors.RED)
    
    def _get_path_to_cluster_map(self):
        """Возвращает dict {path: cluster_id} для всех изображений в кластерах.
        
        Оптимизировано с кэшированием - перестраивается только при изменении кластеров.
        """
        # Проверяем, нужно ли обновлять кэш
        if (self._cached_path_to_cluster is not None and 
            self._cached_path_to_cluster_scope == id(self.clusters)):
            return self._cached_path_to_cluster
        
        # Перестраиваем кэш
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

    def _make_gallery_item(self, path: str, scope: str, gallery: ft.GridView, path_to_cluster: dict = None, thumb_path: str = None):
        """Создаёт элемент галереи (изображение + бейдж категории для поиска)."""
        is_selected = path in self.selected_images
        
        # Используем предварительно сгенерированную миниатюру, либо генерируем на лету
        if thumb_path is None:
            thumb_path = thumbnail_cache.get_thumbnail(path, size=150)
        if thumb_path is not None:
            image = ft.Image(
                src=thumb_path,
                fit="contain",
                width=150,
                height=150,
            )
        else:
            # Fallback на оригинальный путь
            image = ft.Image(
                src=path,
                fit="contain",
                width=150,
                height=150,
            )
        
        # Для поиска показываем номер категории в углу
        if scope in ("search_results", "export") and path_to_cluster:
            cluster_id = path_to_cluster.get(path)
            if cluster_id is not None:
                badge = ft.Container(
                    content=ft.Text(
                        str(int(cluster_id) + 2),
                        size=10,
                        weight=ft.FontWeight.NORMAL,
                        color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE_VARIANT),
                    ),
                    bgcolor=ft.Colors.with_opacity(0.4, ft.Colors.SURFACE_CONTAINER_HIGHEST),
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
        
        return ft.GestureDetector(
            content=ft.Container(
                content=content,
                border_radius=8,
                border=ft.Border.all(3, ft.Colors.PRIMARY) if is_selected else None,
                data=path,  # <-- важно: храним оригинальный путь для поиска контейнера
            ),
            on_tap_down=on_tap_down,
            on_secondary_tap=on_secondary_tap,
        )

    async def _generate_thumbnails_parallel(self, paths: list, size: int = 150, max_workers: int = 4) -> dict:
        """Пакетная генерация миниатюр с параллельными потоками.

        Использует asyncio.Semaphore для ограничения числа одновременных
        потоков, чтобы не перегружать CPU при генерации.
        """
        semaphore = asyncio.Semaphore(max_workers)

        async def generate_one(path):
            async with semaphore:
                return path, await asyncio.to_thread(thumbnail_cache.get_thumbnail, path, size)

        results = await asyncio.gather(*[generate_one(p) for p in paths])
        return dict(results)

    def _generate_thumbnails_batch(self, paths: list, size: int = 150) -> dict:
        """Пакетная генерация миниатюр для списка путей.
        
        Предназначен для выполнения в фоновом потоке через asyncio.to_thread,
        чтобы не блокировать UI-поток при создании галереи.
        
        Returns:
            dict {path: thumb_path_or_None}
        """
        result = {}
        for path in paths:
            result[path] = thumbnail_cache.get_thumbnail(path, size=size)
        return result

    async def create_gallery(self, paths: list, scope: str):
        """Создание галереи с lazy loading.
        
        Генерация миниатюр выполняется в фоновом потоке через asyncio.to_thread,
        чтобы не блокировать UI при переключении категорий.
        """
        # Validate paths
        if not paths or not isinstance(paths, list):
            return ft.Column([ft.Text("Нет изображений")])
        
        # Загружаем выделение из БД
        saved_selection = database.load_selected_files(scope=scope)
        if saved_selection:
            self.selected_images.update(saved_selection)
        
        # Для поиска и экспорта строим map path -> cluster_id (один раз)
        path_to_cluster = self._get_path_to_cluster_map() if scope in ("search_results", "export") else None
        
        # Состояние для lazy loading
        session_key = f"gallery_offset_{scope}"
        if not hasattr(self.page.session, session_key):
            setattr(self.page.session, session_key, 0)
        
        # Всегда начинаем с 0. _restore_gallery_scroll догрузит элементы
        # до сохранённой lazy позиции, чтобы скролл мог восстановиться.
        offset = 0
        setattr(self.page.session, session_key, 0)
        
        # Для обзора используем меньший page_size для производительности
        if scope == "overview":
            page_size = 20
        elif scope.startswith("cluster_"):
            page_size = 40
        else:
            page_size = 60
        
        try:
            page_paths = paths[offset:offset + page_size]
        except (TypeError, ValueError):
            page_size = 20 if scope == "overview" else (40 if scope.startswith("cluster_") else 60)
            page_paths = paths[:page_size]
            offset = 0
        
        # Генерируем миниатюры параллельно в фоновом режиме, чтобы не блокировать UI
        thumbs = await self._generate_thumbnails_parallel(page_paths, 150, max_workers=4)
        
        # Создаём GridView
        gallery = ft.GridView(
            expand=True,
            runs_count=5,
            child_aspect_ratio=1.0,
            max_extent=150,
            spacing=5,
            run_spacing=5,
            on_scroll=self.create_scroll_handler(paths, scope, page_size),
        )
        
        # Добавляем изображения (минимальная работа в UI-потоке — только создание контролов)
        for path in page_paths:
            gallery.controls.append(
                self._make_gallery_item(path, scope, gallery, path_to_cluster, thumbs.get(path))
            )
        
        return gallery
    
    def create_scroll_handler(self, paths: list, scope: str, page_size: int):
        """Создать обработчик скролла для lazy loading (Infinite Scroll)"""
        # Отслеживаем последний загруженный offset для этого scope
        last_loaded_key = f"gallery_last_loaded_{scope}"
        
        def on_scroll(e: ft.ScrollEvent):
            try:
                # В Flet 0.86.5 ScrollEvent имеет local_position (Offset) и scroll_delta (Offset)
                # Получаем текущую позицию скролла из local_position.y
                local_pos = getattr(e, 'local_position', None)
                scroll_y = 0.0
                if local_pos is not None and hasattr(local_pos, 'y'):
                    scroll_y = float(local_pos.y)
                
                # Сохраняем текущий scroll offset для восстановления при переключении
                if scroll_y > 0:
                    scroll_key = f"gallery_scroll_{scope}"
                    setattr(self.page.session, scroll_key, scroll_y)
                    
                    # Сохраняем lazy offset (сколько элементов загружено) для восстановления
                    lazy_key = f"gallery_lazy_offset_{scope}"
                    lazy_offset = getattr(self.page.session, f"gallery_offset_{scope}", 0)
                    if lazy_offset is not None and isinstance(lazy_offset, (int, float)):
                        setattr(self.page.session, lazy_key, int(lazy_offset))
                
                # Получаем текущий offset
                session_key = f"gallery_offset_{scope}"
                offset = getattr(self.page.session, session_key, 0)
                if offset is None or not isinstance(offset, (int, float)):
                    offset = 0
                offset_int = int(offset)
                
                paths_len = int(len(paths)) if paths is not None else 0
                
                # Проверяем, дошли ли до конца
                if offset_int + page_size >= paths_len:
                    return
                
                # Проверяем, не загружаем ли мы уже (защита от повторных вызовов)
                last_loaded = getattr(self.page.session, last_loaded_key, 0)
                if last_loaded is not None and isinstance(last_loaded, (int, float)):
                    last_loaded = int(last_loaded)
                else:
                    last_loaded = 0
                
                if last_loaded >= offset_int + page_size:
                    return
                
                # Определяем, близко ли пользователь к концу загруженного контента
                should_load = False
                
                # Отслеживаем максимальную виденную позицию скролла
                max_seen_key = f"gallery_max_scroll_{scope}"
                max_seen = getattr(self.page.session, max_seen_key, 0)
                if max_seen is None or not isinstance(max_seen, (int, float)):
                    max_seen = 0
                
                if scroll_y > max_seen:
                    setattr(self.page.session, max_seen_key, scroll_y)
                
                # Если прокрутили больше 80% от максимальной виденной позиции — загружаем
                if max_seen > 0 and scroll_y >= max_seen * 0.8:
                    should_load = True
                # Или если scroll_y очень большой (более 3 экранов)
                elif scroll_y > 1500:
                    should_load = True
                
                if should_load:
                    setattr(self.page.session, last_loaded_key, offset_int + page_size)
                    asyncio.create_task(self.load_more(paths, scope, e.control))
                    
            except Exception as ex:
                # Log error for debugging
                print(f"Scroll error: {ex}")
                import traceback
                traceback.print_exc()
        
        return on_scroll
    
    async def load_more(self, paths: list, scope: str, gallery: ft.GridView):
        """Загрузка следующей порции (Infinite Scroll).
        
        Генерация миниатюр выполняется в фоновом потоке через asyncio.to_thread,
        чтобы не блокировать UI при прокрутке.
        """
        try:
            # Validate inputs
            if not paths or not isinstance(paths, list):
                return
            if gallery is None:
                return
            
            # Используем тот же page_size что и в create_gallery
            if scope == "overview":
                page_size = 20
            elif scope.startswith("cluster_"):
                page_size = 40
            else:
                page_size = 60
            
            session_key = f"gallery_offset_{scope}"
            offset = getattr(self.page.session, session_key, 0)
            
            # Ensure offset is a valid integer
            if offset is None or not isinstance(offset, (int, float)):
                offset = 0
            offset = int(offset)
            
            new_offset = offset + page_size
            
            # Validate new_offset
            if new_offset < 0 or new_offset > len(paths):
                return
            
            page_paths = paths[new_offset:new_offset + page_size]
            
            # Для поиска и экспорта строим map path -> cluster_id
            path_to_cluster = self._get_path_to_cluster_map() if scope in ("search_results", "export") else None
            
            # Генерируем миниатюры параллельно в фоновом режиме
            thumbs = await self._generate_thumbnails_parallel(page_paths, 150, max_workers=4)
            
            # Добавляем новые изображения (минимальная работа в UI-потоке)
            for path in page_paths:
                gallery.controls.append(
                    self._make_gallery_item(path, scope, gallery, path_to_cluster, thumbs.get(path))
                )
            
            setattr(self.page.session, f"gallery_offset_{scope}", new_offset)
            # Сохраняем lazy offset для восстановления скролла
            setattr(self.page.session, f"gallery_lazy_offset_{scope}", new_offset)
            
            self.page.update()
        except Exception as ex:
            # Silently handle errors and reset offset
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
                    container.border = ft.Border.all(3, ft.Colors.PRIMARY) if is_selected else None
                    updated = True
                    break
        
        # Обновляем счётчик выбранных
        if hasattr(self, 'selected_count_text'):
            selected_in_view = len([p for p in self.current_gallery_paths if p in self.selected_images])
            self.selected_count_text.value = f"Выбрано: {selected_in_view}"
        
        # Обновляем иконку "Выбрать/Снять все"
        if self.current_gallery_scope == "overview" and hasattr(self, 'select_all_icon_button'):
            state = self.get_selection_state(self.current_gallery_paths)
            self.select_all_icon_button.icon = self.get_checkbox_icon(state)
        elif self.current_gallery_scope.startswith("cluster_") and hasattr(self, 'cluster_select_all_icon_button'):
            state = self.get_selection_state(self.current_gallery_paths)
            self.cluster_select_all_icon_button.icon = self.get_checkbox_icon(state)
        
        self.page.update()
    
    def select_all(self, paths: list, value: bool):
        """Выбрать все / снять все"""
        if value:
            self.selected_images.update(paths)
        else:
            self.selected_images.difference_update(paths)
        
        # Сохраняем выделение в БД
        database.save_selected_files(list(self.selected_images), scope="global")
        
        self.show_snackbar(f"Выбрано: {len(self.selected_images)} файлов", ft.Colors.BLUE)
        
        # Обновляем текущую вкладку
        if self.current_tab == -1:
            asyncio.create_task(self.show_clusters_tab())
        elif self.current_tab == 0:
            self.show_search_tab()
        elif self.current_tab == 1:
            asyncio.create_task(self.show_export_tab())
    
    def show_preview(self, path: str):
        """Показать preview изображения с зумом, навигацией и полным путём."""
        # Список путей текущей галереи для навигации
        paths = self.current_gallery_paths or [path]
        if path not in paths:
            paths = [path] + paths
        current_idx = paths.index(path) if path in paths else 0
        
        preview_image = ft.Image(
            src=path,
            fit="contain",
            width=800,
            height=600,
        )
        
        path_text = ft.Text(
            path,
            size=12,
            color=ft.Colors.ON_SURFACE_VARIANT,
            selectable=True,
            text_align=ft.TextAlign.CENTER,
        )
        
        counter_text = ft.Text(
            f"{current_idx + 1} / {len(paths)}",
            size=12,
            color=ft.Colors.ON_SURFACE_VARIANT,
        )
        
        def close_dialog(e):
            dialog.open = False
            self.page.update()
        
        def zoom_in(e):
            preview_image.width = min(preview_image.width * 1.2, 1200)
            preview_image.height = min(preview_image.height * 1.2, 900)
            self.page.update()
        
        def zoom_out(e):
            preview_image.width = max(preview_image.width / 1.2, 200)
            preview_image.height = max(preview_image.height / 1.2, 150)
            self.page.update()
        
        def reset_zoom(e):
            preview_image.width = 800
            preview_image.height = 600
            self.page.update()
        
        def navigate(delta):
            nonlocal current_idx
            if not paths:
                return
            current_idx = (current_idx + delta) % len(paths)
            new_path = paths[current_idx]
            preview_image.src = new_path
            path_text.value = new_path
            counter_text.value = f"{current_idx + 1} / {len(paths)}"
            reset_zoom(None)
            self.page.update()
        
        def prev(e):
            navigate(-1)
        
        def next(e):
            navigate(1)
        
        # Оборачиваем изображение в GestureDetector для поддержки двойного клика в десктопе
        # on_double_tap - двойной клик для увеличения
        preview_with_gesture = ft.GestureDetector(
            content=preview_image,
            on_double_tap=zoom_in,
        )
        
        dialog = ft.AlertDialog(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.IconButton(
                                icon=ft.Icons.CLOSE,
                                tooltip="Закрыть",
                                on_click=close_dialog,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.ARROW_BACK,
                                tooltip="Назад",
                                on_click=prev,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.ARROW_FORWARD,
                                tooltip="Вперёд",
                                on_click=next,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.ZOOM_IN,
                                tooltip="Увеличить",
                                on_click=zoom_in,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.ZOOM_OUT,
                                tooltip="Уменьшить",
                                on_click=zoom_out,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.RESTORE,
                                tooltip="Сбросить",
                                on_click=reset_zoom,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    counter_text,
                    preview_with_gesture,
                    path_text,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
                scroll=ft.ScrollMode.AUTO,
            ),
            modal=True,
        )
        
        self.page.overlay.append(dialog)
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
                        content=ft.Text("Увеличить"),
                        icon=ft.Icons.ZOOM_IN,
                        on_click=open_preview,
                    ),
                    ft.PopupMenuItem(
                        content=ft.Text("Расположение"),
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
        """Открыть проводник для выбора папки назначения"""
        async def pick_folder():
            def get_directory_path():
                import tkinter as tk
                from tkinter import filedialog
                
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                folder_path = filedialog.askdirectory(
                    title="Выберите папку назначения",
                    initialdir=text_field.value or "/"
                )
                
                root.destroy()
                return folder_path
            
            # Запускаем в отдельном потоке
            path = await asyncio.to_thread(get_directory_path)
            if path:
                text_field.value = path
                self.page.update()
        
        asyncio.create_task(pick_folder())

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
    ft.app(
        target=ImageDedupApp,
    )


if __name__ == "__main__":
    main()