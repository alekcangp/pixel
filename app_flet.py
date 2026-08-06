import flet as ft
import asyncio
import os
import shutil
import time
from pathlib import Path
from core import database, scanner, dedup, embedder, clusterer, search


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
        
        # Создаём UI
        self.create_layout()
        
        # Загружаем статистику
        self.load_stats()
        
        # Показываем первую вкладку
        self.show_overview_tab()
        
        # Фоновая загрузка модели
        asyncio.create_task(self.preload_model())
        
        # Обработчик изменения размера окна
        self.page.on_resize = self.on_window_resize
    
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
        self.stat_duplicates = ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.RED)
        self.stat_unique = ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN)
        self.model_status = ft.Text("⏳ Загрузка модели...", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        
        return ft.Column(
            [
                ft.Text("Статистика", size=18, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [
                        ft.Column(
                            [self.stat_total, ft.Text("Всего", size=12)],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        ft.Column(
                            [self.stat_duplicates, ft.Text("Дубли", size=12)],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        ft.Column(
                            [self.stat_unique, ft.Text("Уникальных", size=12)],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                ),
                self.model_status,
            ],
            spacing=10,
        )
    
    def create_clusters_section(self):
        """Секция категорий - список категорий с возможностью переключения"""
        self.cluster_dropdown = ft.Dropdown(
            label="Выберите категорию",
            options=[],
            width=250,
            on_select=self.on_cluster_select,
            visible=False,  # Скрываем dropdown, используем список
        )
        
        self.clusters_list = ft.Column(
            [],
            scroll=ft.ScrollMode.AUTO,
            spacing=5,
        )
        
        self.categories_count_text = ft.Text("Категории: 0", size=14, color=ft.Colors.ON_SURFACE_VARIANT, visible=False)
        
        return ft.Column(
            [
                ft.Text("Категории", size=18, weight=ft.FontWeight.BOLD),
                self.categories_count_text,
                self.cluster_dropdown,
                self.clusters_list,
            ],
            spacing=10,
        )
    
    def update_clusters_list(self):
        """Обновление списка категорий в боковой панели"""
        self.clusters_list.controls.clear()
        
        if not self.clusters:
            self.clusters_list.controls.append(
                ft.Text("Нет категорий", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
            )
            self.categories_count_text.value = "Категории: 0"
            return
        
        # Обновляем количество категорий
        self.categories_count_text.value = f"Категории: {len(self.clusters)}"
        
        for cluster_id, members in sorted(self.clusters.items()):
            # Используем автоматически сгенерированное имя или стандартное
            cluster_name = self.cluster_names.get(cluster_id, f"Категория {cluster_id}")
            count = len(members)
            
            # Создаём контейнер для категории
            cluster_item = ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(
                            ft.Icons.FOLDER,
                            color=ft.Colors.PRIMARY,
                            size=20,
                        ),
                        ft.Column(
                            [
                                ft.Text(cluster_name, size=13, weight=ft.FontWeight.BOLD),
                                ft.Text(f"{count} фото", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                            ],
                            spacing=2,
                            expand=True,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                ),
                padding=10,
                border_radius=8,
                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                on_click=lambda e, cid=cluster_id: self.on_cluster_click(cid),
            )
            
            self.clusters_list.controls.append(cluster_item)
    
    def on_cluster_click(self, cluster_id):
        """Обработчик клика на кластер в боковой панели"""
        self.cluster_dropdown.value = str(cluster_id)
        self.current_tab = -1  # Кластерный режим (не активная вкладка)
        self.show_clusters_tab()
    
    def on_window_resize(self, e):
        """Обработчик изменения размера окна - пересоздаём галерею"""
        # Сохраняем текущий offset
        current_tab = self.current_tab
        if current_tab == 0:
            scope = "overview"
        elif current_tab == -1:
            scope = f"cluster_{self.cluster_dropdown.value}"
        elif current_tab == 1:
            scope = "search_results"
        elif current_tab == 2:
            scope = "export"
        else:
            return
        
        # Пересоздаём текущую вкладку
        if current_tab == 0:
            self.show_overview_tab()
        elif current_tab == -1:
            self.show_clusters_tab()
        elif current_tab == 1:
            self.show_search_tab()
        elif current_tab == 2:
            self.show_export_tab()
    
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
        
        return ft.Row(
            [self.scan_path_input, browse_scan_path_button, self.scan_button],
            spacing=10,
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
                    "Обзор",
                    icon=ft.Icons.PHOTO_LIBRARY,
                    on_click=lambda e: self.switch_tab(0),
                ),
                ft.ElevatedButton(
                    "Поиск",
                    icon=ft.Icons.SEARCH,
                    on_click=lambda e: self.switch_tab(1),
                ),
                ft.ElevatedButton(
                    "Экспорт",
                    icon=ft.Icons.FOLDER_OPEN,
                    on_click=lambda e: self.switch_tab(2),
                ),
            ],
            spacing=5,
        )
        
        # Контент вкладок
        self.tab_content = ft.Container(
            content=ft.Text("Загрузка..."),
            expand=True,
        )
        
        # Текущая вкладка
        self.current_tab = 0
        
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
            self.model_status.value = "✅ Модель загружена"
            self.model_status.color = ft.Colors.GREEN
        except Exception as e:
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
            
            # Загрузить кластеры и их имена
            clusters, cluster_names = database.load_clusters_with_names()
            self.clusters = clusters or {}
            self.cluster_names = cluster_names or {}
            
            # Обновить список категорий в боковой панели
            self.update_clusters_list()
            
            self.page.update()
        except Exception as e:
            print(f"Ошибка загрузки статистики: {e}")

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
    
    async def toggle_scan(self, e):
        """Запуск/остановка сканирования"""
        if self.scanning:
            # Останавливаем сканирование
            self.scanning = False
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
            # Не запускаем автоматически, показываем сообщение
            self.show_snackbar("Выберите конкретный путь для сканирования", ft.Colors.ORANGE)
            return
        elif not os.path.exists(scan_path):
            self.show_snackbar(f"Путь не существует: {scan_path}", ft.Colors.RED)
            return
        
        self.scanning = True
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
            # 1. Сканирование
            self.progress_text.value = "Сканирование..."
            self.progress_bar.value = 0
            self.page.update()
            
            files = await asyncio.to_thread(
                scanner.run,
                scan_path,
                None, None, None,
                incremental=True,
                progress_callback=self._progress_callback,
            )
            
            if not self.scanning:
                return
            
            if not files:
                self.show_snackbar("Файлы не найдены", ft.Colors.ORANGE)
                return
            
            # 2. Дедупликация
            self.progress_text.value = "Дедупликация..."
            self.progress_bar.value = 0
            self.page.update()
            
            await asyncio.to_thread(dedup.run, incremental=True, progress_callback=self._progress_callback)

            if not self.scanning:
                return

            # Обновляем статистику после дедупликации (дубликаты уже сохранены в БД)
            self.load_stats()
            print(f"[DEBUG] После дедупликации: всего={self.stat_total.value}, дублей={self.stat_duplicates.value}, уник={self.stat_unique.value}")

            # 3. Эмбеддинги
            self.progress_text.value = "Эмбеддинги..."
            self.progress_bar.value = 0
            self.page.update()
            
            print("[DEBUG] Запуск embedder.run...")
            result = await asyncio.to_thread(embedder.run, incremental=True, progress_callback=self._progress_callback)
            print(f"[DEBUG] embedder.run вернул: {result}")
            
            if not self.scanning:
                return
            
            # 4. Кластеризация
            self.progress_text.value = "Кластеризация..."
            self.progress_bar.value = 0
            self.page.update()
            
            print("[DEBUG] Запуск clusterer.run...")
            await asyncio.to_thread(clusterer.run, progress_callback=self._progress_callback)
            print("[DEBUG] clusterer.run завершён")
            
            if not self.scanning:
                return
            
            # Обновить статистику
            self.load_stats()
            
            self.show_snackbar("Готово!", ft.Colors.GREEN)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if self.scanning:
                self.show_snackbar(f"Ошибка: {e}", ft.Colors.RED)
        finally:
            self.scanning = False
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
    
    def on_cluster_select(self, e):
        """Обработчик выбора категории"""
        self.current_tab = -1  # Кластерный режим (не активная вкладка)
        self.show_clusters_tab()
    
    def switch_tab(self, index: int):
        """Переключение вкладки"""
        self.current_tab = index
        if index == 0:
            self.show_overview_tab()
        elif index == 1:
            self.show_search_tab()
        elif index == 2:
            self.show_export_tab()
        # NOTE: offset'ы галерей НЕ сбрасываются — они хранятся отдельно
        # для каждого scope (gallery_offset_{scope}) и сохраняются между
        # переключениями вкладок.
    
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
    
    def show_overview_tab(self):
        """Показать вкладку 'Обзор' (без дубликатов)"""
        images = database.load_images() or []
        
        if not images:
            self.tab_content.content = ft.Text("База пуста. Нажмите «Сканировать»")
            self.page.update()
            return
        
        # Загружаем пути дубликатов, чтобы исключить их из обзора
        duplicate_paths = database.load_duplicate_paths() or set()
        
        # Показываем только уникальные изображения (не дубликаты)
        paths = [img["path"] for img in images if img["path"] not in duplicate_paths]
        
        # Кнопка "Выбрать/Снять все" с интерактивной иконкой
        selection_state = self.get_selection_state(paths)
        self.select_all_icon_button = ft.IconButton(
            icon=self.get_checkbox_icon(selection_state),
            tooltip="Выбрать/Снять все",
            on_click=lambda: self.toggle_select_all(paths),
        )
        
        # Счётчик выбранных
        self.selected_count_text = ft.Text(f"Выбрано: {len([p for p in paths if p in self.selected_images])}", size=14)
        
        # Кнопки управления (без копирования)
        controls_row = ft.Row(
            [
                self.select_all_icon_button,
                self.selected_count_text,
            ],
            spacing=10,
        )
        
        # Сохраняем текущий контекст галереи
        self.current_gallery_paths = paths
        self.current_gallery_scope = "overview"
        
        # Галерея
        gallery = self.create_gallery(paths, "overview")
        
        self.tab_content.content = ft.Column(
            [controls_row, ft.Divider(), gallery],
            expand=True,
        )
        self.page.update()
    
    def show_export_tab(self):
        """Показать вкладку 'Экспорт'"""
        images = database.load_images() or []
        
        if not images:
            self.tab_content.content = ft.Text("База пуста. Нажмите «Сканировать»")
            self.page.update()
            return
        
        # Загружаем кластеры
        clusters = self.clusters or database.load_clusters() or {}
        
        # Вычисляем статистику экспорта:
        # 1. Реальные (непустые) кластеры, которые будут экспортированы
        # 2. Файлы, которые реально попадут в экспорт (в кластерах + без категории)
        # 3. Их суммарный размер
        
        # Собираем пути, которые реально будут экспортированы (только существующие файлы),
        # повторяя логику export_all_files
        export_paths = set()
        non_empty_clusters = 0
        files_by_cluster = {}
        unclustered = []
        
        for img in images:
            path = img["path"]
            if not os.path.exists(path):
                continue  # Файл не существует - не будет экспортирован
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
        
        # Считаем только непустые кластеры с существующими файлами
        for cluster_id, files in files_by_cluster.items():
            if files:
                non_empty_clusters += 1
                export_paths.update(files)
        
        export_paths.update(unclustered)
        
        total_files = len(export_paths)
        total_categories = non_empty_clusters
        
        # Вычисляем общий размер файлов, подлежащих экспорту
        total_size_bytes = 0
        for path in export_paths:
            try:
                if os.path.exists(path):
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
        
        # Заголовок
        header = ft.Text("Экспорт", size=18, weight=ft.FontWeight.BOLD)
        
        # Статистика
        stats_row = ft.Row(
            [
                ft.Column(
                    [ft.Text(str(total_categories), size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                     ft.Text("Категорий", size=12)],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Column(
                    [ft.Text(str(total_files), size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                     ft.Text("Файлов", size=12)],
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
            "Экспортировать все",
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
        
        self.tab_content.content = ft.Column(
            [header, stats_row, ft.Divider(), controls_row],
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )
        self.page.update()
    
    def export_all_files(self, e):
        """Экспорт всех файлов по категориям"""
        images = database.load_images() or []
        if not images:
            self.show_snackbar("Нет файлов для экспорта", ft.Colors.ORANGE)
            return
        
        dest_folder = self.export_dest_folder.value if hasattr(self, 'export_dest_folder') else "Фотоальбом"
        dest_path = Path(dest_folder).expanduser()
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Загружаем кластеры
        clusters = self.clusters or database.load_clusters() or {}
        
        # Группируем файлы по категориям
        files_by_cluster = {}
        unclustered = []
        
        for img in images:
            path = img["path"]
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
            for file_path in files:
                if os.path.exists(file_path):
                    try:
                        shutil.copy2(file_path, cluster_folder / Path(file_path).name)
                        total_copied += 1
                    except Exception as ex:
                        print(f"Ошибка копирования {file_path}: {ex}")
        
        # Копируем некатегоризированные файлы
        if unclustered:
            unclustered_folder = dest_path / "Без_категории"
            unclustered_folder.mkdir(parents=True, exist_ok=True)
            for file_path in unclustered:
                if os.path.exists(file_path):
                    try:
                        shutil.copy2(file_path, unclustered_folder / Path(file_path).name)
                        total_copied += 1
                    except Exception as ex:
                        print(f"Ошибка копирования {file_path}: {ex}")
        
        self.show_snackbar(f"Скопировано {total_copied} файлов в {dest_path}", ft.Colors.GREEN)

    def show_clusters_tab(self):
        """Показать вкладку 'Категории'"""
        if not self.clusters:
            self.tab_content.content = ft.Text("Категории ещё не созданы. Запустите полный цикл.")
            self.page.update()
            return
        
        # Получить выбранную категорию
        cluster_id = self.cluster_dropdown.value
        if not cluster_id:
            cluster_id = str(sorted(self.clusters.keys())[0])
        
        members = self.clusters.get(int(cluster_id), [])
        
        # Получаем имя категории
        cluster_name = self.cluster_names.get(int(cluster_id), f"Категория {cluster_id}")
        
        # Заголовок с именем категории
        header = ft.Text(f"{cluster_name} ({len(members)} файлов)", size=18, weight=ft.FontWeight.BOLD)
        
        # Кнопка "Выбрать/Снять все" с интерактивной иконкой
        selection_state = self.get_selection_state(members)
        self.cluster_select_all_icon_button = ft.IconButton(
            icon=self.get_checkbox_icon(selection_state),
            tooltip="Выбрать/Снять все",
            on_click=lambda: self.toggle_select_all(members),
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
        
        # Галерея
        gallery = self.create_gallery(members, f"cluster_{cluster_id}")
        
        self.tab_content.content = ft.Column(
            [header, controls_row, ft.Divider(), gallery],
            expand=True,
        )
        self.page.update()
    
    def show_search_tab(self):
        """Показать вкладку 'Поиск'"""
        # Создаём элементы только один раз, чтобы сохранять состояние
        # (введённый текст и результаты) при переключении вкладок
        if not hasattr(self, 'search_input'):
            self.search_input = ft.TextField(
                label="Поиск по описанию",
                hint_text="например: кот на окне",
                expand=True,
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
            
            # Поиск в фоне. top_k = 1000 — берём максимум, затем фильтруем
            # по порогу близости в core/search.py (динамическое количество).
            results = await asyncio.to_thread(
                search.run,
                query,
                top_k=1000
            )
            
            if results:
                paths = [p for p, _ in results]
                gallery = self.create_gallery(paths, "search_results")
                
                self.search_results_container.controls = [
                    ft.Text(f"Найдено: {len(results)}", size=14),
                    ft.Divider(),
                    gallery,
                ]
            else:
                self.search_results_container.controls = [
                    ft.Text("Ничего не найдено", size=14),
                ]
            
            self.page.update()
        except Exception as e:
            self.show_snackbar(f"Ошибка поиска: {e}", ft.Colors.RED)
    
    def create_gallery(self, paths: list, scope: str):
        """Создание галереи с lazy loading"""
        # Validate paths
        if not paths or not isinstance(paths, list):
            return ft.Column([ft.Text("Нет изображений")])
        
        # Загружаем выделение из БД
        saved_selection = database.load_selected_files(scope=scope)
        if saved_selection:
            self.selected_images.update(saved_selection)
        
        # Состояние для lazy loading
        session_key = f"gallery_offset_{scope}"
        if not hasattr(self.page.session, session_key):
            setattr(self.page.session, session_key, 0)
        
        offset = getattr(self.page.session, session_key, 0)
        if offset is None:
            offset = 0
            setattr(self.page.session, session_key, 0)
        
        # Ensure offset is an integer
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            offset = 0
            setattr(self.page.session, session_key, 0)
        
        page_size = 50
        try:
            page_paths = paths[offset:offset + page_size]
        except (TypeError, ValueError):
            page_paths = paths[:50]
            offset = 0
        
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
        
        # Добавляем изображения
        for path in page_paths:
            is_selected = path in self.selected_images
            gallery.controls.append(
                ft.Container(
                    content=ft.Image(
                        src=path,
                        fit="contain",
                        width=150,
                        height=150,
                    ),
                    border_radius=8,
                    border=ft.Border.all(3, ft.Colors.PRIMARY) if is_selected else None,
                    on_click=lambda e, p=path: self.toggle_image_selection(p, gallery),
                    on_long_press=lambda e, p=path: self.show_preview(p),
                )
            )
        
        # Кнопка "Загрузить ещё"
        try:
            if offset + page_size < len(paths):
                gallery.controls.append(
                    ft.ElevatedButton(
                        "Загрузить ещё...",
                        on_click=lambda: self.load_more(paths, scope, gallery),
                        icon=ft.Icons.ADD,
                    )
                )
        except (TypeError, ValueError):
            pass
        
        return gallery
    
    def create_scroll_handler(self, paths: list, scope: str, page_size: int):
        """Создать обработчик скролла для lazy loading"""
        def on_scroll(e: ft.ScrollEvent):
            try:
                # Если прокрутились до конца, загружаем ещё
                scroll_delta = getattr(e, 'scroll_delta', None)
                if scroll_delta is None or not isinstance(scroll_delta, (int, float)):
                    return
                
                if scroll_delta > 0:  # Скролл вниз
                    session_key = f"gallery_offset_{scope}"
                    offset = getattr(self.page.session, session_key, 0)
                    
                    # Ensure offset is a valid number
                    if offset is None or not isinstance(offset, (int, float)):
                        offset = 0
                    
                    # Convert to int for comparison
                    offset_int = int(offset)
                    paths_len = int(len(paths)) if paths is not None else 0
                    
                    # Проверяем, дошли ли до конца
                    if offset_int + page_size >= paths_len:
                        return
                    
                    # Загружаем следующую порцию
                    self.load_more(paths, scope, e.control)
            except Exception as ex:
                # Log error for debugging
                print(f"Scroll error: {ex}")
                import traceback
                traceback.print_exc()
        
        return on_scroll
    
    def load_more(self, paths: list, scope: str, gallery: ft.GridView):
        """Загрузка следующей порции"""
        try:
            # Validate inputs
            if not paths or not isinstance(paths, list):
                return
            if gallery is None:
                return
            
            page_size = 50
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
            
            # Удаляем кнопку "Загрузить ещё" если есть
            if gallery.controls and len(gallery.controls) > 0:
                last_control = gallery.controls[-1]
                if isinstance(last_control, ft.ElevatedButton):
                    gallery.controls.pop()
            
            # Добавляем новые изображения
            for path in page_paths:
                is_selected = path in self.selected_images
                gallery.controls.append(
                    ft.Container(
                        content=ft.Image(
                            src=path,
                            fit="contain",
                            width=150,
                            height=150,
                        ),
                        border_radius=8,
                        border=ft.Border.all(3, ft.Colors.PRIMARY) if is_selected else None,
                        on_click=lambda e, p=path: self.toggle_image_selection(p, gallery),
                        on_long_press=lambda e, p=path: self.show_preview(p),
                    )
                )
            
            setattr(self.page.session, f"gallery_offset_{scope}", new_offset)
            
            # Добавляем кнопку "Загрузить ещё" если есть ещё
            paths_len = int(len(paths)) if paths is not None else 0
            if new_offset + page_size < paths_len:
                gallery.controls.append(
                    ft.ElevatedButton(
                        "Загрузить ещё...",
                        on_click=lambda: self.load_more(paths, scope, gallery),
                        icon=ft.Icons.ADD,
                    )
                )
            
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
        if gallery:
            for control in gallery.controls:
                if isinstance(control, ft.Container):
                    # Проверяем, является ли этот контейнер искомым изображением
                    img_control = control.content
                    if isinstance(img_control, ft.Image) and img_control.src == path:
                        is_selected = path in self.selected_images
                        control.border = ft.Border.all(3, ft.Colors.PRIMARY) if is_selected else None
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
        if self.current_tab == 0:
            self.show_overview_tab()
        elif self.current_tab == -1:
            self.show_clusters_tab()
        elif self.current_tab == 2:
            self.show_export_tab()
    
    def show_preview(self, path: str):
        """Показать preview изображения с зумом"""
        preview_image = ft.Image(
            src=path,
            fit="contain",
            width=800,
            height=600,
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
        
        dialog = ft.AlertDialog(
            content=ft.Column(
                [
                    ft.Row(
                        [
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
                    preview_image,
                    ft.ElevatedButton(
                        "Закрыть",
                        on_click=close_dialog,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
                scroll=ft.ScrollMode.AUTO,
            ),
            modal=True,
        )
        
        self.page.dialog = dialog
        dialog.open = True
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
    
def main():
    ft.app(
        target=ImageDedupApp,
    )


if __name__ == "__main__":
    main()