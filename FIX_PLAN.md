# Детальный план исправлений

## Проблема 1: Зависание на эмбеддингах
**Причина:** Функция `core/embedder.py:run()` каждый раз создаёт новый экземпляр `SiglipEmbedder()` через `get_embedder()`, что загружает модель с диска. При вызове через `asyncio.to_thread()` это вызывает зависание из-за thread-safety issues в transformers.

## Решение 1: Сделать эмбеддер синглтоном

### Изменения в `core/embedder.py`:

**Шаг 1.1:** Добавить глобальную переменную для синглтона (после импортов, перед классом):
```python
_embedder_instance = None
```

**Шаг 1.2:** Заменить функцию `get_embedder()`:
```python
def get_embedder() -> SiglipEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = SiglipEmbedder()
    return _embedder_instance
```

**Шаг 1.3:** В функции `run()` удалить строку 193:
```python
# УДАЛИТЬ: embedder = get_embedder()
```
Модель уже загружена в `preload_model()`, используем预загруженный экземпляр.

## Решение 2: Прогресс автоименования кластеров

### Изменения в `core/clustererhdb.py`:

**Шаг 2.1:** Добавить параметр `progress_callback` в функцию `_auto_name_clusters_via_logits()`:
```python
def _auto_name_clusters_via_logits(clusters, paths, max_samples_per_cluster=5, progress_callback=None):
```

**Шаг 2.2:** Добавить счётчик и вызов callback после каждого кластера:
```python
total_clusters = len([c for c in clusters if c != -1])
processed = 0

for cluster_id, members in sorted(clusters.items()):
    if cluster_id == -1:
        cluster_names[cluster_id] = "Разное"
        if progress_callback:
            progress_callback("cluster", processed, total_clusters, "Автоименование...")
        continue
    
    # ... существующий код обработки кластера ...
    
    processed += 1
    if progress_callback:
        progress_callback("cluster", processed, total_clusters, f"Автоименование: {cluster_names.get(cluster_id, '...')}")
```

**Шаг 2.3:** Передать callback из `run()`:
```python
# В функции run(), строка 331:
cluster_names = _auto_name_clusters_via_logits(clusters, paths, progress_callback=progress_callback)
```

## Решение 3: Проверить импорты в `app_flet.py`

**Шаг 3.1:** Убедиться, что импортируется `clustererhdb`:
```python
from core import clustererhdb
```

**Шаг 3.2:** Проверить вызов в `toggle_scan()`:
```python
await asyncio.to_thread(clustererhdb.run, progress_callback=self._progress_callback)
```

## Проверка целостности цикла

### Последовательность в `app_flet.py:398-514`:
1. `scanner.run()` — сканирование файлов ✓
2. `dedup.run()` — дедупликация ✓
3. `embedder.run()` — эмбеддинги (использует预загруженную модель) ✓
4. `clustererhdb.run()` — кластеризация + автоименование ✓

### Загрузка модели:
- `app_flet.py:50` — `preload_model()` создаёт синглтон один раз
- Все последующие вызовы `get_embedder()` возвращают существующий экземпляр
- Нет повторной загрузки модели

## Ожидаемый результат:
1. Модель загружается один раз при старте приложения
2. Эмбеддинги не зависают (используется预загруженная модель)
3. Полный цикл scan→dedup→embed→cluster работает без зависаний
4. При автоименовании показывается прогресс: "Автоименование: 5/20 (25%) — Природа"

## Критические точки для тестирования:
- При первом запуске: модель загружается ~5-10 секунд
- При сканировании: прогресс-бар показывает этапы
- При эмбеддингах: нет зависания, используются预загруженные веса
- При кластеризации: виден прогресс автоименования