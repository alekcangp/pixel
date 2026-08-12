import argparse
import sys

import config

# Гарантируем построчную отдачу логов даже при перенаправлении в pipe/файл
# (иначе stdout блочно-буферизуется и логи появляются большими задержками).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Тяжёлые зависимости (torch/transformers/faiss) импортируются лениво
# внутри соответствующих команд, чтобы scan/dedup/clear стартовали быстро
# и сразу выводили лог, не дожидаясь загрузки моделей.


def cmd_clear(args):
    from core import database
    database.clear_all()
    print("База данных очищена: %s" % config.DB_FILE)


def cmd_scan(args):
    from core import scanner
    ext_list = [e.strip() for e in args.ext.split(",") if e.strip()]
    exclude_list = None
    if args.exclude:
        exclude_list = [e.strip() for e in args.exclude.split(",") if e.strip()]
    min_size = args.min_size * 1024 if args.min_size is not None else None
    scanner.run(args.path, ext_list, exclude_list, min_size=min_size, incremental=not args.full)


def cmd_run(args):
    from core import scanner, dedup
    print("Запуск: свежие scan + dedup для %s" % args.path)
    ext_list = [e.strip() for e in args.ext.split(",") if e.strip()]
    exclude_list = None
    if args.exclude:
        exclude_list = [e.strip() for e in args.exclude.split(",") if e.strip()]
    min_size = args.min_size * 1024 if args.min_size is not None else None
    # По умолчанию инкрементальный режим; --full для полного пересчёта
    scanner.run(args.path, ext_list, exclude_list, min_size=min_size, incremental=not args.full)
    dedup.run(move_to=args.move, incremental=not args.full)


def cmd_dedup(args):
    from core import scanner, dedup
    if args.path:
        ext_list = None
        if args.ext:
            ext_list = [e.strip() for e in args.ext.split(",") if e.strip()]
        exclude_list = None
        if args.exclude:
            exclude_list = [e.strip() for e in args.exclude.split(",") if e.strip()]
        min_size = args.min_size * 1024 if args.min_size is not None else None
        scanner.run(args.path, ext_list, exclude_list, min_size=min_size, incremental=not args.full)
    dedup.run(move_to=args.move, incremental=not args.full)


def cmd_embed(args):
    from core import embedder
    embedder.run(incremental=not args.full)


def cmd_cluster_hdb(args):
    from core import clustererhdb
    clustererhdb.run()


def cmd_search(args):
    from core import search
    search.run(args.query, args.top_k)


def cmd_phash_search(args):
    from core import search
    image_path = args.query
    top_k = args.top_k
    search.run_hash_search(image_path, top_k=top_k)


def cmd_ui_flet(args):
    from app_flet import main
    main()


def main():
    parser = argparse.ArgumentParser(description="Дедупликация, кластеризация и семантический поиск изображений")
    sub = parser.add_subparsers(dest="command", required=True)

    p_clear = sub.add_parser("clear", help="Полная очистка кэша (storage/)")
    p_clear.set_defaults(func=cmd_clear)

    p_run = sub.add_parser("run", help="Сканирование + дедупликация за один проход (по умолчанию инкрементально)")
    p_run.add_argument("--path", required=True, help="Путь к директории")
    p_run.add_argument("--ext", default=",".join(config.DEFAULT_EXTENSIONS),
                       help="Расширения через запятую (по умолчанию все поддерживаемые)")
    p_run.add_argument("--exclude", default=None,
                       help="Дополнительные директории для исключения через запятую (добавляются к стандартным)")
    p_run.add_argument("--min-size", type=int, default=None,
                       help="Минимальный размер файла в КБ (по умолчанию %d КБ)" % (config.MIN_FILE_SIZE // 1024))
    p_run.add_argument("--move", default=None,
                       help="Переместить дубликаты (всё, кроме одного из группы) в указанную папку")
    p_run.add_argument("--full", action="store_true",
                       help="Полный пересчёт (вместо инкрементального)")
    p_run.set_defaults(func=cmd_run)

    p_scan = sub.add_parser("scan", help="Сканирование файлов (по умолчанию инкрементально)")
    p_scan.add_argument("--path", required=True, help="Путь к директории")
    p_scan.add_argument("--ext", default=",".join(config.DEFAULT_EXTENSIONS),
                        help="Расширения через запятую (по умолчанию все поддерживаемые)")
    p_scan.add_argument("--exclude", default=None,
                        help="Дополнительные директории для исключения через запятую (добавляются к стандартным)")
    p_scan.add_argument("--min-size", type=int, default=None,
                        help="Минимальный размер файла в КБ (по умолчанию %d КБ)" % (config.MIN_FILE_SIZE // 1024))
    p_scan.add_argument("--full", action="store_true",
                        help="Полный пересчёт (вместо инкрементального)")
    p_scan.set_defaults(func=cmd_scan)

    p_dedup = sub.add_parser("dedup", help="Дедупликация (pHash + LSH-кластеризация, по умолчанию инкрементально)")
    p_dedup.add_argument("--path", default=None,
                         help="Если задан — сначала пересканировать эту директорию (свежий rescan)")
    p_dedup.add_argument("--ext", default=None,
                         help="Расширения для rescan (только с --path)")
    p_dedup.add_argument("--exclude", default=None,
                         help="Директории для исключения rescan (только с --path)")
    p_dedup.add_argument("--min-size", type=int, default=None,
                         help="Минимальный размер файла в КБ для rescan (по умолчанию %d КБ)" % (config.MIN_FILE_SIZE // 1024))
    p_dedup.add_argument("--move", default=None,
                         help="Переместить дубликаты (всё, кроме одного из группы) в указанную папку")
    p_dedup.add_argument("--full", action="store_true",
                         help="Полный пересчёт (вместо инкрементального)")
    p_dedup.set_defaults(func=cmd_dedup)

    p_embed = sub.add_parser("embed", help="Эмбеддинги SigLIP2 (по умолчанию инкрементально)")
    p_embed.add_argument("--full", action="store_true",
                         help="Полный пересчёт (вместо инкрементального)")
    p_embed.set_defaults(func=cmd_embed)

    p_cluster_hdb = sub.add_parser("cluster-hdb", help="Кластеризация (HDBSCAN + UMAP)")
    p_cluster_hdb.set_defaults(func=cmd_cluster_hdb)

    p_search = sub.add_parser("search", help="Семантический поиск по тексту")
    p_search.add_argument("query", help="Текстовый запрос")
    p_search.add_argument("--top-k", type=int, default=None, help="Кол-во результатов")
    p_search.set_defaults(func=cmd_search)

    p_phash_search = sub.add_parser("phash-search", help="Поиск похожих изображений по pHash через LSH")
    p_phash_search.add_argument("query", help="Путь к изображению")
    p_phash_search.add_argument("--top-k", type=int, default=None, help="Кол-во результатов")
    p_phash_search.set_defaults(func=cmd_phash_search)

    p_ui_flet = sub.add_parser("ui-flet", help="Запустить Flet desktop UI")
    p_ui_flet.set_defaults(func=cmd_ui_flet)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()