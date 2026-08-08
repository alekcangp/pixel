import sqlite3
import config

conn = sqlite3.connect(config.DB_FILE)
conn.row_factory = sqlite3.Row

# Count duplicates in dedup_groups table
dup_count = conn.execute('SELECT COUNT(*) FROM dedup_groups WHERE is_canonical = 0').fetchone()[0]
print(f'Дубликатов в БД (is_canonical=0): {dup_count}')

# Count total in dedup_groups
total_dedup = conn.execute('SELECT COUNT(*) FROM dedup_groups').fetchone()[0]
print(f'Всего записей в dedup_groups: {total_dedup}')

# Find orphaned dedup entries (paths not in images table)
orphaned = conn.execute('''
    SELECT COUNT(*) 
    FROM dedup_groups dg
    LEFT JOIN images i ON dg.image_id = i.id
    WHERE i.id IS NULL AND dg.is_canonical = 0
''').fetchone()[0]
print(f'Осиротевших дубликатов (нет в images): {orphaned}')

# Show some examples if any
if orphaned > 0:
    examples = conn.execute('''
        SELECT dg.group_id, dg.image_id, dg.is_canonical
        FROM dedup_groups dg
        LEFT JOIN images i ON dg.image_id = i.id
        WHERE i.id IS NULL AND dg.is_canonical = 0
        LIMIT 5
    ''').fetchall()
    print('Примеры orphaned записей:')
    for ex in examples:
        group_id = ex["group_id"]
        image_id = ex["image_id"]
        is_canonical = ex["is_canonical"]
        print(f'  group_id={group_id}, image_id={image_id}, is_canonical={is_canonical}')

# Check total images
total_images = conn.execute('SELECT COUNT(*) FROM images').fetchone()[0]
print(f'\nВсего изображений в images: {total_images}')

# Check if there are canonical entries that are orphaned too
orphaned_canonical = conn.execute('''
    SELECT COUNT(*) 
    FROM dedup_groups dg
    LEFT JOIN images i ON dg.image_id = i.id
    WHERE i.id IS NULL AND dg.is_canonical = 1
''').fetchone()[0]
print(f'Осиротевших эталонов (is_canonical=1, нет в images): {orphaned_canonical}')

conn.close()