#!/usr/bin/env python3
"""
KeyHunt Smart Coordinator - Database Backup & Export Tool
==========================================================
Creates comprehensive backups of all scan data and state files.

Usage:
    python3 backup_database.py                    # Create backup
    python3 backup_database.py --restore backup.tar.gz  # Restore from backup
    python3 backup_database.py --export-sql       # Export to SQL format
    python3 backup_database.py --list             # List available backups

Version: 1.0.0
"""

import os
import sys
import sqlite3
import json
import shutil
import tarfile
import argparse
from datetime import datetime
from pathlib import Path

# Supported puzzle numbers
PUZZLE_NUMBERS = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80]

class DatabaseBackup:
    """Handles backup and restore operations for KeyHunt databases"""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.backup_dir = self.base_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

    def get_database_files(self):
        """Find all database files"""
        db_files = []
        for puzzle_num in PUZZLE_NUMBERS:
            db_path = self.base_dir / f"scan_data_puzzle_{puzzle_num}.db"
            if db_path.exists():
                db_files.append(db_path)
        return db_files

    def get_state_files(self):
        """Find all state JSON files"""
        state_files = []
        for puzzle_num in PUZZLE_NUMBERS:
            state_path = self.base_dir / f"keyhunt_state_puzzle_{puzzle_num}.json"
            if state_path.exists():
                state_files.append(state_path)
        return state_files

    def get_database_stats(self, db_path):
        """Get statistics from a database file"""
        stats = {
            'pool_scanned': 0,
            'my_scanned': 0,
            'size_bytes': os.path.getsize(db_path)
        }
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pool_scanned")
            stats['pool_scanned'] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM my_scanned")
            stats['my_scanned'] = cursor.fetchone()[0]
            conn.close()
        except Exception as e:
            print(f"  Warning: Could not read stats from {db_path}: {e}")
        return stats

    def create_backup(self, output_name=None):
        """Create a comprehensive backup archive"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_name:
            backup_name = output_name
        else:
            backup_name = f"keyhunt_backup_{timestamp}"

        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        temp_dir = self.backup_dir / f"temp_{timestamp}"
        temp_dir.mkdir(exist_ok=True)

        print("=" * 70)
        print("KEYHUNT DATABASE BACKUP TOOL")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base Directory: {self.base_dir}")
        print()

        # Copy database files
        db_files = self.get_database_files()
        print(f"Found {len(db_files)} database file(s):")
        total_pool = 0
        total_my = 0

        for db_path in db_files:
            stats = self.get_database_stats(db_path)
            total_pool += stats['pool_scanned']
            total_my += stats['my_scanned']

            dest = temp_dir / db_path.name
            shutil.copy2(db_path, dest)

            size_kb = stats['size_bytes'] / 1024
            print(f"  - {db_path.name}")
            print(f"      Pool blocks: {stats['pool_scanned']:,}")
            print(f"      My blocks: {stats['my_scanned']:,}")
            print(f"      Size: {size_kb:.1f} KB")

        print()

        # Copy state files
        state_files = self.get_state_files()
        print(f"Found {len(state_files)} state file(s):")

        for state_path in state_files:
            dest = temp_dir / state_path.name
            shutil.copy2(state_path, dest)

            # Read state info
            try:
                with open(state_path) as f:
                    state = json.load(f)
                print(f"  - {state_path.name}")
                print(f"      Block index: {state.get('current_block_index', 'N/A'):,}")
                print(f"      Range: {state.get('range_start', 'N/A')} - {state.get('range_end', 'N/A')}")
            except:
                print(f"  - {state_path.name} (could not read)")

        print()

        # Create metadata file
        metadata = {
            'backup_date': datetime.now().isoformat(),
            'backup_tool_version': '1.0.0',
            'coordinator_version': '3.8.0',
            'databases': [f.name for f in db_files],
            'state_files': [f.name for f in state_files],
            'statistics': {
                'total_pool_blocks': total_pool,
                'total_my_blocks': total_my,
                'database_count': len(db_files),
                'state_file_count': len(state_files)
            }
        }

        metadata_path = temp_dir / "backup_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create tar.gz archive
        print("Creating archive...")
        with tarfile.open(backup_path, "w:gz") as tar:
            for item in temp_dir.iterdir():
                tar.add(item, arcname=item.name)

        # Cleanup temp directory
        shutil.rmtree(temp_dir)

        # Print summary
        backup_size = os.path.getsize(backup_path)
        print()
        print("=" * 70)
        print("BACKUP COMPLETE")
        print("=" * 70)
        print(f"Archive: {backup_path}")
        print(f"Size: {backup_size / 1024:.1f} KB ({backup_size / 1024 / 1024:.2f} MB)")
        print()
        print("Summary:")
        print(f"  - Total pool blocks: {total_pool:,}")
        print(f"  - Total my blocks: {total_my:,}")
        print(f"  - Database files: {len(db_files)}")
        print(f"  - State files: {len(state_files)}")
        print()
        print("To restore on another machine:")
        print(f"  python3 backup_database.py --restore {backup_path.name}")
        print("=" * 70)

        return backup_path

    def restore_backup(self, backup_file):
        """Restore from a backup archive"""
        backup_path = Path(backup_file)

        if not backup_path.exists():
            # Check in backups directory
            backup_path = self.backup_dir / backup_file

        if not backup_path.exists():
            print(f"Error: Backup file not found: {backup_file}")
            return False

        print("=" * 70)
        print("KEYHUNT DATABASE RESTORE")
        print("=" * 70)
        print(f"Restoring from: {backup_path}")
        print(f"Target directory: {self.base_dir}")
        print()

        # Extract archive
        temp_dir = self.backup_dir / f"restore_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(exist_ok=True)

        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(temp_dir)

            # Read metadata
            metadata_path = temp_dir / "backup_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                print(f"Backup date: {metadata.get('backup_date', 'Unknown')}")
                print(f"Coordinator version: {metadata.get('coordinator_version', 'Unknown')}")
                print()

            # Restore files
            restored_count = 0
            for item in temp_dir.iterdir():
                if item.name == "backup_metadata.json":
                    continue

                dest = self.base_dir / item.name

                # Backup existing file if present
                if dest.exists():
                    backup_existing = dest.with_suffix(dest.suffix + '.bak')
                    print(f"  Backing up existing: {dest.name} -> {backup_existing.name}")
                    shutil.move(dest, backup_existing)

                shutil.copy2(item, dest)
                print(f"  Restored: {item.name}")
                restored_count += 1

            print()
            print("=" * 70)
            print("RESTORE COMPLETE")
            print("=" * 70)
            print(f"Files restored: {restored_count}")
            print()

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)

        return True

    def export_to_sql(self, output_file=None):
        """Export all databases to SQL format for maximum portability"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_file:
            sql_path = Path(output_file)
        else:
            sql_path = self.backup_dir / f"keyhunt_export_{timestamp}.sql"

        print("=" * 70)
        print("KEYHUNT DATABASE SQL EXPORT")
        print("=" * 70)

        db_files = self.get_database_files()

        with open(sql_path, 'w') as sql_file:
            sql_file.write("-- KeyHunt Smart Coordinator Database Export\n")
            sql_file.write(f"-- Generated: {datetime.now().isoformat()}\n")
            sql_file.write(f"-- Coordinator Version: 3.8.0\n")
            sql_file.write("-- \n\n")

            for db_path in db_files:
                puzzle_num = db_path.stem.split('_')[-1]
                print(f"Exporting {db_path.name}...")

                sql_file.write(f"-- =========================================\n")
                sql_file.write(f"-- Database: {db_path.name}\n")
                sql_file.write(f"-- Puzzle: {puzzle_num}\n")
                sql_file.write(f"-- =========================================\n\n")

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Export pool_scanned
                cursor.execute("SELECT * FROM pool_scanned")
                rows = cursor.fetchall()

                if rows:
                    sql_file.write(f"-- Pool scanned blocks ({len(rows)} records)\n")
                    sql_file.write(f"-- For puzzle {puzzle_num}: INSERT INTO pool_scanned VALUES ...\n")

                    for row in rows:
                        values = ', '.join([f"'{v}'" if v else 'NULL' for v in row])
                        sql_file.write(f"INSERT OR REPLACE INTO pool_scanned VALUES ({values});\n")

                    sql_file.write("\n")

                # Export my_scanned
                cursor.execute("SELECT * FROM my_scanned")
                rows = cursor.fetchall()

                if rows:
                    sql_file.write(f"-- My scanned blocks ({len(rows)} records)\n")

                    for row in rows:
                        values = ', '.join([f"'{v}'" if isinstance(v, str) else (str(v) if v else 'NULL') for v in row])
                        sql_file.write(f"INSERT OR REPLACE INTO my_scanned VALUES ({values});\n")

                    sql_file.write("\n")

                conn.close()

        print()
        print(f"SQL export saved to: {sql_path}")
        print(f"Size: {os.path.getsize(sql_path) / 1024:.1f} KB")

        return sql_path

    def list_backups(self):
        """List all available backups"""
        print("=" * 70)
        print("AVAILABLE BACKUPS")
        print("=" * 70)

        backups = list(self.backup_dir.glob("*.tar.gz"))

        if not backups:
            print("No backups found.")
            print(f"Backup directory: {self.backup_dir}")
            return

        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for backup in backups:
            stat = backup.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            size_mb = stat.st_size / 1024 / 1024

            print(f"  {backup.name}")
            print(f"      Date: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      Size: {size_mb:.2f} MB")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="KeyHunt Smart Coordinator - Database Backup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 backup_database.py                     # Create backup
  python3 backup_database.py --name my_backup    # Create named backup
  python3 backup_database.py --restore backup.tar.gz
  python3 backup_database.py --export-sql
  python3 backup_database.py --list
        """
    )

    parser.add_argument('--restore', metavar='FILE', help='Restore from backup file')
    parser.add_argument('--export-sql', action='store_true', help='Export to SQL format')
    parser.add_argument('--list', action='store_true', help='List available backups')
    parser.add_argument('--name', metavar='NAME', help='Custom backup name')
    parser.add_argument('--dir', metavar='DIR', help='Base directory (default: script directory)')

    args = parser.parse_args()

    backup = DatabaseBackup(base_dir=args.dir)

    if args.restore:
        backup.restore_backup(args.restore)
    elif args.export_sql:
        backup.export_to_sql()
    elif args.list:
        backup.list_backups()
    else:
        backup.create_backup(output_name=args.name)


if __name__ == "__main__":
    main()
