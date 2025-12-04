#!/usr/bin/env python3
"""
Clear Database Utility

This script removes all stored face data from the output directory,
effectively resetting the face recognition database.

Usage:
    python recog/clear_database.py [--confirm]

Options:
    --confirm    Skip confirmation prompt and delete immediately
"""

import os
import shutil
import argparse

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def get_database_stats():
    """Get statistics about the current database."""
    if not os.path.exists(OUTPUT_DIR):
        return 0, 0

    person_count = 0
    total_images = 0

    for dirname in os.listdir(OUTPUT_DIR):
        if dirname.startswith("person_"):
            person_count += 1
            person_dir = os.path.join(OUTPUT_DIR, dirname)
            # Count image files
            for filename in os.listdir(person_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    total_images += 1

    return person_count, total_images


def clear_database(confirm=False):
    """Clear all face data from the output directory."""

    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print("No database found. Output directory does not exist.")
        return

    # Get current stats
    person_count, image_count = get_database_stats()

    if person_count == 0:
        print("Database is already empty.")
        return

    # Show current state
    print("=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total persons: {person_count}")
    print(f"Total images:  {image_count}")
    print(f"Location:      {OUTPUT_DIR}")
    print("=" * 60)

    # Confirm deletion
    if not confirm:
        response = input("\nAre you sure you want to delete ALL face data? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return

    # Delete all person directories
    deleted_count = 0
    for dirname in os.listdir(OUTPUT_DIR):
        if dirname.startswith("person_"):
            person_dir = os.path.join(OUTPUT_DIR, dirname)
            try:
                shutil.rmtree(person_dir)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {dirname}: {e}")

    print(f"\n✓ Successfully deleted {deleted_count} person(s)")
    print("✓ Database cleared. Ready for fresh start.")


def main():
    parser = argparse.ArgumentParser(
        description="Clear all stored face recognition data"
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt and delete immediately'
    )

    args = parser.parse_args()

    clear_database(confirm=args.confirm)


if __name__ == "__main__":
    main()
