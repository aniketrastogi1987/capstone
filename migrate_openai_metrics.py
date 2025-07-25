#!/usr/bin/env python3
"""
Migration script to add OpenAI metrics columns to existing databases
"""

import sqlite3
import psycopg2
import logging
from monitoring.postgres_monitor import PostgresMonitor
from monitoring.session_database import SessionDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_sqlite_database():
    """Add OpenAI metrics columns to SQLite session database"""
    try:
        print("🔄 Migrating SQLite database...")
        
        # Connect to SQLite database
        conn = sqlite3.connect("sessions.db")
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(session_interactions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add OpenAI columns if they don't exist
        openai_columns = [
            ("openai_validation_score", "REAL DEFAULT 0.0"),
            ("openai_hallucination_detected", "BOOLEAN DEFAULT FALSE"),
            ("openai_validation_time", "REAL DEFAULT 0.0"),
            ("openai_corrections_applied", "BOOLEAN DEFAULT FALSE"),
            ("openai_validation_success", "BOOLEAN DEFAULT FALSE"),
            ("openai_validation_details", "TEXT")
        ]
        
        for column_name, column_type in openai_columns:
            if column_name not in columns:
                cursor.execute(f"ALTER TABLE session_interactions ADD COLUMN {column_name} {column_type}")
                print(f"✅ Added column: {column_name}")
            else:
                print(f"ℹ️ Column already exists: {column_name}")
        
        conn.commit()
        conn.close()
        print("✅ SQLite migration completed!")
        
    except Exception as e:
        print(f"❌ Error migrating SQLite database: {e}")

def migrate_postgres_database():
    """Add OpenAI metrics columns to PostgreSQL monitoring database"""
    try:
        print("🔄 Migrating PostgreSQL database...")
        
        # Initialize PostgresMonitor to get connection
        postgres_monitor = PostgresMonitor()
        conn = postgres_monitor._get_connection()
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'chat_metrics'
        """)
        columns = [column[0] for column in cursor.fetchall()]
        
        # Add OpenAI columns if they don't exist
        openai_columns = [
            ("openai_validation_score", "REAL DEFAULT 0.0"),
            ("openai_hallucination_detected", "BOOLEAN DEFAULT FALSE"),
            ("openai_validation_time", "REAL DEFAULT 0.0"),
            ("openai_corrections_applied", "BOOLEAN DEFAULT FALSE"),
            ("openai_validation_success", "BOOLEAN DEFAULT FALSE"),
            ("openai_validation_details", "JSONB")
        ]
        
        for column_name, column_type in openai_columns:
            if column_name not in columns:
                cursor.execute(f"ALTER TABLE chat_metrics ADD COLUMN {column_name} {column_type}")
                print(f"✅ Added column: {column_name}")
            else:
                print(f"ℹ️ Column already exists: {column_name}")
        
        conn.commit()
        conn.close()
        print("✅ PostgreSQL migration completed!")
        
    except Exception as e:
        print(f"❌ Error migrating PostgreSQL database: {e}")

def verify_migration():
    """Verify that the migration was successful"""
    print("\n🔍 Verifying migration...")
    
    # Check SQLite
    try:
        conn = sqlite3.connect("sessions.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(session_interactions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        openai_columns = [
            "openai_validation_score",
            "openai_hallucination_detected", 
            "openai_validation_time",
            "openai_corrections_applied",
            "openai_validation_success",
            "openai_validation_details"
        ]
        
        missing_columns = [col for col in openai_columns if col not in columns]
        if missing_columns:
            print(f"❌ Missing SQLite columns: {missing_columns}")
        else:
            print("✅ All SQLite OpenAI columns present")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error verifying SQLite: {e}")
    
    # Check PostgreSQL
    try:
        postgres_monitor = PostgresMonitor()
        conn = postgres_monitor._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'chat_metrics'
        """)
        columns = [column[0] for column in cursor.fetchall()]
        
        openai_columns = [
            "openai_validation_score",
            "openai_hallucination_detected", 
            "openai_validation_time",
            "openai_corrections_applied",
            "openai_validation_success",
            "openai_validation_details"
        ]
        
        missing_columns = [col for col in openai_columns if col not in columns]
        if missing_columns:
            print(f"❌ Missing PostgreSQL columns: {missing_columns}")
        else:
            print("✅ All PostgreSQL OpenAI columns present")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error verifying PostgreSQL: {e}")

def main():
    """Run the migration"""
    print("🚀 Starting OpenAI Metrics Database Migration")
    print("=" * 50)
    
    # Migrate databases
    migrate_sqlite_database()
    migrate_postgres_database()
    
    # Verify migration
    verify_migration()
    
    print("\n✅ Migration completed!")

if __name__ == "__main__":
    main() 