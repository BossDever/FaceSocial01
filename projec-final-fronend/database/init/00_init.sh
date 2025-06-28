#!/bin/bash
# Database initialization script for Docker
# This script will run when PostgreSQL container starts for the first time

set -e

echo "🚀 Initializing FaceSocial Database..."

# Create extensions
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Enable required extensions
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    -- Try to create vector extension (might not be available in Alpine)
    DO \$\$
    BEGIN
        CREATE EXTENSION IF NOT EXISTS "vector";
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'Vector extension not available, using FLOAT8[] for embeddings';
    END
    \$\$;
    
    -- Show installed extensions
    SELECT extname, extversion FROM pg_extension;
EOSQL

echo "✅ Database extensions installed"

# Execute the main schema files in order
if [ -f "/docker-entrypoint-initdb.d/01_schema.sql" ]; then
    echo "📄 Executing basic schema..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "/docker-entrypoint-initdb.d/01_schema.sql"
    echo "✅ Basic schema completed"
fi

if [ -f "/docker-entrypoint-initdb.d/02_social_features.sql" ]; then
    echo "📄 Executing social features schema..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "/docker-entrypoint-initdb.d/02_social_features.sql"
    echo "✅ Social features schema completed"
fi

if [ -f "/docker-entrypoint-initdb.d/03_sample_data.sql" ]; then
    echo "📄 Executing sample data..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "/docker-entrypoint-initdb.d/03_sample_data.sql"
    echo "✅ Sample data inserted"
fi

# Verify installation
echo "🔍 Verifying database setup..."
table_count=$(psql -t --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';")

if [ $table_count -gt 15 ]; then
    echo "✅ Database setup completed successfully! Created $table_count tables"
else
    echo "⚠️ Warning: Only $table_count tables found. Setup may be incomplete."
fi

echo "🎉 FaceSocial database is ready!"
