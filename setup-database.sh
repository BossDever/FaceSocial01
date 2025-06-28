#!/bin/bash

# FaceSocial Database Setup Script
# สำหรับติดตั้งและตั้งค่าฐานข้อมูล FaceSocial

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Database configuration
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-facesocial}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-password}

echo -e "${BLUE}🚀 FaceSocial Database Setup${NC}"
echo "=================================="
echo "Host: $DB_HOST:$DB_PORT"
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo ""

# Function to execute SQL file
execute_sql() {
    local file=$1
    local description=$2
    
    echo -e "${YELLOW}📄 Executing: $description${NC}"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$file"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Success: $description${NC}"
    else
        echo -e "${RED}❌ Failed: $description${NC}"
        exit 1
    fi
    echo ""
}

# Check if PostgreSQL is running
echo -e "${YELLOW}🔍 Checking PostgreSQL connection...${NC}"
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "SELECT version();" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PostgreSQL connection successful${NC}"
else
    echo -e "${RED}❌ Cannot connect to PostgreSQL${NC}"
    echo "Please check your database configuration and make sure PostgreSQL is running."
    exit 1
fi

# Create database if it doesn't exist
echo -e "${YELLOW}🏗️  Creating database if not exists...${NC}"
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || echo "Database already exists"

# Check if required extensions can be installed
echo -e "${YELLOW}🔌 Checking required PostgreSQL extensions...${NC}"
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" > /dev/null 2>&1
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS \"vector\";" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Extensions available${NC}"
else
    echo -e "${YELLOW}⚠️  Vector extension might not be available. Face embeddings will use FLOAT8[] instead.${NC}"
fi

# Navigate to database directory
cd "$(dirname "$0")/projec-final-fronend/database/init" || {
    echo -e "${RED}❌ Cannot find database init directory${NC}"
    exit 1
}

# Execute SQL files in order
echo -e "${BLUE}📊 Setting up database schema...${NC}"
echo ""

# 1. Basic schema
if [ -f "01_schema.sql" ]; then
    execute_sql "01_schema.sql" "Basic schema (users, face_embeddings, sessions)"
else
    echo -e "${RED}❌ 01_schema.sql not found${NC}"
    exit 1
fi

# 2. Social features
if [ -f "02_social_features.sql" ]; then
    execute_sql "02_social_features.sql" "Social features (posts, chat, face tags)"
else
    echo -e "${RED}❌ 02_social_features.sql not found${NC}"
    exit 1
fi

# 3. Sample data (optional)
if [ -f "03_sample_data.sql" ]; then
    read -p "$(echo -e ${YELLOW}"Do you want to insert sample data? [y/N]: "${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        execute_sql "03_sample_data.sql" "Sample data for testing"
    else
        echo -e "${BLUE}ℹ️  Skipping sample data${NC}"
    fi
fi

# Verify installation
echo -e "${YELLOW}🔍 Verifying installation...${NC}"
table_count=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';")

if [ $table_count -gt 15 ]; then
    echo -e "${GREEN}✅ Database setup completed successfully!${NC}"
    echo -e "${GREEN}📊 Created $table_count tables${NC}"
else
    echo -e "${RED}❌ Database setup may have failed. Only $table_count tables found.${NC}"
    exit 1
fi

# Show summary
echo ""
echo -e "${BLUE}📋 Setup Summary${NC}"
echo "=================================="
echo "✅ Basic user management"
echo "✅ Face recognition system"
echo "✅ Posts with face tagging"
echo "✅ Chat and messaging"
echo "✅ Social connections"
echo "✅ Notifications system"
echo "✅ User profiles and preferences"
echo ""

# Show connection info
echo -e "${GREEN}🔗 Database Connection Info${NC}"
echo "=================================="
echo "Host: $DB_HOST"
echo "Port: $DB_PORT"
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo ""

# Show next steps
echo -e "${BLUE}🚀 Next Steps${NC}"
echo "=================================="
echo "1. Update your .env file with database credentials"
echo "2. Install required packages: npm install"
echo "3. Start your application: npm run dev"
echo "4. Visit http://localhost:3000 to test"
echo ""
echo -e "${GREEN}🎉 Happy coding with FaceSocial!${NC}"
