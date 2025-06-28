# ชุดคำสั่ง: การจัดการฐานข้อมูล
## Database Management และ PostgreSQL Schema

### 📋 สารบัญ
6.1 [ภาพรวม Database Architecture](#61-ภาพรวม-database-architecture)
6.2 [PostgreSQL Schema Design](#62-postgresql-schema-design)
6.3 [Social Media Features](#63-social-media-features)
6.4 [Face Embeddings Storage](#64-face-embeddings-storage)
6.5 [Database Functions](#65-database-functions)
6.6 [Performance Optimization](#66-performance-optimization)
6.7 [Backup และ Recovery](#67-backup-และ-recovery)
6.8 [Database Scripts](#68-database-scripts)

---

## 6.1 ภาพรวม Database Architecture

ระบบฐานข้อมูล PostgreSQL ที่ออกแบบมาสำหรับ Social Media Platform พร้อม Face Recognition features

### 🗄️ โครงสร้างฐานข้อมูล
- **User Management**: ข้อมูลผู้ใช้และโปรไฟล์
- **Social Features**: Posts, Comments, Likes, Shares
- **Face Data**: Face embeddings และ detection logs  
- **Performance**: Indexing และ optimization สำหรับ large scale

---

## 6.2 PostgreSQL Schema Design

### 6.2.1 Database Schema สำหรับระบบ FaceSocial

```sql
-- FaceSocial Database Schema
-- สร้างฐานข้อมูลสำหรับระบบ FaceSocial

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Users table (ข้อมูลผู้ใช้)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    date_of_birth DATE,
    profile_image_url TEXT,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    email_verified_at TIMESTAMP,
    phone_verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP
);

-- Face embeddings table (ข้อมูล face embedding)
CREATE TABLE face_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding_model VARCHAR(50) NOT NULL DEFAULT 'facenet', -- 'facenet' or 'adaface'
    embedding_data FLOAT8[] NOT NULL, -- Array of embedding vectors
    image_url TEXT,
    image_hash VARCHAR(64), -- MD5 hash ของภาพ
    quality_score FLOAT DEFAULT 0.0, -- คะแนนคุณภาพของภาพ
    detection_confidence FLOAT DEFAULT 0.0, -- ความมั่นใจในการตรวจจับ
    landmark_data JSONB, -- Face landmarks data
    is_primary BOOLEAN DEFAULT false, -- embedding หลักของผู้ใช้
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions table (การจัดการ session)
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address INET,
    user_agent TEXT,
    device_info JSONB,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Login attempts table (ป้องกัน brute force)
CREATE TABLE login_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255),
    ip_address INET NOT NULL,
    success BOOLEAN DEFAULT false,
    failure_reason VARCHAR(100),
    user_agent TEXT,
    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User preferences table
CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    language VARCHAR(10) DEFAULT 'th',
    timezone VARCHAR(50) DEFAULT 'Asia/Bangkok',
    theme VARCHAR(20) DEFAULT 'light',
    notifications_enabled BOOLEAN DEFAULT true,
    email_notifications BOOLEAN DEFAULT true,
    push_notifications BOOLEAN DEFAULT true,
    privacy_face_recognition BOOLEAN DEFAULT true,
    privacy_auto_tagging BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_face_embeddings_user_id ON face_embeddings(user_id);
CREATE INDEX idx_face_embeddings_model ON face_embeddings(embedding_model);
CREATE INDEX idx_face_embeddings_primary ON face_embeddings(is_primary);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_login_attempts_ip ON login_attempts(ip_address);
CREATE INDEX idx_login_attempts_email ON login_attempts(email);
```

### 1.2 ชุดคำสั่ง Social Features Schema

```sql
-- FaceSocial - Social Features Database Schema
-- ฐานข้อมูลสำหรับระบบ แชท, ฟีด, และโปรไฟล์

-- Posts table (โพสต์)
CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT,
    image_urls TEXT[], -- Array ของ URLs รูปภาพ
    video_url TEXT,
    location VARCHAR(255),
    privacy_level VARCHAR(20) DEFAULT 'public', -- 'public', 'friends', 'private'
    is_archived BOOLEAN DEFAULT false,
    likes_count INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    shares_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face tags in posts (แท็กใบหน้าในโพสต์)
CREATE TABLE post_face_tags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    tagged_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tagger_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL, -- รูปที่มีการแท็ก
    face_bbox JSONB, -- Bounding box ของใบหน้า {x, y, width, height}
    face_embedding FLOAT8[], -- Face embedding ที่ใช้ในการแท็ก
    confidence_score FLOAT DEFAULT 0.0, -- ความมั่นใจในการแท็ก
    is_confirmed BOOLEAN DEFAULT false, -- ยืนยันโดยผู้ถูกแท็ก
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Post likes (การกดไลค์)
CREATE TABLE post_likes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(post_id, user_id)
);

-- Post comments (ความคิดเห็น)
CREATE TABLE post_comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    parent_comment_id UUID REFERENCES post_comments(id) ON DELETE CASCADE, -- สำหรับ reply
    content TEXT NOT NULL,
    image_url TEXT,
    likes_count INTEGER DEFAULT 0,
    replies_count INTEGER DEFAULT 0,
    is_edited BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User connections/friends (เพื่อน)
CREATE TABLE user_connections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    requester_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    addressee_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'accepted', 'rejected', 'blocked'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(requester_id, addressee_id)
);

-- Chat conversations (การสนทนา)
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(20) DEFAULT 'direct', -- 'direct', 'group'
    name VARCHAR(255), -- ชื่อกลุ่ม (สำหรับ group chat)
    description TEXT,
    image_url TEXT, -- รูปกลุ่ม
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT true,
    last_message_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat participants (ผู้เข้าร่วมแชท)
CREATE TABLE conversation_participants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) DEFAULT 'member', -- 'admin', 'member'
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_read_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Messages (ข้อความ)
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT,
    message_type VARCHAR(20) DEFAULT 'text', -- 'text', 'image', 'video', 'file', 'voice'
    file_url TEXT,
    file_name VARCHAR(255),
    file_size INTEGER,
    reply_to_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    is_edited BOOLEAN DEFAULT false,
    is_deleted BOOLEAN DEFAULT false,
    delivered_at TIMESTAMP,
    read_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Notifications (การแจ้งเตือน)
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL, -- 'face_tag', 'friend_request', 'post_like', 'comment', 'message'
    title VARCHAR(255) NOT NULL,
    body TEXT,
    data JSONB, -- Additional data
    is_read BOOLEAN DEFAULT false,
    action_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for social features
CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_created_at ON posts(created_at);
CREATE INDEX idx_posts_privacy ON posts(privacy_level);
CREATE INDEX idx_face_tags_post_id ON post_face_tags(post_id);
CREATE INDEX idx_face_tags_tagged_user ON post_face_tags(tagged_user_id);
CREATE INDEX idx_face_tags_status ON post_face_tags(status);
CREATE INDEX idx_post_likes_post_user ON post_likes(post_id, user_id);
CREATE INDEX idx_comments_post_id ON post_comments(post_id);
CREATE INDEX idx_connections_users ON user_connections(requester_id, addressee_id);
CREATE INDEX idx_connections_status ON user_connections(status);
CREATE INDEX idx_conversations_created_by ON conversations(created_by);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_sender ON messages(sender_id);
CREATE INDEX idx_notifications_user ON notifications(user_id);
CREATE INDEX idx_notifications_read ON notifications(is_read);
```

## 6.3 ชุดคำสั่งการติดตั้งและตั้งค่าฐานข้อมูล

### 2.1 ชุดคำสั่ง Database Setup Script

```bash
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
    echo "Please check your connection settings:"
    echo "  Host: $DB_HOST"
    echo "  Port: $DB_PORT"
    echo "  User: $DB_USER"
    exit 1
fi

# Create database if it doesn't exist
echo -e "${YELLOW}🗃️ Creating database if not exists...${NC}"
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 || PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME"
echo -e "${GREEN}✅ Database ready${NC}"
echo ""

# Execute schema files
echo -e "${BLUE}📊 Setting up database schema...${NC}"

# Core schema
execute_sql "database/init/01_schema.sql" "Core database schema"

# Social features
execute_sql "database/init/02_social_features.sql" "Social features schema"

# Sample data (optional)
if [ "$1" = "--with-sample-data" ]; then
    echo -e "${BLUE}📝 Adding sample data...${NC}"
    execute_sql "database/init/03_sample_data.sql" "Sample data"
fi

# Run migrations
echo -e "${BLUE}🔄 Running migrations...${NC}"
for migration in database/migrations/*.sql; do
    if [ -f "$migration" ]; then
        filename=$(basename "$migration")
        execute_sql "$migration" "Migration: $filename"
    fi
done

echo -e "${GREEN}🎉 Database setup completed successfully!${NC}"
echo ""
echo "Database Information:"
echo "  Host: $DB_HOST:$DB_PORT"
echo "  Database: $DB_NAME"
echo "  Tables created: $(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")"
echo ""
echo "Next steps:"
echo "1. Update your application configuration with database credentials"
echo "2. Start your application server"
echo "3. Test the face recognition features"
```

### 2.2 ชุดคำสั่ง Environment Configuration

```bash
# .env.example
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=facesocial
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_SSL=false

# Application Configuration
NODE_ENV=development
APP_PORT=3000
APP_HOST=localhost
APP_SECRET=your_jwt_secret_key

# Face Recognition API
FACE_API_URL=http://localhost:8000
FACE_API_TIMEOUT=30000

# File Upload
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp

# Redis (for caching and sessions)
REDIS_URL=redis://localhost:6379
REDIS_PREFIX=facesocial:

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
EMAIL_FROM=noreply@facesocial.com

# Storage Configuration (AWS S3 or local)
STORAGE_TYPE=local  # local, s3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
AWS_S3_BUCKET=facesocial-uploads
```

## 6.4 ชุดคำสั่งการจัดการข้อมูล Face Embeddings

### 3.1 ชุดคำสั่งการบันทึก Face Embeddings

```sql
-- Function to insert face embedding
CREATE OR REPLACE FUNCTION insert_face_embedding(
    p_user_id UUID,
    p_embedding_model VARCHAR,
    p_embedding_data FLOAT8[],
    p_image_url TEXT,
    p_image_hash VARCHAR,
    p_quality_score FLOAT,
    p_detection_confidence FLOAT,
    p_is_primary BOOLEAN DEFAULT false
) RETURNS UUID AS $$
DECLARE
    embedding_id UUID;
BEGIN
    -- If this is set as primary, unset other primary embeddings for this user
    IF p_is_primary THEN
        UPDATE face_embeddings 
        SET is_primary = false 
        WHERE user_id = p_user_id AND embedding_model = p_embedding_model;
    END IF;
    
    -- Insert new embedding
    INSERT INTO face_embeddings (
        user_id,
        embedding_model,
        embedding_data,
        image_url,
        image_hash,
        quality_score,
        detection_confidence,
        is_primary
    ) VALUES (
        p_user_id,
        p_embedding_model,
        p_embedding_data,
        p_image_url,
        p_image_hash,
        p_quality_score,
        p_detection_confidence,
        p_is_primary
    ) RETURNING id INTO embedding_id;
    
    RETURN embedding_id;
END;
$$ LANGUAGE plpgsql;

-- Function to find similar face embeddings using cosine similarity
CREATE OR REPLACE FUNCTION find_similar_faces(
    p_embedding FLOAT8[],
    p_model VARCHAR DEFAULT 'facenet',
    p_threshold FLOAT DEFAULT 0.8,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE (
    user_id UUID,
    embedding_id UUID,
    similarity FLOAT,
    quality_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fe.user_id,
        fe.id as embedding_id,
        (fe.embedding_data <#> p_embedding::vector) as similarity,
        fe.quality_score
    FROM face_embeddings fe
    WHERE fe.embedding_model = p_model
      AND (fe.embedding_data <#> p_embedding::vector) <= (1 - p_threshold)
    ORDER BY similarity ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to get user's best quality embedding
CREATE OR REPLACE FUNCTION get_user_best_embedding(
    p_user_id UUID,
    p_model VARCHAR DEFAULT 'facenet'
) RETURNS TABLE (
    embedding_id UUID,
    embedding_data FLOAT8[],
    quality_score FLOAT,
    is_primary BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fe.id,
        fe.embedding_data,
        fe.quality_score,
        fe.is_primary
    FROM face_embeddings fe
    WHERE fe.user_id = p_user_id 
      AND fe.embedding_model = p_model
    ORDER BY 
        fe.is_primary DESC,
        fe.quality_score DESC,
        fe.created_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;
```

### 3.2 ชุดคำสั่งการจัดการ Face Tags

```sql
-- Function to create face tag with automatic recognition
CREATE OR REPLACE FUNCTION create_face_tag(
    p_post_id UUID,
    p_tagger_user_id UUID,
    p_image_url TEXT,
    p_face_bbox JSONB,
    p_face_embedding FLOAT8[]
) RETURNS TABLE (
    tag_id UUID,
    suggested_user_id UUID,
    confidence_score FLOAT
) AS $$
DECLARE
    new_tag_id UUID;
    similar_face RECORD;
BEGIN
    -- Create the face tag
    INSERT INTO post_face_tags (
        post_id,
        tagger_user_id,
        image_url,
        face_bbox,
        face_embedding,
        status
    ) VALUES (
        p_post_id,
        p_tagger_user_id,
        p_image_url,
        p_face_bbox,
        p_face_embedding,
        'pending'
    ) RETURNING id INTO new_tag_id;
    
    -- Find similar faces
    SELECT * INTO similar_face
    FROM find_similar_faces(p_face_embedding, 'facenet', 0.8, 1)
    LIMIT 1;
    
    -- If found similar face, suggest the user
    IF similar_face.user_id IS NOT NULL THEN
        UPDATE post_face_tags 
        SET 
            tagged_user_id = similar_face.user_id,
            confidence_score = 1 - similar_face.similarity,
            status = 'suggested'
        WHERE id = new_tag_id;
        
        RETURN QUERY SELECT new_tag_id, similar_face.user_id, (1 - similar_face.similarity);
    ELSE
        RETURN QUERY SELECT new_tag_id, NULL::UUID, 0.0;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to confirm face tag
CREATE OR REPLACE FUNCTION confirm_face_tag(
    p_tag_id UUID,
    p_user_id UUID  -- User who is confirming (should be the tagged user)
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE post_face_tags 
    SET 
        status = 'approved',
        is_confirmed = true,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_tag_id 
      AND tagged_user_id = p_user_id 
      AND status IN ('pending', 'suggested');
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;
```

## 6.5 ชุดคำสั่งการดูแลรักษาฐานข้อมูล

### 4.1 ชุดคำสั่ง Database Maintenance

```sql
-- Clean up old login attempts (keep only last 30 days)
CREATE OR REPLACE FUNCTION cleanup_login_attempts() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM login_attempts 
    WHERE attempted_at < (CURRENT_TIMESTAMP - INTERVAL '30 days');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions 
    WHERE expires_at < CURRENT_TIMESTAMP OR is_active = false;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Update user statistics
CREATE OR REPLACE FUNCTION update_user_statistics() RETURNS VOID AS $$
BEGIN
    -- Update posts count, friends count, etc.
    -- This would typically update a user_stats table
    
    -- Update posts counts
    UPDATE posts SET 
        likes_count = (SELECT COUNT(*) FROM post_likes WHERE post_id = posts.id),
        comments_count = (SELECT COUNT(*) FROM post_comments WHERE post_id = posts.id);
        
    -- Update comments counts
    UPDATE post_comments SET
        likes_count = (SELECT COUNT(*) FROM comment_likes WHERE comment_id = post_comments.id),
        replies_count = (SELECT COUNT(*) FROM post_comments replies WHERE replies.parent_comment_id = post_comments.id);
END;
$$ LANGUAGE plpgsql;

-- Vacuum and analyze tables for better performance
CREATE OR REPLACE FUNCTION maintenance_vacuum() RETURNS VOID AS $$
BEGIN
    VACUUM ANALYZE users;
    VACUUM ANALYZE face_embeddings;
    VACUUM ANALYZE posts;
    VACUUM ANALYZE post_face_tags;
    VACUUM ANALYZE messages;
    VACUUM ANALYZE notifications;
END;
$$ LANGUAGE plpgsql;
```

### 4.2 ชุดคำสั่ง Backup และ Restore

```bash
#!/bin/bash

# Database Backup Script
BACKUP_DIR="/var/backups/facesocial"
DB_NAME="facesocial"
DB_USER="postgres"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup directory
mkdir -p $BACKUP_DIR

# Full database backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME | gzip > "$BACKUP_DIR/facesocial_full_$TIMESTAMP.sql.gz"

# Schema-only backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME --schema-only > "$BACKUP_DIR/facesocial_schema_$TIMESTAMP.sql"

# Data-only backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME --data-only | gzip > "$BACKUP_DIR/facesocial_data_$TIMESTAMP.sql.gz"

# Clean up old backups (keep last 7 days)
find $BACKUP_DIR -name "facesocial_*.sql.gz" -mtime +7 -delete

echo "Backup completed: $TIMESTAMP"
```

```bash
#!/bin/bash

# Database Restore Script
if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

BACKUP_FILE=$1
DB_NAME="facesocial"
DB_USER="postgres"

echo "Restoring database from: $BACKUP_FILE"

# Drop and recreate database
dropdb -h localhost -U $DB_USER $DB_NAME
createdb -h localhost -U $DB_USER $DB_NAME

# Restore from backup
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | psql -h localhost -U $DB_USER -d $DB_NAME
else
    psql -h localhost -U $DB_USER -d $DB_NAME < $BACKUP_FILE
fi

echo "Database restored successfully"
```

---

*เอกสารนี้แสดงชุดคำสั่งหลักสำหรับการจัดการฐานข้อมูลของระบบ FaceSocial รวมถึงการสร้าง schema, การติดตั้ง, การจัดการ face embeddings, และการดูแลรักษาระบบ*
