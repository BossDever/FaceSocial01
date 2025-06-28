-- Migration: Add missing face_image_data column
-- Date: 2025-06-22
-- Description: Fix Prisma schema mismatch with database

-- Add face_image_data column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'face_embeddings' 
        AND column_name = 'face_image_data'
    ) THEN
        ALTER TABLE face_embeddings ADD COLUMN face_image_data TEXT;
        RAISE NOTICE 'Added face_image_data column to face_embeddings table';
    ELSE
        RAISE NOTICE 'face_image_data column already exists';
    END IF;
END $$;
