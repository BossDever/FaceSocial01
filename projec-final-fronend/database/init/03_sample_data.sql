-- Sample data for FaceSocial
-- ข้อมูลตัวอย่างสำหรับทดสอบระบบ

-- เพิ่มผู้ใช้ตัวอย่าง
INSERT INTO users (
    username, email, password_hash, first_name, last_name, 
    is_active, is_verified, email_verified_at
) VALUES 
    ('john_doe', 'john@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeN8WS5wC2.P5x2oS', 'John', 'Doe', true, true, CURRENT_TIMESTAMP),
    ('jane_smith', 'jane@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeN8WS5wC2.P5x2oS', 'Jane', 'Smith', true, true, CURRENT_TIMESTAMP),
    ('bob_wilson', 'bob@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeN8WS5wC2.P5x2oS', 'Bob', 'Wilson', true, true, CURRENT_TIMESTAMP),
    ('alice_brown', 'alice@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeN8WS5wC2.P5x2oS', 'Alice', 'Brown', true, true, CURRENT_TIMESTAMP);

-- เพิ่ม user preferences สำหรับผู้ใช้ใหม่
INSERT INTO user_preferences (user_id) 
SELECT id FROM users WHERE username IN ('john_doe', 'jane_smith', 'bob_wilson', 'alice_brown');

-- เพิ่ม user profiles
INSERT INTO user_profiles (user_id, bio, location, work)
SELECT 
    u.id,
    CASE u.username
        WHEN 'john_doe' THEN 'Software Developer | Love photography and travel'
        WHEN 'jane_smith' THEN 'Graphic Designer | Art enthusiast'
        WHEN 'bob_wilson' THEN 'Marketing Manager | Coffee lover'
        WHEN 'alice_brown' THEN 'Data Scientist | Tech blogger'
    END,
    CASE u.username
        WHEN 'john_doe' THEN 'Bangkok, Thailand'
        WHEN 'jane_smith' THEN 'Chiang Mai, Thailand'
        WHEN 'bob_wilson' THEN 'Phuket, Thailand'
        WHEN 'alice_brown' THEN 'Bangkok, Thailand'
    END,
    CASE u.username
        WHEN 'john_doe' THEN 'Tech Startup'
        WHEN 'jane_smith' THEN 'Creative Agency'
        WHEN 'bob_wilson' THEN 'Marketing Firm'
        WHEN 'alice_brown' THEN 'AI Research Lab'
    END
FROM users u
WHERE u.username IN ('john_doe', 'jane_smith', 'bob_wilson', 'alice_brown');

-- เพิ่มการเชื่อมต่อระหว่างผู้ใช้ (เพื่อน)
INSERT INTO user_connections (requester_id, addressee_id, status, updated_at)
SELECT 
    u1.id, u2.id, 'accepted', CURRENT_TIMESTAMP
FROM users u1, users u2
WHERE 
    (u1.username = 'john_doe' AND u2.username = 'jane_smith') OR
    (u1.username = 'john_doe' AND u2.username = 'bob_wilson') OR
    (u1.username = 'jane_smith' AND u2.username = 'alice_brown') OR
    (u1.username = 'bob_wilson' AND u2.username = 'alice_brown');

-- เพิ่มโพสต์ตัวอย่าง
DO $$
DECLARE
    john_id UUID;
    jane_id UUID;
    bob_id UUID;
    alice_id UUID;
    post1_id UUID;
    post2_id UUID;
    post3_id UUID;
BEGIN
    -- ดึง user IDs
    SELECT id INTO john_id FROM users WHERE username = 'john_doe';
    SELECT id INTO jane_id FROM users WHERE username = 'jane_smith';
    SELECT id INTO bob_id FROM users WHERE username = 'bob_wilson';
    SELECT id INTO alice_id FROM users WHERE username = 'alice_brown';
    
    -- สร้างโพสต์
    INSERT INTO posts (id, user_id, content, privacy_level) VALUES
        (uuid_generate_v4(), john_id, 'สวัสดีครับ! วันนี้ไปเที่ยวที่ประตูน้ำ มาถ่ายรูปกับเพื่อนๆ 📸', 'public')
        RETURNING id INTO post1_id;
    
    INSERT INTO posts (id, user_id, content, privacy_level) VALUES
        (uuid_generate_v4(), jane_id, 'ผลงานออกแบบใหม่ล่าสุด ใครชอบบ้างคะ? 🎨', 'public')
        RETURNING id INTO post2_id;
    
    INSERT INTO posts (id, user_id, content, privacy_level) VALUES
        (uuid_generate_v4(), bob_id, 'Coffee break time! ใครอยากมาดื่มกาแฟด้วยกันบ้าง ☕', 'friends')
        RETURNING id INTO post3_id;
    
    -- เพิ่ม likes
    INSERT INTO post_likes (post_id, user_id) VALUES
        (post1_id, jane_id),
        (post1_id, bob_id),
        (post2_id, john_id),
        (post2_id, alice_id),
        (post3_id, alice_id);
    
    -- เพิ่ม comments
    INSERT INTO post_comments (post_id, user_id, content) VALUES
        (post1_id, jane_id, 'รูปสวยมากเลย! ไปกับใครบ้างคะ?'),
        (post1_id, bob_id, 'เก่งมาก ถ่ายรูปได้สวย'),
        (post2_id, john_id, 'ชอบมากครับ สีสันสวยงาม'),
        (post3_id, alice_id, 'อยากไปด้วย! ร้านไหนดีคะ?');
        
    -- เพิ่ม face tags ตัวอย่าง (สมมติว่ามีการแท็กในโพสต์)
    INSERT INTO post_face_tags (post_id, tagged_user_id, tagger_user_id, image_url, face_bbox, confidence_score, status) VALUES
        (post1_id, jane_id, john_id, '/uploads/post1_image.jpg', '{"x": 100, "y": 150, "width": 120, "height": 150}', 0.95, 'approved'),
        (post1_id, bob_id, john_id, '/uploads/post1_image.jpg', '{"x": 250, "y": 140, "width": 115, "height": 145}', 0.92, 'approved');
END $$;

-- เพิ่ม conversations ตัวอย่าง
DO $$
DECLARE
    john_id UUID;
    jane_id UUID;
    bob_id UUID;
    alice_id UUID;
    conv1_id UUID;
    conv2_id UUID;
    conv3_id UUID;
BEGIN
    -- ดึง user IDs
    SELECT id INTO john_id FROM users WHERE username = 'john_doe';
    SELECT id INTO jane_id FROM users WHERE username = 'jane_smith';
    SELECT id INTO bob_id FROM users WHERE username = 'bob_wilson';
    SELECT id INTO alice_id FROM users WHERE username = 'alice_brown';
    
    -- สร้าง direct conversations
    INSERT INTO conversations (id, type, created_by) VALUES
        (uuid_generate_v4(), 'direct', john_id)
        RETURNING id INTO conv1_id;
        
    INSERT INTO conversations (id, type, created_by) VALUES
        (uuid_generate_v4(), 'direct', jane_id)
        RETURNING id INTO conv2_id;
    
    -- สร้าง group conversation
    INSERT INTO conversations (id, type, name, description, created_by) VALUES
        (uuid_generate_v4(), 'group', 'เพื่อนสนิท', 'กลุ่มแชทสำหรับคุยกันเรื่องทั่วไป', john_id)
        RETURNING id INTO conv3_id;
    
    -- เพิ่ม participants
    INSERT INTO conversation_participants (conversation_id, user_id, role) VALUES
        (conv1_id, john_id, 'member'),
        (conv1_id, jane_id, 'member'),
        (conv2_id, jane_id, 'member'),
        (conv2_id, bob_id, 'member'),
        (conv3_id, john_id, 'admin'),
        (conv3_id, jane_id, 'member'),
        (conv3_id, bob_id, 'member'),
        (conv3_id, alice_id, 'member');
    
    -- เพิ่มข้อความตัวอย่าง
    INSERT INTO messages (conversation_id, sender_id, content, message_type) VALUES
        (conv1_id, john_id, 'สวัสดีครับ Jane!', 'text'),
        (conv1_id, jane_id, 'สวัสดีค่ะ John! เป็นไงบ้างคะ?', 'text'),
        (conv1_id, john_id, 'ดีครับ วันนี้ไปถ่ายรูปมา', 'text'),
        (conv2_id, jane_id, 'Bob ไปดื่มกาแฟกันไหมคะ?', 'text'),
        (conv2_id, bob_id, 'ได้เลย! ร้านไหนดีครับ?', 'text'),
        (conv3_id, john_id, 'สวัสดีทุกคน! วันนี้เป็นไงบ้าง?', 'text'),
        (conv3_id, alice_id, 'สวัสดีค่ะ! วันนี้ทำงานเยอะมาก', 'text'),
        (conv3_id, bob_id, 'ผมก็เหมือนกัน ไปพักผ่อนกันไหม?', 'text');
END $$;

-- เพิ่ม notifications ตัวอย่าง
DO $$
DECLARE
    john_id UUID;
    jane_id UUID;
    bob_id UUID;
    alice_id UUID;
BEGIN
    SELECT id INTO john_id FROM users WHERE username = 'john_doe';
    SELECT id INTO jane_id FROM users WHERE username = 'jane_smith';
    SELECT id INTO bob_id FROM users WHERE username = 'bob_wilson';
    SELECT id INTO alice_id FROM users WHERE username = 'alice_brown';
    
    INSERT INTO notifications (user_id, from_user_id, type, title, content) VALUES
        (john_id, jane_id, 'like', 'มีคนไลค์โพสต์ของคุณ', 'Jane Smith ได้ไลค์โพสต์ของคุณ'),
        (john_id, bob_id, 'comment', 'มีคนคอมเมนต์โพสต์ของคุณ', 'Bob Wilson ได้คอมเมนต์ในโพสต์ของคุณ'),
        (jane_id, john_id, 'face_tag', 'คุณถูกแท็กในรูปภาพ', 'John Doe ได้แท็กคุณในรูปภาพ'),
        (bob_id, jane_id, 'message', 'ข้อความใหม่', 'Jane Smith ส่งข้อความถึงคุณ'),
        (alice_id, bob_id, 'friend_request', 'คำขอเป็นเพื่อน', 'Bob Wilson ส่งคำขอเป็นเพื่อนถึงคุณ');
END $$;

-- เพิ่ม user activities log
DO $$
DECLARE
    john_id UUID;
    jane_id UUID;
    bob_id UUID;
    alice_id UUID;
BEGIN
    SELECT id INTO john_id FROM users WHERE username = 'john_doe';
    SELECT id INTO jane_id FROM users WHERE username = 'jane_smith';
    SELECT id INTO bob_id FROM users WHERE username = 'bob_wilson';
    SELECT id INTO alice_id FROM users WHERE username = 'alice_brown';
    
    INSERT INTO user_activities (user_id, activity_type, target_type, metadata) VALUES
        (john_id, 'post_created', 'post', '{"content": "สวัสดีครับ! วันนี้ไปเที่ยวที่ประตูน้ำ"}'),
        (jane_id, 'post_liked', 'post', '{"post_owner": "john_doe"}'),
        (jane_id, 'post_commented', 'post', '{"post_owner": "john_doe", "comment": "รูปสวยมากเลย!"}'),
        (bob_id, 'face_tagged', 'post', '{"tagger": "john_doe", "confidence": 0.92}'),
        (alice_id, 'message_sent', 'conversation', '{"conversation_type": "group", "message_type": "text"}');
END $$;
