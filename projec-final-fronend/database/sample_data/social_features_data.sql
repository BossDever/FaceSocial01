-- Sample data for testing social features

-- Insert sample posts
INSERT INTO posts (user_id, content, image_urls, created_at) VALUES
((SELECT id FROM users LIMIT 1), 'Hello everyone! This is my first post on FaceSocial! 🎉', NULL, NOW() - INTERVAL '2 hours'),
((SELECT id FROM users LIMIT 1), 'Beautiful sunset today! 🌅', ARRAY['/uploads/sample-sunset.jpg'], NOW() - INTERVAL '1 hour'),
((SELECT id FROM users LIMIT 1), 'Working on some exciting new features for the app. Stay tuned! 💻✨', NULL, NOW() - INTERVAL '30 minutes');

-- Insert sample post likes
INSERT INTO post_likes (user_id, post_id) VALUES
((SELECT id FROM users LIMIT 1), (SELECT id FROM posts ORDER BY created_at DESC LIMIT 1 OFFSET 1));

-- Insert sample comments
INSERT INTO post_comments (user_id, post_id, content, created_at) VALUES
((SELECT id FROM users LIMIT 1), (SELECT id FROM posts ORDER BY created_at DESC LIMIT 1), 'Great work! Looking forward to the new features! 👍', NOW() - INTERVAL '15 minutes'),
((SELECT id FROM users LIMIT 1), (SELECT id FROM posts ORDER BY created_at DESC LIMIT 1 OFFSET 1), 'Amazing photo! Where was this taken? 📸', NOW() - INTERVAL '45 minutes');

-- Insert sample messages (if there are multiple users)
-- Note: This will only work if there are at least 2 users in the database
DO $$
DECLARE
    user1_id UUID;
    user2_id UUID;
BEGIN
    -- Get first two users
    SELECT id INTO user1_id FROM users ORDER BY created_at LIMIT 1;
    SELECT id INTO user2_id FROM users ORDER BY created_at LIMIT 1 OFFSET 1;
    
    -- Only insert if we have 2 different users
    IF user1_id IS NOT NULL AND user2_id IS NOT NULL AND user1_id != user2_id THEN
        INSERT INTO messages (sender_id, receiver_id, content, created_at) VALUES
        (user1_id, user2_id, 'Hey! How are you doing?', NOW() - INTERVAL '3 hours'),
        (user2_id, user1_id, 'Hi there! I''m doing great, thanks for asking! 😊', NOW() - INTERVAL '2 hours 30 minutes'),
        (user1_id, user2_id, 'That''s awesome! Are you trying out the new social features?', NOW() - INTERVAL '2 hours'),
        (user2_id, user1_id, 'Yes! The feed and chat features are really nice. Great work on the UI! 👏', NOW() - INTERVAL '1 hour 45 minutes'),
        (user1_id, user2_id, 'Thanks! Let me know if you find any bugs or have suggestions for improvements.', NOW() - INTERVAL '1 hour 30 minutes');
    END IF;
END $$;
