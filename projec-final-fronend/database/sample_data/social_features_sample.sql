-- Insert sample users for testing social features
INSERT INTO users (username, email, password_hash, first_name, last_name, profile_image_url) VALUES
('john_doe', 'john@example.com', '$2b$10$dummy_hash_1', 'John', 'Doe', '/images/avatars/john.jpg'),
('jane_smith', 'jane@example.com', '$2b$10$dummy_hash_2', 'Jane', 'Smith', '/images/avatars/jane.jpg'),
('alice_wong', 'alice@example.com', '$2b$10$dummy_hash_3', 'Alice', 'Wong', '/images/avatars/alice.jpg'),
('bob_chen', 'bob@example.com', '$2b$10$dummy_hash_4', 'Bob', 'Chen', '/images/avatars/bob.jpg'),
('sarah_kim', 'sarah@example.com', '$2b$10$dummy_hash_5', 'Sarah', 'Kim', '/images/avatars/sarah.jpg')
ON CONFLICT (username) DO NOTHING;

-- Get user IDs for inserting posts and messages
DO $$
DECLARE
    john_id uuid;
    jane_id uuid;
    alice_id uuid;
    bob_id uuid;
    sarah_id uuid;
BEGIN
    SELECT id INTO john_id FROM users WHERE username = 'john_doe';
    SELECT id INTO jane_id FROM users WHERE username = 'jane_smith';
    SELECT id INTO alice_id FROM users WHERE username = 'alice_wong';
    SELECT id INTO bob_id FROM users WHERE username = 'bob_chen';
    SELECT id INTO sarah_id FROM users WHERE username = 'sarah_kim';

    -- Insert sample posts
    INSERT INTO posts (user_id, content, created_at) VALUES
    (john_id, 'Hello everyone! Just joined FaceSocial 🎉', NOW() - INTERVAL '2 hours'),
    (jane_id, 'Beautiful sunset today! Anyone else enjoying this weather? ☀️', NOW() - INTERVAL '4 hours'),
    (alice_id, 'Working on a new project. Face recognition is fascinating! 🤖', NOW() - INTERVAL '6 hours'),
    (bob_id, 'Coffee break ☕ What''s everyone up to today?', NOW() - INTERVAL '8 hours'),
    (sarah_id, 'Just finished reading a great book about AI and machine learning 📚', NOW() - INTERVAL '10 hours'),
    (john_id, 'The weather is perfect for a weekend getaway! 🏖️', NOW() - INTERVAL '12 hours'),
    (jane_id, 'Trying out a new recipe today. Wish me luck! 👩‍🍳', NOW() - INTERVAL '14 hours'),
    (alice_id, 'Technology is advancing so fast these days. Exciting times! 🚀', NOW() - INTERVAL '16 hours')
    ON CONFLICT DO NOTHING;

    -- Insert sample post likes
    INSERT INTO post_likes (user_id, post_id) 
    SELECT 
        (ARRAY[john_id, jane_id, alice_id, bob_id, sarah_id])[floor(random() * 5 + 1)],
        p.id
    FROM posts p
    WHERE random() > 0.3
    ON CONFLICT DO NOTHING;

    -- Insert sample post comments
    INSERT INTO post_comments (user_id, post_id, content, created_at)
    SELECT 
        jane_id,
        p.id,
        'Great post! 👍',
        p.created_at + INTERVAL '30 minutes'
    FROM posts p
    WHERE p.user_id = john_id
    LIMIT 2
    ON CONFLICT DO NOTHING;

    INSERT INTO post_comments (user_id, post_id, content, created_at)
    SELECT 
        alice_id,
        p.id,
        'I totally agree with this! Thanks for sharing.',
        p.created_at + INTERVAL '1 hour'
    FROM posts p
    WHERE p.user_id = jane_id
    LIMIT 1
    ON CONFLICT DO NOTHING;

    INSERT INTO post_comments (user_id, post_id, content, created_at)
    SELECT 
        bob_id,
        p.id,
        'Very interesting perspective 🤔',
        p.created_at + INTERVAL '45 minutes'
    FROM posts p
    WHERE p.user_id = alice_id
    LIMIT 1
    ON CONFLICT DO NOTHING;    -- Insert sample conversations and messages for chat
    -- Create direct conversations
    WITH conv1 AS (
        INSERT INTO conversations (type, created_by) 
        VALUES ('direct', john_id) 
        RETURNING id as conversation_id
    ),
    conv2 AS (
        INSERT INTO conversations (type, created_by) 
        VALUES ('direct', alice_id) 
        RETURNING id as conversation_id
    ),
    conv3 AS (
        INSERT INTO conversations (type, created_by) 
        VALUES ('direct', sarah_id) 
        RETURNING id as conversation_id
    )
    
    -- Insert conversation participants
    INSERT INTO conversation_participants (conversation_id, user_id)
    SELECT conversation_id, john_id FROM conv1
    UNION ALL
    SELECT conversation_id, jane_id FROM conv1
    UNION ALL
    SELECT conversation_id, alice_id FROM conv2
    UNION ALL
    SELECT conversation_id, bob_id FROM conv2
    UNION ALL
    SELECT conversation_id, sarah_id FROM conv3
    UNION ALL
    SELECT conversation_id, john_id FROM conv3;

    -- Insert sample messages
    INSERT INTO messages (conversation_id, sender_id, content, created_at)
    SELECT 
        c.id,
        john_id,
        'Hey Jane! How are you doing?',
        NOW() - INTERVAL '3 hours'
    FROM conversations c
    JOIN conversation_participants cp1 ON c.id = cp1.conversation_id AND cp1.user_id = john_id
    JOIN conversation_participants cp2 ON c.id = cp2.conversation_id AND cp2.user_id = jane_id
    WHERE c.type = 'direct'
    LIMIT 1;

    INSERT INTO messages (conversation_id, sender_id, content, created_at)
    SELECT 
        c.id,
        jane_id,
        'Hi John! I''m great, thanks for asking. How about you?',
        NOW() - INTERVAL '2.5 hours'
    FROM conversations c
    JOIN conversation_participants cp1 ON c.id = cp1.conversation_id AND cp1.user_id = john_id
    JOIN conversation_participants cp2 ON c.id = cp2.conversation_id AND cp2.user_id = jane_id
    WHERE c.type = 'direct'
    LIMIT 1;

END $$;
