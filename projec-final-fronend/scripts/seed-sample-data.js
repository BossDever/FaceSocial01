const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcryptjs');

const prisma = new PrismaClient();

async function createSampleData() {
  try {
    console.log('🌱 สร้างข้อมูลตัวอย่าง...');

    // สร้างผู้ใช้ตัวอย่าง
    const users = [];
    
    // ผู้ใช้หลัก (admin01 มีอยู่แล้ว)
    const existingAdmin = await prisma.user.findUnique({
      where: { username: 'admin01' }
    });

    if (existingAdmin) {
      users.push(existingAdmin);
      console.log('✅ ใช้ admin01 ที่มีอยู่แล้ว');
    }

    // สร้างผู้ใช้ใหม่
    const sampleUsers = [
      {
        username: 'john_doe',
        email: 'john@example.com',
        firstName: 'John',
        lastName: 'Doe',
        passwordHash: await bcrypt.hash('password123', 10)
      },
      {
        username: 'jane_smith',
        email: 'jane@example.com',
        firstName: 'Jane',
        lastName: 'Smith',
        passwordHash: await bcrypt.hash('password123', 10)
      },
      {
        username: 'bob_wilson',
        email: 'bob@example.com',
        firstName: 'Bob',
        lastName: 'Wilson',
        passwordHash: await bcrypt.hash('password123', 10)
      },
      {
        username: 'alice_brown',
        email: 'alice@example.com',
        firstName: 'Alice',
        lastName: 'Brown',
        passwordHash: await bcrypt.hash('password123', 10)
      }
    ];

    for (const userData of sampleUsers) {
      const existingUser = await prisma.user.findUnique({
        where: { username: userData.username }
      });

      if (!existingUser) {
        const user = await prisma.user.create({
          data: userData
        });
        users.push(user);
        console.log(`✅ สร้างผู้ใช้: ${user.username}`);
      } else {
        users.push(existingUser);
        console.log(`ℹ️ ผู้ใช้ ${userData.username} มีอยู่แล้ว`);
      }
    }

    // สร้างการเป็นเพื่อน
    console.log('👥 สร้างความสัมพันธ์เพื่อน...');
    
    const friendships = [
      { userId: users[0].id, friendId: users[1].id, status: 'ACCEPTED' },
      { userId: users[0].id, friendId: users[2].id, status: 'ACCEPTED' },
      { userId: users[1].id, friendId: users[2].id, status: 'PENDING' },
      { userId: users[2].id, friendId: users[3].id, status: 'ACCEPTED' }
    ];

    for (const friendship of friendships) {
      const existing = await prisma.friendship.findFirst({
        where: {
          OR: [
            { userId: friendship.userId, friendId: friendship.friendId },
            { userId: friendship.friendId, friendId: friendship.userId }
          ]
        }
      });

      if (!existing) {
        await prisma.friendship.create({
          data: friendship
        });
        console.log(`✅ สร้างมิตรภาพ: ${friendship.userId} <-> ${friendship.friendId}`);
      }
    }

    // สร้างโพสต์ตัวอย่าง
    console.log('📝 สร้างโพสต์ตัวอย่าง...');
    
    const samplePosts = [
      {
        userId: users[0].id,
        content: 'สวัสดีทุกคน! นี่คือโพสต์แรกของฉันใน FaceSocial 🎉',
        isPublic: true
      },
      {
        userId: users[1].id,
        content: 'วันนี้อากาศดีมาก เหมาะกับการออกไปเดินเล่น ☀️',
        isPublic: true
      },
      {
        userId: users[2].id,
        content: 'เพิ่งลองร้านอาหารใหม่ อร่อยมาก! แนะนำเลยครับ 🍜',
        location: 'กรุงเทพฯ',
        isPublic: true
      },
      {
        userId: users[3].id,
        content: 'กำลังเรียนโปรแกรมมิ่ง ใครมีคำแนะนำบ้างไหมคะ? 💻',
        isPublic: true
      }
    ];

    const createdPosts = [];
    for (const postData of samplePosts) {
      const post = await prisma.post.create({
        data: postData
      });
      createdPosts.push(post);
      console.log(`✅ สร้างโพสต์: ${post.content.substring(0, 30)}...`);
    }

    // สร้างไลค์และคอมเมนต์
    console.log('❤️ สร้างไลค์และคอมเมนต์...');
    
    // ไลค์
    const likes = [
      { userId: users[1].id, postId: createdPosts[0].id },
      { userId: users[2].id, postId: createdPosts[0].id },
      { userId: users[0].id, postId: createdPosts[1].id },
      { userId: users[3].id, postId: createdPosts[2].id }
    ];

    for (const like of likes) {
      const existing = await prisma.like.findUnique({
        where: {
          userId_postId: {
            userId: like.userId,
            postId: like.postId
          }
        }
      });

      if (!existing) {
        await prisma.like.create({
          data: like
        });
        console.log(`✅ สร้างไลค์`);
      }
    }

    // คอมเมนต์
    const comments = [
      {
        userId: users[1].id,
        postId: createdPosts[0].id,
        content: 'ยินดีต้อนรับครับ! 🎊'
      },
      {
        userId: users[2].id,
        postId: createdPosts[1].id,
        content: 'จริงๆ อากาศดีมากเลย!'
      },
      {
        userId: users[0].id,
        postId: createdPosts[2].id,
        content: 'ร้านไหนครับ? อยากลองเหมือนกัน'
      }
    ];

    for (const comment of comments) {
      await prisma.comment.create({
        data: comment
      });
      console.log(`✅ สร้างคอมเมนต์`);
    }

    // สร้างการแจ้งเตือน
    console.log('🔔 สร้างการแจ้งเตือน...');
    
    const notifications = [
      {
        userId: users[0].id,
        senderId: users[1].id,
        type: 'POST_LIKE',
        title: 'มีคนไลค์โพสต์ของคุณ',
        message: 'ไลค์โพสต์ของคุณ',
        data: { postId: createdPosts[0].id }
      },
      {
        userId: users[1].id,
        senderId: users[2].id,
        type: 'FRIEND_REQUEST',
        title: 'คำขอเป็นเพื่อน',
        message: 'ส่งคำขอเป็นเพื่อนถึงคุณ',
        data: { friendshipId: 'sample-id' }
      }
    ];

    for (const notification of notifications) {
      await prisma.notification.create({
        data: notification
      });
      console.log(`✅ สร้างการแจ้งเตือน: ${notification.title}`);
    }

    console.log('🎉 สร้างข้อมูลตัวอย่างสำเร็จ!');
    console.log(`📊 สรุป:`);
    console.log(`   - ผู้ใช้: ${users.length} คน`);
    console.log(`   - โพสต์: ${createdPosts.length} โพสต์`);
    console.log(`   - การแจ้งเตือน: ${notifications.length} รายการ`);

  } catch (error) {
    console.error('❌ เกิดข้อผิดพลาด:', error);
  } finally {
    await prisma.$disconnect();
  }
}

createSampleData();
