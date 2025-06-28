import { PrismaClient } from '@prisma/client';
import { hash } from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  // Create test users
  const testUsers = [
    {
      username: 'testuser',
      email: 'test@example.com',
      firstName: 'John',
      lastName: 'Doe',
      phone: '0812345678',
    },
    {
      username: 'admin',
      email: 'admin@company.com',
      firstName: 'Admin',
      lastName: 'User',
      phone: '0987654321',
    },
    {
      username: 'user123',
      email: 'user@test.com',
      firstName: 'Test',
      lastName: 'User',
      phone: '0856789012',
    }
  ];
  for (const userData of testUsers) {
    const passwordHash = await hash('password123', 12);
    
    // Check if user already exists by email or username
    const existingUser = await prisma.user.findFirst({
      where: {
        OR: [
          { email: userData.email },
          { username: userData.username }
        ]
      }
    });

    let user;
    if (existingUser) {
      console.log(`User already exists: ${userData.email} (${userData.username})`);
      user = existingUser;
    } else {
      user = await prisma.user.create({
        data: {
          ...userData,
          passwordHash,
          isActive: true,
          isVerified: true,
        }      });
      console.log(`Created user: ${userData.email}`);
    }
    
    // Add dummy face embedding
    const existingEmbedding = await prisma.faceEmbedding.findFirst({
      where: { 
        userId: user.id,
        isPrimary: true 
      }
    });

    if (!existingEmbedding) {
      await prisma.faceEmbedding.create({
        data: {
          userId: user.id,
          embeddingModel: 'facenet',
          embeddingData: Array.from({ length: 512 }, () => Math.random()), // 512-dimension dummy embedding
          qualityScore: 0.85,
          detectionConfidence: 0.9,
          isPrimary: true,
        }      });
    }
  }
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
