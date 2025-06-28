const { PrismaClient } = require('@prisma/client');

async function checkDatabase() {
  const prisma = new PrismaClient();
  
  try {
    // Check posts table structure
    const posts = await prisma.$queryRaw`
      SELECT * FROM posts LIMIT 1
    `;
    console.log('Posts table structure:');
    console.log(posts[0] || 'No posts found');
    
    // Check notifications table structure  
    const notifications = await prisma.$queryRaw`
      SELECT * FROM notifications LIMIT 1
    `;
    console.log('\nNotifications table structure:');
    console.log(notifications[0] || 'No notifications found');
    
    // Check if tables exist
    const tables = await prisma.$queryRaw`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
    `;
    console.log('\nExisting tables:');
    console.table(tables);
    
  } catch (error) {
    console.error('Error:', error);
  } finally {
    await prisma.$disconnect();
  }
}

checkDatabase();
