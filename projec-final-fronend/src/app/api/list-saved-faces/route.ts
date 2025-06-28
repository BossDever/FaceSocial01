import { NextRequest, NextResponse } from 'next/server';
import { readdir, stat } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';

export async function GET() {
  try {
    const rootDir = process.cwd();
    const facesDir = path.join(rootDir, 'registration-faces');
    
    if (!existsSync(facesDir)) {
      return NextResponse.json({
        success: true,
        message: 'No registration faces directory found',
        images: []
      });
    }

    // Get all subdirectories (user folders)
    const userDirs = await readdir(facesDir, { withFileTypes: true });
    const allImages: Array<{
      fileName: string;
      path: string;
      userFolder: string;
      size: string;
      modified: string;
    }> = [];

    for (const userDir of userDirs) {
      if (userDir.isDirectory()) {
        const userPath = path.join(facesDir, userDir.name);
        
        try {
          const files = await readdir(userPath, { withFileTypes: true });
          
          for (const file of files) {
            if (file.isFile() && (file.name.endsWith('.jpg') || file.name.endsWith('.jpeg') || file.name.endsWith('.png'))) {
              const filePath = path.join(userPath, file.name);
              const fileStats = await stat(filePath);
              const relativePath = path.relative(rootDir, filePath).replace(/\\/g, '/');
              
              allImages.push({
                fileName: file.name,
                path: `/${relativePath}`, // Make it web accessible
                userFolder: userDir.name,
                size: `${Math.round(fileStats.size / 1024)}KB`,
                modified: fileStats.mtime.toISOString()
              });
            }
          }
        } catch (error) {
          console.warn(`Could not read user directory ${userDir.name}:`, error);
        }
      }
    }

    // Sort by modification time (newest first)
    allImages.sort((a, b) => new Date(b.modified).getTime() - new Date(a.modified).getTime());

    return NextResponse.json({
      success: true,
      message: `Found ${allImages.length} saved face images`,
      images: allImages,
      totalUsers: userDirs.filter(d => d.isDirectory()).length
    });

  } catch (error) {
    console.error('Error listing saved faces:', error);
    return NextResponse.json(
      { 
        success: false, 
        message: 'Failed to list saved face images',
        error: error instanceof Error ? error.message : 'Unknown error',
        images: []
      },
      { status: 500 }
    );
  }
}
