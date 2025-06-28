import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { imageData, fileName, userId } = body;

    if (!imageData || !fileName) {
      return NextResponse.json(
        { success: false, message: 'Missing image data or filename' },
        { status: 400 }
      );
    }

    // Create registration-faces directory in root if it doesn't exist
    const rootDir = process.cwd();
    const facesDir = path.join(rootDir, 'registration-faces');
    
    if (!existsSync(facesDir)) {
      await mkdir(facesDir, { recursive: true });
    }

    // Create user-specific directory
    const userDir = path.join(facesDir, userId || 'temp');
    if (!existsSync(userDir)) {
      await mkdir(userDir, { recursive: true });
    }

    // Convert base64 to buffer
    const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');
    const buffer = Buffer.from(base64Data, 'base64');

    // Save the file
    const filePath = path.join(userDir, fileName);
    await writeFile(filePath, buffer);

    const relativePath = path.relative(rootDir, filePath);

    console.log(`✅ Saved cropped face image: ${relativePath}`);

    return NextResponse.json({
      success: true,
      message: 'Image saved successfully',
      filePath: relativePath,
      fullPath: filePath
    });

  } catch (error) {
    console.error('❌ Error saving face image:', error);
    return NextResponse.json(
      { 
        success: false, 
        message: 'Failed to save image',
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
