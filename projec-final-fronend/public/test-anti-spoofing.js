/**
 * Browser Test for Anti-Spoofing API
 * Run this in browser console to test the exact same code as the component
 */

// Test function to simulate the exact process
async function testAntiSpoofingInBrowser() {
    console.log('🧪 Testing Anti-Spoofing API in Browser');
    
    try {
        // Get test image from a simple canvas (simulating cropped face)
        const canvas = document.createElement('canvas');
        canvas.width = 200;
        canvas.height = 200;
        const ctx = canvas.getContext('2d');
        
        // Draw a simple test pattern
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, 200, 200);
        ctx.fillStyle = '#666';
        ctx.fillRect(50, 50, 100, 100);
        ctx.fillStyle = '#333';
        ctx.fillRect(75, 75, 50, 50);
        
        // Convert to base64
        const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
        const base64Data = imageDataUrl.split(',')[1];
        
        console.log('Generated test image:', {
            base64Length: base64Data.length,
            dataUrlLength: imageDataUrl.length
        });
        
        // Convert to blob (same as component)
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/jpeg' });
        
        console.log('Blob created:', {
            size: blob.size,
            type: blob.type
        });
        
        // Create FormData (same as component)
        const formData = new FormData();
        formData.append('image', blob, 'test_face.jpg');
        formData.append('confidence_threshold', '0.3');
        
        console.log('FormData created, sending request...');
        
        // Send request
        const response = await fetch('http://localhost:8080/api/anti-spoofing/detect-upload', {
            method: 'POST',
            body: formData
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', [...response.headers.entries()]);
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Success:', result);
            return true;
        } else {
            const errorText = await response.text();
            console.error('❌ Error:', response.status, response.statusText);
            console.error('Error body:', errorText);
            
            try {
                const errorJson = JSON.parse(errorText);
                console.error('Error details:', errorJson);
            } catch (e) {
                console.log('Error is not JSON format');
            }
            return false;
        }
        
    } catch (error) {
        console.error('❌ Exception:', error);
        return false;
    }
}

// Test with real image from file input
async function testWithRealImage() {
    console.log('🧪 Testing with real image file');
    
    // Create file input for testing
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    
    return new Promise((resolve) => {
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) {
                console.log('No file selected');
                resolve(false);
                return;
            }
            
            console.log('Selected file:', {
                name: file.name,
                size: file.size,
                type: file.type
            });
            
            try {
                // Create FormData directly with file
                const formData = new FormData();
                formData.append('image', file);
                formData.append('confidence_threshold', '0.3');
                
                const response = await fetch('http://localhost:8080/api/anti-spoofing/detect-upload', {
                    method: 'POST',
                    body: formData
                });
                
                console.log('Response status:', response.status);
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('✅ Success with real image:', result);
                    resolve(true);
                } else {
                    const errorText = await response.text();
                    console.error('❌ Error with real image:', response.status);
                    console.error('Error body:', errorText);
                    resolve(false);
                }
                
            } catch (error) {
                console.error('❌ Exception with real image:', error);
                resolve(false);
            }
        };
        
        input.click();
    });
}

// Auto-run tests
console.log('🚀 Starting Anti-Spoofing Browser Tests');
console.log('Run testAntiSpoofingInBrowser() to test with generated image');
console.log('Run testWithRealImage() to test with file upload');

// Uncomment to auto-run
// testAntiSpoofingInBrowser();
