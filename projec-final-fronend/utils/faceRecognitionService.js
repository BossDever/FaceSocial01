/**
 * Face Recognition Integration Module
 * Uses the verified working endpoint: /api/face-analysis/analyze
 */

export class FaceRecognitionService {
    constructor(apiBaseUrl = 'http://localhost:8080') {
        this.apiBaseUrl = apiBaseUrl;
    }

    /**
     * Recognize face using the working endpoint
     * @param {File|Buffer|string} imageData - Image file, buffer, or base64 string
     * @param {Object} options - Recognition options
     * @returns {Promise<Object>} Recognition results
     */
    async recognizeFace(imageData, options = {}) {
        const {
            mode = 'full_analysis',
            confidence_threshold = 0.5,
            similarity_threshold = 0.6,
            max_faces = 1
        } = options;

        try {
            const formData = new FormData();
            
            // Handle different input types
            if (imageData instanceof File) {
                formData.append('file', imageData);
            } else if (Buffer.isBuffer(imageData)) {
                const blob = new Blob([imageData], { type: 'image/jpeg' });
                formData.append('file', blob, 'face.jpg');
            } else if (typeof imageData === 'string') {
                // Convert base64 to blob
                const byteCharacters = atob(imageData);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], { type: 'image/jpeg' });
                formData.append('file', blob, 'face.jpg');
            }

            // Add options
            formData.append('mode', mode);
            formData.append('confidence_threshold', confidence_threshold.toString());
            formData.append('similarity_threshold', similarity_threshold.toString());
            formData.append('max_faces', max_faces.toString());

            const response = await fetch(`${this.apiBaseUrl}/api/face-analysis/analyze`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            return this.processRecognitionResult(result);

        } catch (error) {
            console.error('Face recognition error:', error);
            throw new Error(`Face recognition failed: ${error.message}`);
        }
    }

    /**
     * Process and normalize recognition results
     * @param {Object} apiResult - Raw API response
     * @returns {Object} Processed results
     */
    processRecognitionResult(apiResult) {
        if (!apiResult.success) {
            return {
                success: false,
                error: apiResult.error || 'Recognition failed',
                faces: []
            };
        }

        if (!apiResult.faces || apiResult.faces.length === 0) {
            return {
                success: true,
                message: 'No faces detected in image',
                faces: [],
                faceCount: 0
            };
        }

        const processedFaces = apiResult.faces.map(face => ({
            // Detection info
            bbox: face.bbox,
            detectionConfidence: face.detection_confidence,
            qualityScore: face.quality_score,
            
            // Recognition info
            hasIdentity: face.has_identity,
            identity: face.identity,
            identityName: face.identity_name,
            recognitionConfidence: face.recognition_confidence,
            
            // Match info
            matches: face.matches || [],
            bestMatch: face.best_match,
            
            // Processing info
            processingTime: face.processing_time,
            modelUsed: face.model_used,
            recognitionModel: face.recognition_model
        }));

        return {
            success: true,
            faces: processedFaces,
            faceCount: processedFaces.length,
            performance: apiResult.performance,
            modelsUsed: apiResult.models_used,
            statistics: apiResult.statistics,
            metadata: apiResult.analysis_metadata
        };
    }

    /**
     * Check if a face matches a specific identity with sufficient confidence
     * @param {Object} recognitionResult - Result from recognizeFace()
     * @param {string} expectedIdentity - Expected identity to match
     * @param {number} minConfidence - Minimum confidence threshold (0-1)
     * @returns {Object} Match result
     */
    checkIdentityMatch(recognitionResult, expectedIdentity, minConfidence = 0.6) {
        if (!recognitionResult.success || recognitionResult.faceCount === 0) {
            return {
                isMatch: false,
                reason: 'No faces detected',
                confidence: 0
            };
        }

        const face = recognitionResult.faces[0]; // Use first/best face
        
        if (!face.hasIdentity) {
            return {
                isMatch: false,
                reason: 'No identity found for detected face',
                confidence: 0
            };
        }

        if (face.identity !== expectedIdentity) {
            return {
                isMatch: false,
                reason: `Identity mismatch: expected '${expectedIdentity}', got '${face.identity}'`,
                confidence: face.recognitionConfidence,
                actualIdentity: face.identity
            };
        }

        if (face.recognitionConfidence < minConfidence) {
            return {
                isMatch: false,
                reason: `Confidence too low: ${(face.recognitionConfidence * 100).toFixed(1)}% < ${(minConfidence * 100).toFixed(1)}%`,
                confidence: face.recognitionConfidence
            };
        }

        return {
            isMatch: true,
            reason: 'Identity match confirmed',
            confidence: face.recognitionConfidence,
            identity: face.identity,
            identityName: face.identityName,
            bestMatch: face.bestMatch
        };
    }

    /**
     * Get gallery status
     * @returns {Promise<Object>} Gallery information
     */
    async getGalleryStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/face-recognition/gallery/get`);
            
            if (!response.ok) {
                throw new Error(`Gallery request failed: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.gallery) {
                const personIds = Object.keys(result.gallery);
                return {
                    success: true,
                    totalPersons: personIds.length,
                    persons: personIds,
                    gallery: result.gallery
                };
            }

            return {
                success: false,
                error: 'Invalid gallery response format'
            };

        } catch (error) {
            console.error('Gallery status error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }
}

// Node.js compatible version for backend use
export class NodeFaceRecognitionService extends FaceRecognitionService {
    constructor(apiBaseUrl = 'http://localhost:8080') {
        super(apiBaseUrl);
        // Import fetch for Node.js if needed
        if (typeof fetch === 'undefined') {
            this.fetch = require('node-fetch');
        } else {
            this.fetch = fetch;
        }
    }

    /**
     * Recognize face from buffer (Node.js backend usage)
     * @param {Buffer} imageBuffer - Image buffer
     * @param {Object} options - Recognition options
     * @returns {Promise<Object>} Recognition results
     */
    async recognizeFaceFromBuffer(imageBuffer, options = {}) {
        const FormData = require('form-data');
        const formData = new FormData();
        
        formData.append('file', imageBuffer, {
            filename: 'face.jpg',
            contentType: 'image/jpeg'
        });
        
        formData.append('mode', options.mode || 'full_analysis');
        formData.append('confidence_threshold', (options.confidence_threshold || 0.5).toString());
        formData.append('similarity_threshold', (options.similarity_threshold || 0.6).toString());
        formData.append('max_faces', (options.max_faces || 1).toString());

        try {
            const response = await this.fetch(`${this.apiBaseUrl}/api/face-analysis/analyze`, {
                method: 'POST',
                body: formData,
                headers: formData.getHeaders()
            });

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            return this.processRecognitionResult(result);

        } catch (error) {
            console.error('Face recognition error:', error);
            throw new Error(`Face recognition failed: ${error.message}`);
        }
    }
}

// Example usage functions
export const FaceRecognitionExamples = {
    /**
     * Example: Login with face image
     */
    async loginWithFace(imageFile, expectedUserId) {
        const faceService = new FaceRecognitionService();
        
        try {
            // 1. Recognize face
            const recognition = await faceService.recognizeFace(imageFile);
            
            // 2. Check if matches expected user
            const match = faceService.checkIdentityMatch(recognition, expectedUserId, 0.6);
            
            if (match.isMatch) {
                return {
                    success: true,
                    message: 'Login successful',
                    userId: match.identity,
                    confidence: match.confidence,
                    details: recognition
                };
            } else {
                return {
                    success: false,
                    message: match.reason,
                    confidence: match.confidence || 0,
                    details: recognition
                };
            }
        } catch (error) {
            return {
                success: false,
                message: error.message,
                error: true
            };
        }
    },

    /**
     * Example: Register new face
     */
    async registerFace(imageFile, userId) {
        // Note: This would require a registration endpoint
        // For now, this is a placeholder showing the expected flow
        console.log(`Register face for user: ${userId}`);
        return { success: false, message: 'Registration endpoint not implemented yet' };
    }
};

// Default export
export default FaceRecognitionService;
