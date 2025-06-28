'use client';

import { useAIServiceStatus } from '@/hooks/useAIServiceStatus';
import AIStatusIndicator from '@/components/ai/AIStatusIndicator';

export default function Home() {
  // AI Services Status Hooks - ใช้ endpoint ที่ถูกต้องตามแต่ละ service
  const faceDetection = useAIServiceStatus('/api/face-detection/health', 15000);
  const faceRecognition = useAIServiceStatus('/api/face-recognition/health', 15000);
  const faceAnalysis = useAIServiceStatus('/api/face-analysis/health', 15000);
  const antiSpoofing = useAIServiceStatus('/api/anti-spoofing/health', 15000);
  // Age-Gender ใช้ health endpoint แต่ไม่มี /api prefix
  const ageGender = useAIServiceStatus('/api/age-gender/health', 30000);

  // Smooth scroll function
  const smoothScrollTo = (elementId: string) => {
    const element = document.getElementById(elementId);
    if (element) {
      element.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold text-blue-600">🤖 FaceSocial</h1>
              </div>
            </div>
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                <button 
                  onClick={() => smoothScrollTo('features')} 
                  className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium cursor-pointer"
                >
                  ฟีเจอร์
                </button>
                <button 
                  onClick={() => smoothScrollTo('api')} 
                  className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium cursor-pointer"
                >
                  API
                </button>
                <a href="/login" className="text-blue-600 hover:text-blue-800 px-3 py-2 rounded-md text-sm font-medium">เข้าสู่ระบบ</a>
                <a href="/register" className="bg-blue-600 text-white hover:bg-blue-700 px-4 py-2 rounded-md text-sm font-medium">สมัครสมาชิก</a>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-20 pb-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              🤖 <span className="text-blue-600">FaceSocial</span>
            </h1>
            <h2 className="text-xl md:text-2xl text-gray-600 mb-8">
              AI-Powered Social Network
            </h2>
            <p className="text-lg text-gray-700 max-w-3xl mx-auto mb-12">
              เชื่อมต่อ แบ่งปัน และปกป้องด้วยเทคโนโลยีการจดจำใบหน้าขั้นสูง
            </p>
            
            <div className="flex flex-col sm:flex-row justify-center gap-4 mb-16">
              <a href="/ai-testing" className="bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors text-center">
                🧪 ทดลองฟีเจอร์ AI
              </a>
              <button className="border border-blue-600 text-blue-600 px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-50 transition-colors">
                📚 เรียนรู้เพิ่มเติม
              </button>
            </div>

            {/* Demo Video Placeholder */}
            <div className="bg-gray-100 rounded-xl p-12 max-w-4xl mx-auto">
              <div className="text-gray-500 text-center">
                <div className="text-6xl mb-4">📹</div>
                <p className="text-xl">การสาธิตระบบจดจำใบหน้าแบบเรียลไทม์</p>
                <p className="text-sm mt-2">การสาธิตแบบโต้ตอบจะพร้อมใช้งานที่นี่</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Key Features Section */}
      <section id="api" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              🎯 ฟีเจอร์ AI อันทรงพลัง
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Face Recognition */}
            {/* Face Detection */}
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-8 rounded-xl text-center">
              <div className="text-4xl mb-4">🎯</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">ระบบตรวจจับใบหน้า</h3>
              <p className="text-gray-600 mb-4">YOLOv9 & YOLOv11</p>
              <p className="text-sm text-gray-500 mb-6">ตรวจจับใบหน้าหลายใบในรูปภาพและวิดีโอด้วยการระบุตำแหน่งที่แม่นยำ</p>
              <div className="flex justify-center">
                <AIStatusIndicator service={faceDetection.status} />
              </div>
            </div>

            {/* Anti-Spoofing */}
            {/* Anti-Spoofing */}
            <div className="bg-gradient-to-br from-green-50 to-green-100 p-8 rounded-xl text-center">
              <div className="text-4xl mb-4">🛡️</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">ระบบป้องกันการปลอมแปลง</h3>
              <p className="text-gray-600 mb-4">DeepFace Anti-Spoofing</p>
              <p className="text-sm text-gray-500 mb-6">ตรวจจับความพยายามปลอมแปลงและป้องกันด้วยการตรวจสอบความเป็นจริงขั้นสูง</p>
              <div className="flex justify-center">
                <AIStatusIndicator service={antiSpoofing.status} />
              </div>
            </div>

            {/* Deepfake Detection */}
            {/* Face Analysis */}
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-8 rounded-xl text-center">
              <div className="text-4xl mb-4">🔍</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">ระบบวิเคราะห์ใบหน้าแบบครบวงจร</h3>
              <p className="text-gray-600 mb-4">Multi-Model AI Analysis</p>
              <p className="text-sm text-gray-500 mb-6">วิเคราะห์ใบหน้าอย่างครอบคลุม รวมถึงการตรวจจับ จดจำ และ Deepfake</p>
              <div className="flex justify-center">
                <AIStatusIndicator service={faceAnalysis.status} />
              </div>
            </div>

            {/* Face Recognition */}
            <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-8 rounded-xl text-center">
              <div className="text-4xl mb-4">🧠</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">ระบบจดจำใบหน้าอัจฉริยะ</h3>
              <p className="text-gray-600 mb-4">FaceNet, ArcFace, AdaFace</p>
              <p className="text-sm text-gray-500 mb-6">ระบบจดจำใบหน้าขั้นสูงด้วย AI ที่มีความแม่นยำในระดับอุตสาหกรรม</p>
              <div className="flex justify-center">
                <AIStatusIndicator service={faceRecognition.status} />
              </div>
            </div>

            {/* Age & Gender Analysis */}
            <div className="bg-gradient-to-br from-pink-50 to-pink-100 p-8 rounded-xl text-center">
              <div className="text-4xl mb-4">👨‍👩‍👧‍👦</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">ระบบวิเคราะห์อายุและเพศ</h3>
              <p className="text-gray-600 mb-2">DeepFace Backend (CPU)</p>
              <p className="text-sm text-gray-500 mb-4">วิเคราะห์ข้อมูลประชากรศาสตร์ด้วยความแม่นยำสูง</p>
              {ageGender.status.status === 'loading' && (
                <div className="mb-4 p-3 bg-yellow-100 border border-yellow-300 rounded-lg">
                  <p className="text-sm text-yellow-800">
                    🔄 กำลังเริ่มต้นระบบ AI (ใช้เวลา 5-10 วินาที)...
                  </p>
                </div>
              )}
              <div className="flex justify-center">
                <AIStatusIndicator service={ageGender.status} />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Platform Benefits Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              🚀 ทำไมต้องเลือก FaceSocial?
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* Privacy First */}
            <div className="bg-white p-8 rounded-xl shadow-sm">
              <div className="flex items-start">
                <div className="text-3xl mr-4">🔒</div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-4">ความเป็นส่วนตัวเป็นอันดับแรก</h3>
                  <ul className="space-y-2 text-gray-600">
                    <li>• การเข้ารหัสแบบ end-to-end สำหรับข้อมูลใบหน้า</li>
                    <li>• ผู้ใช้ควบคุมการตั้งค่าความเป็นส่วนตัวทั้งหมด</li>
                    <li>• การจัดการข้อมูลที่สอดคล้องกับ GDPR</li>
                    <li>• การเข้าร่วมระบบจดจำใบหน้าแบบสมัครใจ</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Advanced AI */}
            <div className="bg-white p-8 rounded-xl shadow-sm">
              <div className="flex items-start">
                <div className="text-3xl mr-4">🧠</div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-4">AI ขั้นสูง</h3>
                  <ul className="space-y-2 text-gray-600">
                    <li>• เครือข่ายประสาทเทียมล้ำสมัย</li>
                    <li>• ความสามารถในการประมวลผลแบบเรียลไทม์</li>
                    <li>• การปรับปรุงโมเดลอย่างต่อเนื่อง</li>
                    <li>• การผสานรวม AI แบบหลากหลายรูปแบบ</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Developer Friendly */}
            <div className="bg-white p-8 rounded-xl shadow-sm">
              <div className="flex items-start">
                <div className="text-3xl mr-4">👨‍💻</div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-4">เป็นมิตรกับนักพัฒนา</h3>
                  <ul className="space-y-2 text-gray-600">
                    <li>• RESTful API พร้อมเอกสารครบถ้วน</li>
                    <li>• SDK สำหรับแพลตฟอร์มยอดนิยม</li>
                    <li>• สภาพแวดล้อม Sandbox สำหรับทดสอบ</li>
                    <li>• การสนับสนุนทางเทคนิค 24/7</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Scalable Platform */}
            <div className="bg-white p-8 rounded-xl shadow-sm">
              <div className="flex items-start">
                <div className="text-3xl mr-4">📈</div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-4">แพลตฟอร์มที่ขยายได้</h3>
                  <ul className="space-y-2 text-gray-600">
                    <li>• สถาปัตยกรรม Cloud-native</li>
                    <li>• ความสามารถในการขยายอัตโนมัติ</li>
                    <li>• รับประกันการใช้งาน 99.9%</li>
                    <li>• การกระจาย CDN ทั่วโลก</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Additional Features Section */}
      <section id="features" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              ✨ คุณสมบัติเพิ่มเติม
            </h2>
            <p className="text-xl text-gray-600">
              ฟีเจอร์ที่ช่วยให้คุณพัฒนาแอปพลิเคชันได้อย่างมีประสิทธิภาพ
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Real-time Processing */}
            <div className="bg-white p-8 rounded-xl shadow-sm text-center">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">ประมวลผลแบบเรียลไทม์</h3>
              <p className="text-gray-600">ความเร็วในการประมวลผลต่ำกว่า 100ms สำหรับการตรวจจับใบหน้า</p>
            </div>

            {/* Multi-Platform */}
            <div className="bg-white p-8 rounded-xl shadow-sm text-center">
              <div className="text-4xl mb-4">📱</div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">รองรับหลายแพลตฟอร์ม</h3>
              <p className="text-gray-600">ทำงานได้บนเว็บ, มือถือ, และเดสก์ท็อป ด้วย API เดียว</p>
            </div>

            {/* High Accuracy */}
            <div className="bg-white p-8 rounded-xl shadow-sm text-center">
              <div className="text-4xl mb-4">🎯</div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">ความแม่นยำสูง</h3>
              <p className="text-gray-600">ความแม่นยำในการจดจำใบหน้าสูงถึง 99.8% ในสภาพแวดล้อมที่ควบคุมได้</p>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-8">พร้อมที่จะเริ่มต้นแล้วหรือยัง?</h2>
          <div className="space-x-4">
            <a href="/ai-testing" className="bg-blue-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors">
              ทดลองใช้ฟรี
            </a>
            <a href="/register" className="border border-blue-600 text-blue-600 px-8 py-3 rounded-lg font-medium hover:bg-blue-50 transition-colors">
              สมัครสมาชิก
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h3 className="text-2xl font-bold mb-4">🤖 FaceSocial</h3>
            <p className="text-gray-400 mb-8">แพลตฟอร์มโซเชียลเน็ตเวิร์กขับเคลื่อนด้วย AI</p>
            
            <div className="flex justify-center space-x-6 mb-8">
              <a href="#" className="text-gray-400 hover:text-white">นโยบายความเป็นส่วนตัว</a>
              <a href="#" className="text-gray-400 hover:text-white">ข้อกำหนดการให้บริการ</a>
              <a href="#" className="text-gray-400 hover:text-white">ติดต่อเรา</a>
              <a href="#" className="text-gray-400 hover:text-white">เอกสาร API</a>
            </div>
            
            <p className="text-gray-500 text-sm">
              © 2025 FaceSocial. สงวนลิขสิทธิ์ทุกประการ
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
