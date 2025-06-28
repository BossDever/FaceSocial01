/** @type {import('next').NextConfig} */
const nextConfig = {
  // Move turbo config to turbopack (stable)
  turbopack: {
    resolveAlias: {
      underscore: 'lodash',
      mocha: { browser: 'mocha/browser-entry.js' },
    },
  },
  images: {
    domains: ['localhost', 'facesocial.com'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },  // Add static file serving for registration-faces and uploads
  async rewrites() {
    return [
      {
        source: '/registration-faces/:path*',
        destination: '/api/serve-static/:path*'
      },
      {
        source: '/uploads/:path*',
        destination: '/api/serve-uploads/:path*'
      }
    ];
  },
  env: {
    NEXTAUTH_URL: process.env.NEXTAUTH_URL,
    NEXTAUTH_SECRET: process.env.NEXTAUTH_SECRET,
  },
  webpack: (config, { isServer, dev }) => {
    // Suppress console warnings in development
    if (dev && !isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
      
      // Suppress specific warnings
      config.ignoreWarnings = [
        /antd v5 support React is 16 ~ 18/,
        /compatible/,
      ];
    }
    return config;
  },
};

module.exports = nextConfig;
