// Modern Color Palette with High Contrast
export const colors = {
  // Light Theme Colors
  light: {
    primary: '#1890ff',      // Bright blue
    primaryHover: '#40a9ff',
    primaryActive: '#096dd9',
    success: '#52c41a',      // Bright green
    warning: '#faad14',      // Orange
    error: '#ff4d4f',        // Red
    info: '#13c2c2',         // Cyan
    
    // Background colors
    bgPrimary: '#ffffff',    // Pure white
    bgSecondary: '#f5f5f5',  // Light gray
    bgContainer: '#ffffff',   // Card backgrounds
    bgHover: '#f0f0f0',      // Hover states
    
    // Text colors (high contrast)
    textPrimary: '#262626',   // Almost black
    textSecondary: '#595959', // Medium gray
    textTertiary: '#8c8c8c',  // Light gray
    textInverse: '#ffffff',   // White text
      // Border colors
    border: '#d9d9d9',
    borderPrimary: '#d9d9d9',
    borderSecondary: '#f0f0f0',
    borderActive: '#1890ff',
    
    // Layout colors
    bgLayout: '#f5f5f5',
    
    // Shadow colors
    shadowLight: 'rgba(0, 0, 0, 0.04)',
    shadowMedium: 'rgba(0, 0, 0, 0.08)',
    shadowStrong: 'rgba(0, 0, 0, 0.16)',
  },
  
  // Brand colors
  brand: {
    facebook: '#1877f2',
    gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    gradientHover: 'linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)',
  }
};

// Enhanced Component Styles
export const cardStyle = {
  backgroundColor: colors.light.bgContainer,
  borderRadius: '12px',
  border: `1px solid ${colors.light.borderPrimary}`,
  boxShadow: `0 2px 8px ${colors.light.shadowMedium}`,
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: `0 4px 16px ${colors.light.shadowStrong}`,
    transform: 'translateY(-2px)',
  }
};

export const buttonStyle = {
  primary: {
    backgroundColor: colors.light.primary,
    borderColor: colors.light.primary,
    color: colors.light.textInverse,
    borderRadius: '8px',
    fontWeight: 600,
    height: '40px',
    fontSize: '14px',
    transition: 'all 0.3s ease',
    '&:hover': {
      backgroundColor: colors.light.primaryHover,
      borderColor: colors.light.primaryHover,
      transform: 'translateY(-1px)',
      boxShadow: `0 4px 12px ${colors.light.shadowMedium}`,
    }
  },
  secondary: {
    backgroundColor: colors.light.bgContainer,
    borderColor: colors.light.borderPrimary,
    color: colors.light.textPrimary,
    borderRadius: '8px',
    fontWeight: 500,
    height: '40px',
    fontSize: '14px',
    transition: 'all 0.3s ease',
    '&:hover': {
      borderColor: colors.light.primary,
      color: colors.light.primary,
    }
  },
  ghost: {
    backgroundColor: 'transparent',
    borderColor: colors.light.borderPrimary,
    color: colors.light.textSecondary,
    borderRadius: '8px',
    fontWeight: 500,
    height: '40px',
    '&:hover': {
      backgroundColor: colors.light.bgHover,
      borderColor: colors.light.borderActive,
      color: colors.light.primary,
    }
  }
};

export const inputStyle = {
  backgroundColor: colors.light.bgContainer,
  borderRadius: '8px',
  border: `1px solid ${colors.light.borderPrimary}`,
  height: '40px',
  fontSize: '14px',
  color: colors.light.textPrimary,
  transition: 'all 0.3s ease',
  '&:focus': {
    borderColor: colors.light.borderActive,
    boxShadow: `0 0 0 2px ${colors.light.primary}20`,
  },
  '&::placeholder': {
    color: colors.light.textTertiary,
  }
};

// Layout Styles
export const layoutStyle = {
  sider: {
    backgroundColor: colors.light.bgContainer,
    borderRight: `1px solid ${colors.light.borderSecondary}`,
    boxShadow: `2px 0 8px ${colors.light.shadowLight}`,
  },
  header: {
    backgroundColor: colors.light.bgContainer,
    borderBottom: `1px solid ${colors.light.borderSecondary}`,
    boxShadow: `0 2px 8px ${colors.light.shadowLight}`,
    padding: '0 24px',
  },
  content: {
    backgroundColor: colors.light.bgSecondary,
    minHeight: 'calc(100vh - 64px)',
    padding: '24px',
  }
};

// Menu Styles
export const menuStyle = {
  backgroundColor: 'transparent',
  border: 'none',
  fontSize: '14px',
  fontWeight: 500,
  
  // Menu item styles
  '.ant-menu-item': {
    color: colors.light.textSecondary,
    margin: '4px 0',
    borderRadius: '8px',
    height: '48px',
    lineHeight: '48px',
    transition: 'all 0.3s ease',
    
    '&:hover': {
      backgroundColor: colors.light.bgHover,
      color: colors.light.primary,
    },
    
    '&.ant-menu-item-selected': {
      backgroundColor: `${colors.light.primary}10`,
      color: colors.light.primary,
      fontWeight: 600,
    },
    
    '.ant-menu-item-icon': {
      fontSize: '18px',
      marginRight: '12px',
    }
  }
};

// Icon Styles (Fix for dark icons issue)
export const iconStyle = {
  primary: {
    color: colors.light.primary,
    fontSize: '18px',
  },
  secondary: {
    color: colors.light.textSecondary,
    fontSize: '16px',
  },
  large: {
    color: colors.light.textPrimary,
    fontSize: '24px',
  },
  success: {
    color: colors.light.success,
    fontSize: '16px',
  },
  warning: {
    color: colors.light.warning,
    fontSize: '16px',
  },
  error: {
    color: colors.light.error,
    fontSize: '16px',
  }
};

// Typography Styles
export const textStyle = {
  heading1: {
    color: colors.light.textPrimary,
    fontSize: '32px',
    fontWeight: 700,
    lineHeight: 1.2,
    marginBottom: '24px',
  },
  heading2: {
    color: colors.light.textPrimary,
    fontSize: '24px',
    fontWeight: 600,
    lineHeight: 1.3,
    marginBottom: '16px',
  },
  heading3: {
    color: colors.light.textPrimary,
    fontSize: '18px',
    fontWeight: 600,
    lineHeight: 1.4,
    marginBottom: '12px',
  },
  body: {
    color: colors.light.textPrimary,
    fontSize: '14px',
    lineHeight: 1.6,
  },
  caption: {
    color: colors.light.textSecondary,
    fontSize: '12px',
    lineHeight: 1.4,
  }
};

// Utility classes for consistent spacing
export const spacing = {
  xs: '4px',
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '32px',
  '2xl': '48px',
  '3xl': '64px'
};

// Animation utilities
export const animations = {
  fadeIn: 'fadeIn 0.3s ease-in-out',
  slideUp: 'slideUp 0.3s ease-out',
  slideDown: 'slideDown 0.3s ease-out',
  bounce: 'bounce 0.5s ease-in-out',
  scale: 'scale 0.2s ease-out',
};

// Responsive breakpoints
export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px'
};
