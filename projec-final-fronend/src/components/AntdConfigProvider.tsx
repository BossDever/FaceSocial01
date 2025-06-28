'use client';

import React from 'react';
import { ConfigProvider, App } from 'antd';
import thTH from 'antd/locale/th_TH';
import { colors } from '../styles/theme';

interface AntdConfigProviderProps {
  children: React.ReactNode;
}

const AntdConfigProvider: React.FC<AntdConfigProviderProps> = ({ children }) => {
  return (
    <ConfigProvider
      locale={thTH}
      theme={{
        token: {
          // สีหลัก
          colorPrimary: colors.light.primary,
          colorSuccess: colors.light.success,
          colorWarning: colors.light.warning,
          colorError: colors.light.error,
          colorInfo: colors.light.info,
          
          // สีพื้นหลัง
          colorBgContainer: colors.light.bgContainer,
          colorBgElevated: colors.light.bgContainer,
          colorBgLayout: colors.light.bgSecondary,
          
          // สีข้อความ
          colorText: colors.light.textPrimary,
          colorTextSecondary: colors.light.textSecondary,
          colorTextTertiary: colors.light.textTertiary,
          colorTextQuaternary: colors.light.textTertiary,
          
          // สีขอบ
          colorBorder: colors.light.borderPrimary,
          colorBorderSecondary: colors.light.borderSecondary,
          
          // ค่าพื้นฐาน
          borderRadius: 8,
          borderRadiusLG: 12,
          borderRadiusSM: 6,
          fontSize: 14,
          fontSizeLG: 16,
          fontSizeSM: 12,
          lineHeight: 1.6,
          
          // เงา
          boxShadow: `0 2px 8px ${colors.light.shadowMedium}`,
          boxShadowSecondary: `0 1px 4px ${colors.light.shadowLight}`,
          
          // การเคลื่อนไหว
          motionDurationSlow: '0.3s',
          motionDurationMid: '0.2s',
          motionDurationFast: '0.1s',
        },
        components: {
          // Button
          Button: {
            borderRadius: 8,
            fontWeight: 600,
            primaryShadow: `0 2px 4px ${colors.light.shadowMedium}`,
            defaultBorderColor: colors.light.borderPrimary,
            defaultColor: colors.light.textPrimary,
          },
            // Card
          Card: {
            borderRadius: 12,
            boxShadow: `0 2px 8px ${colors.light.shadowMedium}`,
            headerBg: colors.light.bgContainer,
          },
          
          // Input
          Input: {
            borderRadius: 8,
            fontSize: 14,
            hoverBorderColor: colors.light.primary,
            activeBorderColor: colors.light.primary,
          },
          
          // Menu
          Menu: {
            borderRadius: 8,
            itemBorderRadius: 8,
            itemHeight: 48,
            fontSize: 14,
            itemColor: colors.light.textSecondary,
            itemHoverColor: colors.light.primary,
            itemSelectedColor: colors.light.primary,
            itemSelectedBg: `${colors.light.primary}15`,
            iconSize: 18,
          },
          
          // Layout
          Layout: {
            siderBg: colors.light.bgContainer,
            headerBg: colors.light.bgContainer,
            bodyBg: colors.light.bgSecondary,
            footerBg: colors.light.bgContainer,
          },
          
          // Modal
          Modal: {
            borderRadius: 12,
            headerBg: colors.light.bgContainer,
            contentBg: colors.light.bgContainer,
          },
          
          // Typography
          Typography: {
            fontSize: 14,
            lineHeight: 1.6,
            colorText: colors.light.textPrimary,
            colorTextSecondary: colors.light.textSecondary,
          },
            // Avatar
          Avatar: {
            borderRadius: 50,
            colorBgContainer: colors.light.primary,
            colorText: colors.light.textInverse,
            fontSize: 14,
          },
          
          // Badge
          Badge: {
            colorBgContainer: colors.light.primary,
            colorText: colors.light.textInverse,
            fontSize: 12,
          },
          
          // Dropdown
          Dropdown: {
            borderRadius: 8,
            boxShadow: `0 4px 20px ${colors.light.shadowStrong}`,
            padding: 8,
          },
          
          // Select
          Select: {
            borderRadius: 8,
            fontSize: 14,
            hoverBorderColor: colors.light.primary,
            activeBorderColor: colors.light.primary,
          },
          
          // Tooltip
          Tooltip: {
            borderRadius: 6,
            colorBgSpotlight: colors.light.textPrimary,
            colorTextLightSolid: colors.light.textInverse,
          },
          
          // Message
          Message: {
            borderRadius: 8,
            fontSize: 14,
          },
          
          // Spin
          Spin: {
            colorPrimary: colors.light.primary,
          },
          
          // Progress
          Progress: {
            colorSuccess: colors.light.success,
            colorError: colors.light.error,
          },
        },
      }}
      // ปิดการแจ้งเตือน compatibility
      warning={{
        strict: false,
      }}
    >
      <App>
        {children}
      </App>
    </ConfigProvider>
  );
};

export default AntdConfigProvider;
