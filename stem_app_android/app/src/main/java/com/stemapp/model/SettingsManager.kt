package com.stemapp.model

import android.content.Context
import android.content.SharedPreferences

/**
 * 全局设置管理器
 * 持久化存储用户的聚类数、滤波开关等设置
 */
class SettingsManager(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("stem_app_settings", Context.MODE_PRIVATE)

    /**
     * 获取当前设置
     */
    fun getSettings(): AppSettings {
        return AppSettings(
            kmeansK = prefs.getInt("kmeans_k", 2),
            enableFilter = prefs.getBoolean("enable_filter", true),
            filterSigma = prefs.getFloat("filter_sigma", 3.0f).toDouble(),
            enableScale = prefs.getBoolean("enable_scale", false),
            pixelsPerMicron = prefs.getFloat("pixels_per_micron", 1.0f).toDouble(),
            scaleBarLength = prefs.getFloat("scale_bar_length", 500.0f).toDouble(),
            scaleBarPixels = prefs.getFloat("scale_bar_pixels", 0.0f).toDouble()
        )
    }

    /**
     * 保存设置
     */
    fun saveSettings(settings: AppSettings) {
        prefs.edit().apply {
            putInt("kmeans_k", settings.kmeansK)
            putBoolean("enable_filter", settings.enableFilter)
            putFloat("filter_sigma", settings.filterSigma.toFloat())
            putBoolean("enable_scale", settings.enableScale)
            putFloat("pixels_per_micron", settings.pixelsPerMicron.toFloat())
            putFloat("scale_bar_length", settings.scaleBarLength.toFloat())
            putFloat("scale_bar_pixels", settings.scaleBarPixels.toFloat())
            apply()
        }
    }
}