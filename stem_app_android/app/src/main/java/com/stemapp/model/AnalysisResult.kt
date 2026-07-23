package com.stemapp.model

/**
 * 维管束分析结果
 */
data class AnalysisResult(
    val totalBundles: Int,                // 有效维管束总数
    val meanArea: Double,                 // 平均面积 (像素)
    val meanAreaUm2: Double? = null,      // 平均面积 (微米²)，有比例尺时
    val stdDevArea: Double,               // 面积标准差
    val minArea: Double,                  // 最小面积
    val maxArea: Double,                  // 最大面积
    val filteredOutCount: Int,            // 被3σ滤波剔除的数量
    val rawCount: Int,                    // 滤波前的原始数量
    val clusters: List<ClusterInfo>,      // K-means 聚类结果
    val vesselBitmap: android.graphics.Bitmap? = null,  // 标注了编号的维管束图
    val clusterMask: android.graphics.Bitmap? = null,   // 按K-means类别着色的Mask
    val clusterOverlay: android.graphics.Bitmap? = null, // 按K-means类别着色的叠加图
    val inferenceTimeMs: Long = 0
)

/**
 * K-means 聚类结果
 */
data class ClusterInfo(
    val clusterId: Int,                   // 类别ID (0, 1, 2, ...)
    val count: Int,                       // 该类别数量
    val meanArea: Double,                 // 该类别平均面积 (像素)
    val meanAreaUm2: Double? = null,      // 该类别平均面积 (微米²)
    val minArea: Double,                  // 该类别最小面积
    val maxArea: Double,                  // 该类别最大面积
    val stdDev: Double                    // 该类别面积标准差
)

/**
 * 全局设置
 */
data class AppSettings(
    val kmeansK: Int = 3,                 // K-means 聚类数
    val enableFilter: Boolean = true,     // 是否启用 3σ 滤波
    val filterSigma: Double = 3.0,        // 滤波标准差倍数
    val enableScale: Boolean = false,     // 是否启用比例尺换算
    val pixelsPerMicron: Double = 1.0,    // 每微米对应像素数
    val scaleBarLength: Double = 500.0,   // 标尺实际长度 (微米)
    val scaleBarPixels: Double = 0.0      // 标尺在图片上的像素长度
)