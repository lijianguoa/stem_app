package com.stemapp.model

/**
 * 持久化的分析结果
 * 每次推理完成后保存，用于导出汇总
 */
data class StemResult(
    val sampleId: String,
    val timestamp: Long,
    val totalBundles: Int,
    val meanArea: Double,
    val meanAreaUm2: Double?,
    val minArea: Double,
    val maxArea: Double,
    val stdDevArea: Double,
    val filteredOutCount: Int,
    val rawCount: Int,
    val clusters: List<ClusterInfo>,
    val scaleBarPixels: Double,
    val scaleBarLength: Double,
    val hasScale: Boolean
)
