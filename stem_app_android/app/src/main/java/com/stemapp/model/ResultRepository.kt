package com.stemapp.model

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * 结果持久化管理
 * 每次推理完成后保存结果到文件，导出时读取所有已保存结果
 */
class ResultRepository(private val context: Context) {

    private val resultsDir: File
        get() = File(context.filesDir, "stem_results").also { it.mkdirs() }

    /**
     * 保存一条结果到文件
     */
    fun saveResult(sampleId: String, result: AnalysisResult, settings: AppSettings) {
        val json = JSONObject().apply {
            put("sampleId", sampleId)
            put("timestamp", System.currentTimeMillis())
            put("totalBundles", result.totalBundles)
            put("meanArea", result.meanArea)
            put("meanAreaUm2", result.meanAreaUm2 ?: JSONObject.NULL)
            put("minArea", result.minArea)
            put("maxArea", result.maxArea)
            put("stdDevArea", result.stdDevArea)
            put("filteredOutCount", result.filteredOutCount)
            put("rawCount", result.rawCount)
            put("scaleBarPixels", settings.scaleBarPixels)
            put("scaleBarLength", settings.scaleBarLength)
            put("hasScale", settings.enableScale)

            val clustersArr = JSONArray()
            result.clusters.forEach { c ->
                clustersArr.put(JSONObject().apply {
                    put("clusterId", c.clusterId)
                    put("count", c.count)
                    put("meanArea", c.meanArea)
                    put("meanAreaUm2", c.meanAreaUm2 ?: JSONObject.NULL)
                    put("minArea", c.minArea)
                    put("maxArea", c.maxArea)
                    put("stdDev", c.stdDev)
                })
            }
            put("clusters", clustersArr)
        }

        val safeId = sampleId.replace(Regex("[^a-zA-Z0-9_\\-]"), "_")
        val fileName = "${safeId}_${System.currentTimeMillis()}.json"
        File(resultsDir, fileName).writeText(json.toString(2))
    }

    /**
     * 读取所有已保存的结果
     */
    fun loadAllResults(): List<StemResult> {
        val files = resultsDir.listFiles { f -> f.name.endsWith(".json") } ?: return emptyList()
        return files.sortedByDescending { it.lastModified() }.mapNotNull { file ->
            try {
                val json = JSONObject(file.readText())
                val clusters = mutableListOf<ClusterInfo>()
                val clustersArr = json.optJSONArray("clusters")
                if (clustersArr != null) {
                    for (i in 0 until clustersArr.length()) {
                        val c = clustersArr.getJSONObject(i)
                        clusters.add(ClusterInfo(
                            clusterId = c.getInt("clusterId"),
                            count = c.getInt("count"),
                            meanArea = c.getDouble("meanArea"),
                            meanAreaUm2 = if (c.isNull("meanAreaUm2")) null else c.optDouble("meanAreaUm2", -1.0).let { if (it < 0) null else it },
                            minArea = c.getDouble("minArea"),
                            maxArea = c.getDouble("maxArea"),
                            stdDev = c.getDouble("stdDev")
                        ))
                    }
                }
                StemResult(
                    sampleId = json.getString("sampleId"),
                    timestamp = json.getLong("timestamp"),
                    totalBundles = json.getInt("totalBundles"),
                    meanArea = json.getDouble("meanArea"),
                    meanAreaUm2 = if (json.isNull("meanAreaUm2")) null else json.optDouble("meanAreaUm2", -1.0).let { if (it < 0) null else it },
                    minArea = json.getDouble("minArea"),
                    maxArea = json.getDouble("maxArea"),
                    stdDevArea = json.getDouble("stdDevArea"),
                    filteredOutCount = json.getInt("filteredOutCount"),
                    rawCount = json.getInt("rawCount"),
                    clusters = clusters,
                    scaleBarPixels = json.getDouble("scaleBarPixels"),
                    scaleBarLength = json.getDouble("scaleBarLength"),
                    hasScale = json.getBoolean("hasScale")
                )
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }
    }

    /**
     * 生成所有结果的纯文本汇总
     */
    fun generateTextReport(): String {
        val results = loadAllResults()
        if (results.isEmpty()) return "暂无分析结果"

        val sb = StringBuilder()
        sb.appendLine("==========================================")
        sb.appendLine("     茎秆维管束分析结果汇总")
        sb.appendLine("==========================================")
        sb.appendLine()
        sb.appendLine("总样本数: ${results.size}")
        sb.appendLine()

        results.forEachIndexed { idx, r ->
            sb.appendLine("------------------------------------------")
            sb.appendLine("样本 ${idx + 1}: ${r.sampleId}")
            sb.appendLine("  时间: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.getDefault()).format(java.util.Date(r.timestamp))}")
            sb.appendLine("  维管束总数: ${r.totalBundles}")
            sb.appendLine("  平均面积: ${"%.1f".format(r.meanArea)} px" + (if (r.hasScale && r.meanAreaUm2 != null) " (${"%.1f".format(r.meanAreaUm2)} μm²)" else ""))
            if (r.hasScale) {
                sb.appendLine("  比例尺: ${"%.0f".format(r.scaleBarPixels)} px = ${"%.0f".format(r.scaleBarLength)} μm")
            }
            r.clusters.forEachIndexed { ci, cluster ->
                val meanStr = if (r.hasScale && cluster.meanAreaUm2 != null) "${"%.1f".format(cluster.meanAreaUm2)} μm²" else "${"%.1f".format(cluster.meanArea)} px"
                sb.appendLine("  类别 ${ci + 1}: ${cluster.count} 个, 平均 $meanStr")
            }
            sb.appendLine()
        }

        sb.appendLine("==========================================")
        return sb.toString()
    }

    /**
     * 生成所有结果的 JSON 汇总
     */
    fun generateJsonReport(): String {
        val results = loadAllResults()
        val arr = JSONArray()
        results.forEach { r ->
            arr.put(JSONObject().apply {
                put("sampleId", r.sampleId)
                put("timestamp", r.timestamp)
                put("totalBundles", r.totalBundles)
                put("meanArea", r.meanArea)
                put("meanAreaUm2", r.meanAreaUm2 ?: JSONObject.NULL)
                put("minArea", r.minArea)
                put("maxArea", r.maxArea)
                put("stdDevArea", r.stdDevArea)
                put("filteredOutCount", r.filteredOutCount)
                put("rawCount", r.rawCount)
                put("scaleBarPixels", r.scaleBarPixels)
                put("scaleBarLength", r.scaleBarLength)
                put("hasScale", r.hasScale)
                val clustersArr = JSONArray()
                r.clusters.forEach { c ->
                    clustersArr.put(JSONObject().apply {
                        put("clusterId", c.clusterId)
                        put("count", c.count)
                        put("meanArea", c.meanArea)
                        put("meanAreaUm2", c.meanAreaUm2 ?: JSONObject.NULL)
                        put("minArea", c.minArea)
                        put("maxArea", c.maxArea)
                        put("stdDev", c.stdDev)
                    })
                }
                put("clusters", clustersArr)
            })
        }
        return arr.toString(2)
    }

    companion object {
        /**
         * 导出文本报告到文件
         */
        fun exportTextToFile(context: Context, content: String): File? {
            return try {
                val dir = File(context.getExternalFilesDir(null), "stem_exports").also { it.mkdirs() }
                val file = File(dir, "stem_results_${java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.getDefault()).format(java.util.Date())}.txt")
                file.writeText(content)
                file
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }

        /**
         * 导出 JSON 报告到文件
         */
        fun exportJsonToFile(context: Context, content: String): File? {
            return try {
                val dir = File(context.getExternalFilesDir(null), "stem_exports").also { it.mkdirs() }
                val file = File(dir, "stem_results_${java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.getDefault()).format(java.util.Date())}.json")
                file.writeText(content)
                file
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }
    }
}
