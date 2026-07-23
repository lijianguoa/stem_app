package com.stemapp.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.stemapp.model.AnalysisResult
import com.stemapp.model.AppSettings
import com.stemapp.model.ClusterInfo
import java.io.File
import kotlin.math.sqrt

/**
 * ONNX Runtime 推理引擎封装
 * 通用设计：支持任意 [1, C, H, W] → [1, N, H, W] 格式的 ONNX 分割模型
 * 自动兼容 FP32 和 FP16 模型
 * 支持维管束连通域分析、3σ滤波、K-means聚类
 */
class OnnxInference {

    private var ortSession: ai.onnxruntime.OrtSession? = null
    private var ortEnv: ai.onnxruntime.OrtEnvironment? = null
    private var modelInfo: ModelInfo? = null
    private var lastClassMap: IntArray? = null  // 最后一次推理的类别图
    private var lastOutputHeight: Int = 0
    private var lastOutputWidth: Int = 0
    private var lastOriginalBitmap: Bitmap? = null

    data class ModelInfo(
        val inputName: String,
        val outputName: String,
        val inputShape: LongArray,      // [1, C, H, W]
        val outputShape: LongArray,     // [1, N, H, W]
        val inputChannels: Int,
        val inputHeight: Int,
        val inputWidth: Int,
        val numClasses: Int
    )

    data class SegmentationResult(
        val mask: Bitmap,
        val overlay: Bitmap,
        val original: Bitmap,
        val inferenceTimeMs: Long
    )

    // ==================== 模型加载 ====================

    fun loadModel(modelFile: File): Boolean {
        return try {
            ortEnv = ai.onnxruntime.OrtEnvironment.getEnvironment()
            ortSession = ortEnv?.createSession(modelFile.absolutePath)

            val session = ortSession ?: return false
            val inputInfo = session.inputInfo
            val outputInfo = session.outputInfo

            val inputName = inputInfo.keys.first()
            val outputName = outputInfo.keys.first()

            val inputNodeInfo = inputInfo[inputName]!!
            val outputNodeInfo = outputInfo[outputName]!!

            val inputTensorInfo = inputNodeInfo.info as ai.onnxruntime.TensorInfo
            val outputTensorInfo = outputNodeInfo.info as ai.onnxruntime.TensorInfo

            val inputShape = inputTensorInfo.shape
            val outputShape = outputTensorInfo.shape

            modelInfo = ModelInfo(
                inputName = inputName,
                outputName = outputName,
                inputShape = inputShape,
                outputShape = outputShape,
                inputChannels = inputShape[1].toInt(),
                inputHeight = inputShape[2].toInt(),
                inputWidth = inputShape[3].toInt(),
                numClasses = outputShape[1].toInt()
            )

            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    fun unloadModel() {
        ortSession?.close()
        ortSession = null
        modelInfo = null
    }

    fun isModelLoaded(): Boolean = ortSession != null && modelInfo != null
    fun getModelInfo(): ModelInfo? = modelInfo

    // ==================== 推理 ====================

    fun runInference(bitmap: Bitmap): SegmentationResult? {
        val session = ortSession ?: return null
        val info = modelInfo ?: return null

        val startTime = System.currentTimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap, info.inputWidth, info.inputHeight, true
        )
        val inputTensor = preprocess(resizedBitmap, info)

        val result = session.run(mapOf(info.inputName to inputTensor))

        val rawOutput = result[info.outputName]
        val outputTensor = if (rawOutput is java.util.Optional<*>) {
            rawOutput.get() as ai.onnxruntime.OnnxTensor
        } else {
            rawOutput as ai.onnxruntime.OnnxTensor
        }
        val outputArray = readOutputAsFloatArray(outputTensor, info)

        // 保存类别图供后续分析
        val outH = info.outputShape[2].toInt()
        val outW = info.outputShape[3].toInt()
        lastClassMap = argmax(outputArray, info.numClasses, outH, outW)
        lastOutputHeight = outH
        lastOutputWidth = outW
        lastOriginalBitmap = bitmap

        val maskBitmap = postprocessToMask(outputArray, info, bitmap.width, bitmap.height)
        val overlayBitmap = createOverlay(bitmap, maskBitmap)

        val inferenceTime = System.currentTimeMillis() - startTime

        return SegmentationResult(
            mask = maskBitmap,
            overlay = overlayBitmap,
            original = bitmap,
            inferenceTimeMs = inferenceTime
        )
    }

    fun runInferenceFromPath(imagePath: String): SegmentationResult? {
        // 采样解码：限制最大尺寸不超过 2048px，防止 OOM
        val opts = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeFile(imagePath, opts)
        val maxDim = maxOf(opts.outWidth, opts.outHeight)
        val sampleSize = when {
            maxDim > 4096 -> 4  // 超大图
            maxDim > 2048 -> 2  // 大图
            else -> 1           // 正常
        }
        val decodeOpts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = BitmapFactory.decodeFile(imagePath, decodeOpts) ?: return null
        return runInference(bitmap)
    }

    fun runInferenceFromUri(context: Context, uri: Uri): SegmentationResult? {
        val inputStream = context.contentResolver.openInputStream(uri) ?: return null
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream.close()
        return runInference(bitmap)
    }

    // ==================== 维管束分析 ====================

    /**
     * 计算比例尺：像素 → 微米
     * 如果用户设置了标尺像素长度和实际长度，计算每微米像素数
     */
    private fun computePixelsPerMicron(settings: AppSettings): Double? {
        if (!settings.enableScale) return null
        if (settings.scaleBarPixels <= 0 || settings.scaleBarLength <= 0) return null
        return settings.scaleBarPixels / settings.scaleBarLength
    }

    /**
     * 对最后一次推理结果进行维管束分析
     * @param settings 全局设置（k值、滤波开关、比例尺）
     * @return 分析结果
     */
    fun analyzeConnectedComponents(settings: AppSettings): AnalysisResult? {
        val classMap = lastClassMap ?: return null
        val h = lastOutputHeight
        val w = lastOutputWidth

        // 计算比例尺（像素→微米）
        val pxPerUm = computePixelsPerMicron(settings)

        // 1. 提取目标类别的二值图（默认类别1）
        val targetClass = settings.targetClassId
        val binary = IntArray(h * w)
        for (i in 0 until h * w) {
            binary[i] = if (classMap[i] == targetClass) 1 else 0
        }

        // 2. 连通域标记
        val (labels, numLabels) = connectedComponents(binary, h, w)

        // 3. 计算每个连通域的面积
        val areaMap = IntArray(numLabels + 1) // index 0 是背景
        for (i in 0 until h * w) {
            val label = labels[i]
            if (label > 0) {
                areaMap[label]++
            }
        }

        // 收集所有非背景连通域的面积
        val rawAreas = mutableListOf<Int>()
        for (label in 1..numLabels) {
            if (areaMap[label] > 0) {
                rawAreas.add(areaMap[label])
            }
        }

        val rawCount = rawAreas.size
        if (rawCount == 0) {
            return AnalysisResult(
                totalBundles = 0, meanArea = 0.0, stdDevArea = 0.0,
                minArea = 0.0, maxArea = 0.0,
                filteredOutCount = 0, rawCount = 0,
                clusters = emptyList()
            )
        }

        // 4. 3σ 滤波 — 同时跟踪哪些 label 被保留
        val filteredLabelsAreas = mutableListOf<Pair<Int, Double>>() // (label, area)
        if (settings.enableFilter) {
            val doubleAreas = rawAreas.map { it.toDouble() }
            val mean = doubleAreas.average()
            val std = std(doubleAreas, mean)
            val lower = mean - settings.filterSigma * std
            val upper = mean + settings.filterSigma * std
            var rawIdx = 0
            for (label in 1..numLabels) {
                if (areaMap[label] > 0) {
                    val area = rawAreas[rawIdx].toDouble()
                    if (area >= lower && area <= upper) {
                        filteredLabelsAreas.add(Pair(label, area))
                    }
                    rawIdx++
                }
            }
        } else {
            var rawIdx = 0
            for (label in 1..numLabels) {
                if (areaMap[label] > 0) {
                    filteredLabelsAreas.add(Pair(label, rawAreas[rawIdx].toDouble()))
                    rawIdx++
                }
            }
        }

        val filteredAreas = filteredLabelsAreas.map { it.second }
        val filteredOutCount = rawCount - filteredAreas.size

        // 5. 统计
        val meanArea = filteredAreas.average()
        val stdDev = std(filteredAreas, meanArea)
        val minArea = filteredAreas.minOrNull() ?: 0.0
        val maxArea = filteredAreas.maxOrNull() ?: 0.0

        // 6. K-means 聚类（返回每个数据点的类别分配）
        val (clusters, pointLabels) = kmeansWithLabels(filteredAreas, settings.kmeansK)

        // 7. 构建 label → cluster 映射
        val labelToCluster = mutableMapOf<Int, Int>()
        for (i in filteredLabelsAreas.indices) {
            val label = filteredLabelsAreas[i].first
            labelToCluster[label] = pointLabels[i]
        }

        // 8. 计算微米面积（如果启用了比例尺）
        val areaScale = pxPerUm?.let { 1.0 / (it * it) } // px² → μm² 换算系数
        val meanAreaUm2 = areaScale?.let { meanArea * it }
        val clustersWithUm2 = if (areaScale != null) {
            clusters.map { c ->
                c.copy(meanAreaUm2 = c.meanArea * areaScale)
            }
        } else {
            clusters
        }

        // 9. 生成标注图 + 类别着色图
        val vesselBitmap = createComponentLabelBitmap(labels, h, w, numLabels, areaMap, filteredAreas, rawAreas)
        val origBitmap = lastOriginalBitmap
        val (clusterMask, clusterOverlay) = if (origBitmap != null && clusters.size > 1) {
            createClusterMaskAndOverlay(labels, h, w, labelToCluster, clusters.size, origBitmap)
        } else {
            Pair(null as Bitmap?, null as Bitmap?)
        }

        return AnalysisResult(
            totalBundles = filteredAreas.size,
            meanArea = meanArea,
            meanAreaUm2 = meanAreaUm2,
            stdDevArea = stdDev,
            minArea = minArea,
            maxArea = maxArea,
            filteredOutCount = filteredOutCount,
            rawCount = rawCount,
            clusters = clustersWithUm2,
            vesselBitmap = vesselBitmap,
            clusterMask = clusterMask,
            clusterOverlay = clusterOverlay
        )
    }

    // ==================== 连通域分析 ====================

    /**
     * 二值图连通域标记 (8-连通)
     * 使用两遍扫描算法
     */
    private fun connectedComponents(binary: IntArray, h: Int, w: Int): Pair<IntArray, Int> {
        val labels = IntArray(h * w) // 0 = 背景
        var nextLabel = 1
        val equivalences = mutableMapOf<Int, MutableSet<Int>>()

        // 第一遍扫描
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                if (binary[idx] == 0) continue

                // 检查 8-邻域中的已标记像素
                val neighborLabels = mutableSetOf<Int>()
                for (dy in -1..0) {
                    for (dx in -1..1) {
                        if (dy == 0 && dx == 0) continue
                        if (dy == 0 && dx == -1) continue // 只检查上半部分
                        val ny = y + dy
                        val nx = x + dx
                        if (ny in 0 until h && nx in 0 until w) {
                            val nIdx = ny * w + nx
                            if (labels[nIdx] > 0) {
                                neighborLabels.add(labels[nIdx])
                            }
                        }
                    }
                }

                if (neighborLabels.isEmpty()) {
                    labels[idx] = nextLabel
                    equivalences[nextLabel] = mutableSetOf(nextLabel)
                    nextLabel++
                } else {
                    val minLabel = neighborLabels.min()
                    labels[idx] = minLabel
                    for (nl in neighborLabels) {
                        equivalences.getOrPut(nl) { mutableSetOf(nl) }.addAll(neighborLabels)
                        equivalences.getOrPut(minLabel) { mutableSetOf(minLabel) }.addAll(neighborLabels)
                    }
                }
            }
        }

        // 合并等价类
        val labelMapping = mutableMapOf<Int, Int>()
        var currentNewLabel = 1
        for (label in 1 until nextLabel) {
            val root = findRoot(label, equivalences)
            if (root !in labelMapping) {
                labelMapping[root] = currentNewLabel
                currentNewLabel++
            }
        }

        // 第二遍扫描：重新标记
        for (i in labels.indices) {
            if (labels[i] > 0) {
                val root = findRoot(labels[i], equivalences)
                labels[i] = labelMapping[root] ?: 0
            }
        }

        return Pair(labels, currentNewLabel - 1)
    }

    private fun findRoot(label: Int, equivalences: Map<Int, Set<Int>>): Int {
        val eqSet = equivalences[label] ?: return label
        val minVal = eqSet.min()
        return if (minVal == label) label else findRoot(minVal, equivalences)
    }

    // ==================== 滤波 ====================

    /**
     * 3σ 滤波：剔除面积偏离均值超过 sigma 倍标准差的异常值
     */
    private fun filterByStd(areas: List<Int>, sigma: Double): List<Double> {
        val doubleAreas = areas.map { it.toDouble() }
        val mean = doubleAreas.average()
        val std = std(doubleAreas, mean)
        val lower = mean - sigma * std
        val upper = mean + sigma * std
        return doubleAreas.filter { it >= lower && it <= upper }
    }

    /**
     * 计算标准差
     */
    private fun std(values: List<Double>, mean: Double): Double {
        if (values.size <= 1) return 0.0
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return sqrt(variance)
    }

    // ==================== K-means 聚类（带标签分配） ==================

    /**
     * K-means 聚类，返回聚类统计 + 每个数据点的类别分配
     */
    private fun kmeansWithLabels(data: List<Double>, k: Int): Pair<List<ClusterInfo>, IntArray> {
        if (data.isEmpty() || k <= 1) {
            val mean = data.average()
            val std = std(data, mean)
            val clusters = listOf(
                ClusterInfo(
                    clusterId = 0, count = data.size,
                    meanArea = mean, minArea = data.minOrNull() ?: 0.0,
                    maxArea = data.maxOrNull() ?: 0.0, stdDev = std
                )
            )
            val pointLabels = IntArray(data.size) { 0 }
            return Pair(clusters, pointLabels)
        }

        val effectiveK = minOf(k, data.size)
        val n = data.size

        // 初始化中心点（均匀采样）
        val centers = DoubleArray(effectiveK)
        val step = n / effectiveK
        for (i in 0 until effectiveK) {
            centers[i] = data[i * step]
        }

        val pointLabels = IntArray(n)
        var changed = true
        var maxIter = 50

        while (changed && maxIter > 0) {
            changed = false
            maxIter--

            for (i in 0 until n) {
                var minDist = Double.MAX_VALUE
                var bestCluster = 0
                for (c in 0 until effectiveK) {
                    val dist = Math.abs(data[i] - centers[c])
                    if (dist < minDist) {
                        minDist = dist
                        bestCluster = c
                    }
                }
                if (pointLabels[i] != bestCluster) {
                    pointLabels[i] = bestCluster
                    changed = true
                }
            }

            for (c in 0 until effectiveK) {
                val points = data.filterIndexed { idx, _ -> pointLabels[idx] == c }
                if (points.isNotEmpty()) {
                    centers[c] = points.average()
                }
            }
        }

        // 生成聚类结果
        val clusterMap = mutableMapOf<Int, MutableList<Double>>()
        for (i in 0 until n) {
            clusterMap.getOrPut(pointLabels[i]) { mutableListOf() }.add(data[i])
        }

        val clusters = clusterMap.entries.sortedBy { it.key }.map { (id, values) ->
            val mean = values.average()
            val std = std(values, mean)
            ClusterInfo(
                clusterId = id, count = values.size,
                meanArea = mean, minArea = values.minOrNull() ?: 0.0,
                maxArea = values.maxOrNull() ?: 0.0, stdDev = std
            )
        }
        return Pair(clusters, pointLabels)
    }

    // ==================== 标注图生成 ==================

    /**
     * 生成标注了编号和边界的维管束图
     */
    private fun createComponentLabelBitmap(
        labels: IntArray, h: Int, w: Int,
        numLabels: Int, areaMap: IntArray,
        filteredAreas: List<Double>, rawAreas: List<Int>
    ): Bitmap? {
        val origBitmap = lastOriginalBitmap ?: return null
        val resized = Bitmap.createScaledBitmap(origBitmap, w, h, true)
        val canvas = android.graphics.Canvas(resized)
        val paint = android.graphics.Paint()

        // 找出被滤波剔除的标签
        val areaList = rawAreas
        val filteredSet = mutableSetOf<Int>()
        if (areaList.size != filteredAreas.size) {
            var rawIdx = 0
            for (label in 1..numLabels) {
                if (areaMap[label] > 0 && rawIdx < areaList.size) {
                    val area = areaList[rawIdx].toDouble()
                    val mean = filteredAreas.average()
                    val std = std(filteredAreas, mean)
                    val sigma = 3.0
                    if (area < mean - sigma * std || area > mean + sigma * std) {
                        filteredSet.add(label)
                    }
                    rawIdx++
                }
            }
        }

        // 绘制每个连通域的边界和编号
        paint.style = android.graphics.Paint.Style.STROKE
        paint.strokeWidth = 2f

        val textPaint = android.graphics.Paint()
        textPaint.color = android.graphics.Color.WHITE
        textPaint.textSize = 20f
        textPaint.isFakeBoldText = true

        val visited = BooleanArray(numLabels + 1)
        var displayId = 1
        val labelToDisplayId = mutableMapOf<Int, Int>()

        for (y in 0 until h) {
            for (x in 0 until w) {
                val label = labels[y * w + x]
                if (label > 0 && !visited[label]) {
                    visited[label] = true
                    if (label !in filteredSet) {
                        labelToDisplayId[label] = displayId
                        displayId++
                    }
                }
            }
        }

        // 第二次遍历：绘制边界
        visited.fill(false)
        val boundaryPaint = android.graphics.Paint()
        boundaryPaint.style = android.graphics.Paint.Style.STROKE
        boundaryPaint.strokeWidth = 2f

        for (y in 0 until h) {
            for (x in 0 until w) {
                val label = labels[y * w + x]
                if (label > 0 && !visited[label] && label !in filteredSet) {
                    visited[label] = true
                    val display = labelToDisplayId[label] ?: continue

                    // 用不同颜色区分不同连通域
                    val color = colors[(display - 1) % colors.size]
                    boundaryPaint.color = color

                    // 绘制边界
                    drawBoundary(canvas, labels, label, h, w, boundaryPaint, paint)

                    // 在质心绘制编号
                    val cx = findCentroid(labels, label, h, w)
                    if (cx != null) {
                        canvas.drawText("#$display", cx.first, cx.second, textPaint)
                    }
                }
            }
        }

        return resized
    }

    private val colors = intArrayOf(
        0xFFFF0000.toInt(), 0xFF00FF00.toInt(), 0xFF0000FF.toInt(),
        0xFFFFFF00.toInt(), 0xFFFF00FF.toInt(), 0xFF00FFFF.toInt(),
        0xFFFF8800.toInt(), 0xFF8800FF.toInt(), 0xFF0088FF.toInt(),
        0xFFFF8888.toInt(), 0xFF88FF88.toInt(), 0xFF8888FF.toInt()
    )

    private fun drawBoundary(
        canvas: android.graphics.Canvas,
        labels: IntArray, label: Int, h: Int, w: Int,
        boundaryPaint: android.graphics.Paint, tempPaint: android.graphics.Paint
    ) {
        val points = mutableListOf<Pair<Float, Float>>()
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (labels[y * w + x] == label) {
                    // 检查是否为边界（邻域内有其他标签或背景）
                    var isBoundary = false
                    for (dy in -1..1) {
                        for (dx in -1..1) {
                            val ny = y + dy
                            val nx = x + dx
                            if (ny < 0 || ny >= h || nx < 0 || nx >= w) {
                                isBoundary = true
                            } else if (labels[ny * w + nx] != label) {
                                isBoundary = true
                            }
                        }
                    }
                    if (isBoundary) {
                        points.add(Pair(x.toFloat(), y.toFloat()))
                    }
                }
            }
        }
        // 绘制所有边界点（简化，绘制为小圆点）
        for (p in points) {
            canvas.drawCircle(p.first, p.second, 1.5f, boundaryPaint)
        }
    }

    private fun findCentroid(
        labels: IntArray, label: Int, h: Int, w: Int
    ): Pair<Float, Float>? {
        var sumX = 0.0
        var sumY = 0.0
        var count = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (labels[y * w + x] == label) {
                    sumX += x
                    sumY += y
                    count++
                }
            }
        }
        return if (count > 0) Pair((sumX / count).toFloat(), (sumY / count).toFloat())
        else null
    }

    // ==================== 基础工具方法 ====================

    private fun preprocess(bitmap: Bitmap, info: ModelInfo): ai.onnxruntime.OnnxTensor {
        val width = info.inputWidth
        val height = info.inputHeight
        val channels = info.inputChannels

        val inputArray = FloatArray(channels * height * width)
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f
                val idx = y * width + x
                inputArray[idx] = r
                inputArray[height * width + idx] = g
                inputArray[2 * height * width + idx] = b
            }
        }

        val shape = longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())

        return ai.onnxruntime.OnnxTensor.createTensor(
            ortEnv, java.nio.FloatBuffer.wrap(inputArray), shape
        )
    }

    private fun readOutputAsFloatArray(tensor: ai.onnxruntime.OnnxTensor, info: ModelInfo): FloatArray {
        val floatBuffer = tensor.getFloatBuffer()
        val floatArray = FloatArray(floatBuffer.remaining())
        floatBuffer.get(floatArray)
        return floatArray
    }

    private fun argmax(outputData: FloatArray, numClasses: Int, h: Int, w: Int): IntArray {
        val classMap = IntArray(h * w)
        for (i in 0 until h * w) {
            var maxVal = Float.MIN_VALUE
            var maxClass = 0
            for (c in 0 until numClasses) {
                val v = outputData[c * h * w + i]
                if (v > maxVal) { maxVal = v; maxClass = c }
            }
            classMap[i] = maxClass
        }
        return classMap
    }

    // ==================== 类别着色 Mask/Overlay ==================

    /**
     * 生成按 K-means 类别着色的 Mask 和 Overlay
     * 每个连通域按所属类别着不同颜色，背景透明
     */
    private fun createClusterMaskAndOverlay(
        labels: IntArray, h: Int, w: Int,
        labelToCluster: Map<Int, Int>,
        numClusters: Int,
        orig: Bitmap
    ): Pair<Bitmap, Bitmap> {
        val clusterColors = intArrayOf(
            0xFFFF4444.toInt(), 0xFF44BB44.toInt(), 0xFF4488FF.toInt(),
            0xFFFFBB44.toInt(), 0xFFFF44FF.toInt(), 0xFF44FFFF.toInt()
        )

        // 构建像素颜色数组（1024x1024 分辨率）
        val maskColors = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                val label = labels[idx]
                if (label > 0) {
                    val clusterId = labelToCluster[label]
                    val colorIdx = if (clusterId != null) clusterId % clusterColors.size else 0
                    maskColors[idx] = clusterColors[colorIdx]
                } else {
                    maskColors[idx] = 0x00000000.toInt() // 背景透明
                }
            }
        }

        // 生成 Mask（类别着色，背景透明）
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        mask.setPixels(maskColors, 0, w, 0, 0, w, h)

        // 生成 Overlay（着色叠加到原图，先在 1024x1024 生成再缩放）
        val semitransparent = IntArray(w * h)
        for (i in 0 until w * h) {
            val c = maskColors[i]
            if (c != 0) {
                semitransparent[i] = (c and 0x00FFFFFF) or 0xC0000000.toInt() // 75% 透明度
            } else {
                semitransparent[i] = 0
            }
        }
        val alphaMaskSmall = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        alphaMaskSmall.setPixels(semitransparent, 0, w, 0, 0, w, h)
        // 缩放到原图尺寸再叠加
        val alphaMask = Bitmap.createScaledBitmap(alphaMaskSmall, orig.width, orig.height, true)
        alphaMaskSmall.recycle()
        val overlay = orig.copy(Bitmap.Config.ARGB_8888, true)
        val overlayCanvas = android.graphics.Canvas(overlay)
        overlayCanvas.drawBitmap(alphaMask, 0f, 0f, null)
        alphaMask.recycle()

        // 缩放 Mask 到原图尺寸
        val scaledMask = Bitmap.createScaledBitmap(mask, orig.width, orig.height, true)
        mask.recycle()

        return Pair(scaledMask, overlay)
    }

    private fun postprocessToMask(outputData: FloatArray, info: ModelInfo, targetWidth: Int, targetHeight: Int): Bitmap {
        val numClasses = info.numClasses
        val outH = info.outputShape[2].toInt()
        val outW = info.outputShape[3].toInt()
        val classMap = argmax(outputData, numClasses, outH, outW)

        val maskColors = IntArray(outH * outW)
        val clsColors = intArrayOf(
            0xFFFF0000.toInt(), 0xFF00FF00.toInt(), 0xFF0000FF.toInt(),
            0xFFFFFF00.toInt(), 0xFFFF00FF.toInt(), 0xFF00FFFF.toInt(),
            0xFFFF8800.toInt(), 0xFF8800FF.toInt()
        )

        for (i in classMap.indices) {
            val cls = classMap[i]
            maskColors[i] = if (cls < clsColors.size) clsColors[cls] else 0xFFFFFFFF.toInt()
        }

        val maskBitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        maskBitmap.setPixels(maskColors, 0, outW, 0, 0, outW, outH)
        return Bitmap.createScaledBitmap(maskBitmap, targetWidth, targetHeight, true)
    }

    private fun createOverlay(original: Bitmap, mask: Bitmap): Bitmap {
        val overlay = original.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = android.graphics.Canvas(overlay)
        val paint = android.graphics.Paint()
        paint.alpha = 100
        canvas.drawBitmap(mask, 0f, 0f, paint)
        return overlay
    }

    data class ScaleBarLocation(
        val lengthPx: Int,
        val left: Int, val top: Int, val right: Int, val bottom: Int
    )

    companion object {
        private data class Region(val x1: Int, val y1: Int, val x2: Int, val y2: Int)

        /**
         * 自动检测显微图片中的比例尺，返回位置 + 长度
         */
        fun detectScaleBar(bitmap: Bitmap): ScaleBarLocation? {
            val w = bitmap.width; val h = bitmap.height
            if (w < 100 || h < 100) return null
            val regions = listOf(
                Region((w * 0.70).toInt(), (h * 0.85).toInt(), w, h),
                Region(0, 0, (w * 0.30).toInt(), (h * 0.15).toInt()),
                Region(0, (h * 0.85).toInt(), (w * 0.30).toInt(), h),
                Region((w * 0.70).toInt(), 0, w, (h * 0.15).toInt())
            )
            for (r in regions) {
                val hb = scanHorizontal(bitmap, r); if (hb != null) return hb
                val vb = scanVertical(bitmap, r); if (vb != null) return vb
            }
            return null
        }

        // 扫描水平条，返回 ScaleBarLocation
        private fun scanHorizontal(bitmap: Bitmap, reg: Region): ScaleBarLocation? {
            val (x1, y1, x2, y2) = reg; val rw = x2 - x1; val rh = y2 - y1
            if (rw < 60 || rh < 5) return null
            val px = IntArray(rw * rh); bitmap.getPixels(px, 0, rw, x1, y1, rw, rh)
            fun b(p: Int) = 0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF)

            // Map<起始x聚类, List<(长度, 行号)>>
            val cand = mutableMapOf<Int, MutableList<Pair<Int, Int>>>()
            for (row in 0 until rh) {
                val rs = row * rw; var col = 0
                while (col < rw) {
                    if (b(px[rs + col]) < 100f) {
                        val s = col; while (col < rw && b(px[rs + col]) < 100f) col++
                        val len = col - s
                        if (len in 50..(rw * 0.85f).toInt()) {
                            val lb = if (s > 5) (0 until 5).map { b(px[rs + s - 1 - it]) }.average().toFloat() else 130f
                            val rb = if (col + 5 < rw) (0 until 5).map { b(px[rs + col + it]) }.average().toFloat() else 130f
                            val ad = (s until col).map { b(px[rs + it]) }.average().toFloat()
                            if (lb >= 130f && rb >= 130f && lb - ad > 30f && rb - ad > 30f) {
                                cand.getOrPut((s / 10) * 10) { mutableListOf() }.add(Pair(len, row))
                                continue
                            }
                        }
                    }
                    col++
                }
            }

            var best: ScaleBarLocation? = null
            for ((key, rows) in cand) {
                if (rows.size < 3) continue
                val minR = rows.minOf { it.second }; val maxR = rows.maxOf { it.second }
                val thick = maxR - minR + 1
                val lens = rows.map { it.first }.sorted()
                val mid = if (lens.size % 2 == 1) lens[lens.size / 2] else (lens[lens.size / 2 - 1] + lens[lens.size / 2]) / 2
                if (thick > 0 && mid / thick > 4 && (best == null || mid > best.lengthPx)) {
                    best = ScaleBarLocation(mid, x1 + key, y1 + minR, x1 + key + mid, y1 + maxR)
                }
            }
            return best
        }

        // 扫描垂直条，返回 ScaleBarLocation
        private fun scanVertical(bitmap: Bitmap, reg: Region): ScaleBarLocation? {
            val (x1, y1, x2, y2) = reg; val rw = x2 - x1; val rh = y2 - y1
            if (rw < 5 || rh < 60) return null
            val px = IntArray(rw * rh); bitmap.getPixels(px, 0, rw, x1, y1, rw, rh)
            fun b(p: Int) = 0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF)

            val cand = mutableMapOf<Int, MutableList<Pair<Int, Int>>>()
            for (col in 0 until rw) {
                var row = 0
                while (row < rh) {
                    if (b(px[row * rw + col]) < 100f) {
                        val s = row; while (row < rh && b(px[row * rw + col]) < 100f) row++
                        val len = row - s
                        if (len in 50..(rh * 0.85f).toInt()) {
                            val tb = if (s > 5) (0 until 5).map { b(px[(s - 1 - it) * rw + col]) }.average().toFloat() else 130f
                            val bb = if (row + 5 < rh) (0 until 5).map { b(px[(row + it) * rw + col]) }.average().toFloat() else 130f
                            val ad = (s until row).map { b(px[it * rw + col]) }.average().toFloat()
                            if (tb >= 130f && bb >= 130f && tb - ad > 30f && bb - ad > 30f) {
                                cand.getOrPut((s / 10) * 10) { mutableListOf() }.add(Pair(len, col))
                                continue
                            }
                        }
                    }
                    row++
                }
            }

            var best: ScaleBarLocation? = null
            for ((key, cols) in cand) {
                if (cols.size < 3) continue
                val minC = cols.minOf { it.second }; val maxC = cols.maxOf { it.second }
                val thick = maxC - minC + 1
                val lens = cols.map { it.first }.sorted()
                val mid = if (lens.size % 2 == 1) lens[lens.size / 2] else (lens[lens.size / 2 - 1] + lens[lens.size / 2]) / 2
                if (thick > 0 && mid / thick > 4 && (best == null || mid > best.lengthPx)) {
                    best = ScaleBarLocation(mid, x1 + minC, y1 + key, x1 + maxC, y1 + key + mid)
                }
            }
            return best
        }
    }
}