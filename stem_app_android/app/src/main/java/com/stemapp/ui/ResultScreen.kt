package com.stemapp.ui

import android.graphics.Bitmap
import android.os.Environment
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.stemapp.ml.OnnxInference
import com.stemapp.model.AnalysisResult
import com.stemapp.model.ResultRepository
import com.stemapp.model.SettingsManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import kotlin.math.sqrt
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(
    imagePath: String,
    sampleId: String,
    onnxInference: OnnxInference,
    settingsManager: SettingsManager,
    onNavigateBack: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var segResult by remember { mutableStateOf<OnnxInference.SegmentationResult?>(null) }
    var analysisResult by remember { mutableStateOf<AnalysisResult?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    var currentView by remember { mutableStateOf("original") }
    val snackbarHostState = remember { SnackbarHostState() }

    val settings = remember { settingsManager.getSettings() }
    val resultRepo = remember { ResultRepository(context) }

    // 比例尺标定状态
    var isCalibrating by remember { mutableStateOf(false) }
    var calibPointA by remember { mutableStateOf<Offset?>(null) }
    var calibPointB by remember { mutableStateOf<Offset?>(null) }
    var showCalibDialog by remember { mutableStateOf(false) }
    var calibLengthText by remember { mutableStateOf("") }

    // 图片缩放状态
    var zoomScale by remember { mutableFloatStateOf(1f) }
    var zoomOffset by remember { mutableStateOf(Offset.Zero) }

    // 比例尺自动检测状态
    var autoDetectedDist by remember { mutableIntStateOf(-1) }
    var adjustedPixelDist by remember { mutableDoubleStateOf(0.0) }
    var detectedBarLocation by remember { mutableStateOf<OnnxInference.ScaleBarLocation?>(null) }

    // 是否已保存结果
    var resultSaved by remember { mutableStateOf(false) }

    // 进入标定模式时自动检测比例尺
    LaunchedEffect(isCalibrating) {
        if (isCalibrating && segResult != null) {
            autoDetectedDist = -1
            withContext(Dispatchers.IO) {
                val detected = OnnxInference.detectScaleBar(segResult!!.original)
                if (detected != null && detected.lengthPx > 0) {
                    detectedBarLocation = detected
                    autoDetectedDist = detected.lengthPx
                    adjustedPixelDist = detected.lengthPx.toDouble()
                } else { autoDetectedDist = 0; detectedBarLocation = null }
            }
            if (autoDetectedDist > 0) showCalibDialog = true
        }
    }

    // 执行推理 + 分析
    LaunchedEffect(imagePath) {
        withContext(Dispatchers.IO) {
            try {
                val seg = onnxInference.runInferenceFromPath(imagePath)
                if (seg != null) {
                    segResult = seg
                    val analysis = onnxInference.analyzeConnectedComponents(settings)
                    analysisResult = analysis
                } else {
                    errorMessage = "推理失败：请检查模型是否正确加载"
                }
            } catch (e: OutOfMemoryError) {
                errorMessage = "内存不足，请使用较小图片"
            } catch (e: Exception) {
                errorMessage = "推理出错：${e.message}"
            }
            isLoading = false
        }
    }

    // 标定确认对话框
    if (showCalibDialog) {
        val isAutoDetected = autoDetectedDist > 0
        AlertDialog(
            onDismissRequest = { showCalibDialog = false; isCalibrating = false; calibPointA = null; calibPointB = null },
            title = { Text("比例尺标定") },
            text = {
                Column {
                    Text(if (isAutoDetected) "自动检测到标尺" else "手动标定",
                        fontWeight = FontWeight.Bold,
                        color = if (isAutoDetected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary)
                    Spacer(Modifier.height(4.dp))
                    Text("像素长度: ${adjustedPixelDist.toInt()} px",
                        style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(8.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
                        Text("微调: ", fontSize = 12.sp)
                        TextButton(onClick = { adjustedPixelDist = (adjustedPixelDist - 10).coerceAtLeast(10.0) }, modifier = Modifier.height(32.dp)) { Text("-10", fontSize = 13.sp) }
                        TextButton(onClick = { adjustedPixelDist = (adjustedPixelDist - 5).coerceAtLeast(10.0) }, modifier = Modifier.height(32.dp)) { Text("-5", fontSize = 13.sp) }
                        TextButton(onClick = { adjustedPixelDist = (adjustedPixelDist - 1).coerceAtLeast(10.0) }, modifier = Modifier.height(32.dp)) { Text("-1", fontSize = 13.sp) }
                        TextButton(onClick = { adjustedPixelDist = (adjustedPixelDist + 1).coerceAtMost(5000.0) }, modifier = Modifier.height(32.dp)) { Text("+1", fontSize = 13.sp) }
                        TextButton(onClick = { adjustedPixelDist = (adjustedPixelDist + 5).coerceAtMost(5000.0) }, modifier = Modifier.height(32.dp)) { Text("+5", fontSize = 13.sp) }
                        TextButton(onClick = { adjustedPixelDist = (adjustedPixelDist + 10).coerceAtMost(5000.0) }, modifier = Modifier.height(32.dp)) { Text("+10", fontSize = 13.sp) }
                    }
                    if (!isAutoDetected) Text("沿标尺线两端各点一下，然后微调", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Spacer(Modifier.height(12.dp))
                    OutlinedTextField(value = calibLengthText, onValueChange = { calibLengthText = it },
                        label = { Text("实际长度 (μm)") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        singleLine = true, modifier = Modifier.fillMaxWidth())
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    val length = calibLengthText.toDoubleOrNull()
                    if (length != null && length > 0 && adjustedPixelDist > 0) {
                        settingsManager.saveSettings(settings.copy(enableScale = true, scaleBarPixels = adjustedPixelDist, scaleBarLength = length))
                        scope.launch { snackbarHostState.showSnackbar("比例尺已标定: ${adjustedPixelDist.toInt()} px = $length μm") }
                        isCalibrating = false; calibPointA = null; calibPointB = null; showCalibDialog = false; zoomScale = 1f; zoomOffset = Offset.Zero
                    } else { scope.launch { snackbarHostState.showSnackbar("请输入有效的长度数值") } }
                }) { Text("保存标定") }
            },
            dismissButton = { TextButton(onClick = { showCalibDialog = false; isCalibrating = false; calibPointA = null; calibPointB = null; detectedBarLocation = null }) { Text("取消") } }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("分割结果") },
                navigationIcon = { IconButton(onClick = onNavigateBack) { Icon(Icons.Default.ArrowBack, contentDescription = "返回") } },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Column(
            modifier = Modifier.fillMaxSize().padding(paddingValues).verticalScroll(rememberScrollState())
        ) {
            when {
                isLoading -> {
                    Box(Modifier.fillMaxWidth().height(400.dp), contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(); Spacer(Modifier.height(16.dp)); Text("推理 + 分析中...")
                        }
                    }
                }
                errorMessage != null -> {
                    Box(Modifier.fillMaxWidth().height(400.dp), contentAlignment = Alignment.Center) {
                        Text(errorMessage ?: "未知错误", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(16.dp))
                    }
                }
                segResult != null -> {
                    val seg = segResult!!
                    val analysis = analysisResult
                    val hasAnalysis = analysis != null

                    ViewSwitcher(currentView = currentView, onViewChange = { currentView = it; zoomScale = 1f; zoomOffset = Offset.Zero },
                        hasAnalysis = hasAnalysis, isCalibrating = isCalibrating)

                    Spacer(Modifier.height(4.dp))

                    // --- 结果视图 ---
                    if (currentView == "analysis" && hasAnalysis) {
                        ResultInfoCard(
                            sampleId = sampleId,
                            analysis = analysis!!,
                            settings = settings,
                            inferenceTimeMs = seg.inferenceTimeMs,
                            segResult = seg,
                            resultRepo = resultRepo,
                            onSaveResult = {
                                scope.launch(Dispatchers.IO) {
                                    resultRepo.saveResult(sampleId, analysis, settingsManager.getSettings())
                                    resultSaved = true
                                    withContext(Dispatchers.Main) {
                                        snackbarHostState.showSnackbar("结果已保存到本地")
                                    }
                                }
                            },
                            resultSaved = resultSaved,
                            snackbarHostState = snackbarHostState
                        )
                    }

                    // --- 图片展示（所有视图都显示图片） ---
                    val displayBitmap = when {
                        currentView == "original" -> seg.original
                        currentView == "mask" -> if (hasAnalysis) analysis!!.clusterMask ?: seg.mask else seg.mask
                        currentView == "overlay" -> if (hasAnalysis) analysis!!.clusterOverlay ?: seg.overlay else seg.overlay
                        currentView == "calibrate" -> seg.original
                        currentView == "analysis" -> analysis?.vesselBitmap ?: seg.overlay
                        else -> seg.original
                    }

                    Box(
                        modifier = Modifier.fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp)).padding(horizontal = 8.dp)
                    ) {
                        Box(modifier = Modifier.fillMaxWidth().pointerInput(isCalibrating, zoomScale) {
                            awaitEachGesture {
                                var isZooming = false; var lastCentroid = Offset.Zero; var lastSpan = 0f
                                val firstDown = awaitFirstDown(requireUnconsumed = false)
                                val startPos = firstDown.position; var wasTap = true; var consumed = false
                                do {
                                    val event = awaitPointerEvent()
                                    if (event.type == PointerEventType.Release) {
                                        if (wasTap && isCalibrating && !consumed) {
                                            val imgX = (startPos.x - zoomOffset.x) / zoomScale
                                            val imgY = (startPos.y - zoomOffset.y) / zoomScale
                                            val imgOffset = Offset(imgX, imgY)
                                            if (calibPointA == null) { calibPointA = imgOffset
                                            } else if (calibPointB == null) {
                                                calibPointB = imgOffset
                                                val dx = imgOffset.x - calibPointA!!.x; val dy = imgOffset.y - calibPointA!!.y
                                                adjustedPixelDist = sqrt((dx*dx + dy*dy).toDouble()).coerceAtLeast(10.0)
                                                showCalibDialog = true
                                            }
                                        }
                                        event.changes.forEach { it.consume() }; break
                                    }
                                    val changes = event.changes.filter { it.pressed }
                                    if (changes.size >= 2) {
                                        isZooming = true; wasTap = false
                                        val centroid = changes.fold(Offset.Zero) { acc, c -> acc + c.position } / changes.size.toFloat()
                                        val newSpan = sqrt((changes[0].position.x-changes[1].position.x).let{it*it}+(changes[0].position.y-changes[1].position.y).let{it*it})
                                        if (lastSpan > 0f) {
                                            val newScale = (zoomScale * (newSpan / lastSpan)).coerceIn(1f, 5f)
                                            val scaleChange = newScale / zoomScale
                                            zoomOffset = centroid - (centroid - zoomOffset) * scaleChange; zoomScale = newScale
                                        }
                                        if (lastCentroid != Offset.Zero) zoomOffset += centroid - lastCentroid
                                        lastSpan = newSpan; lastCentroid = centroid; consumed = true
                                    } else if (changes.size == 1 && (isZooming || zoomScale > 1f)) {
                                        val c = changes[0]; val delta = c.position - c.previousPosition
                                        if (delta.getDistance() > 8f || isZooming) { wasTap = false; zoomOffset += delta; consumed = true }
                                    }
                                    if (consumed) event.changes.forEach { it.consume() }
                                    lastCentroid = changes.fold(Offset.Zero) { acc, c -> acc + c.position } / changes.size.toFloat()
                                } while (true)
                            }
                        }) {
                            Box(modifier = Modifier.fillMaxWidth().graphicsLayer {
                                scaleX = zoomScale; scaleY = zoomScale; translationX = zoomOffset.x; translationY = zoomOffset.y
                                transformOrigin = androidx.compose.ui.graphics.TransformOrigin(0f, 0f)
                            }) {
                                Image(bitmap = displayBitmap.asImageBitmap(), contentDescription = "结果图像", modifier = Modifier.fillMaxWidth())

                                // 自动检测到的比例尺高亮框
                                if (detectedBarLocation != null) {
                                    Canvas(modifier = Modifier.matchParentSize()) {
                                        val cw = size.width; val ch = size.height
                                        val bw = displayBitmap.width.toFloat(); val bh = displayBitmap.height.toFloat()
                                        val sx = if (bw > 0) cw / bw else 1f; val sy = if (bh > 0) ch / bh else 1f
                                        val loc = detectedBarLocation!!
                                        val hl = loc.left * sx; val ht = loc.top * sy
                                        val hr = loc.right * sx; val hb = loc.bottom * sy
                                        // 半透明绿色填充
                                        drawRect(
                                            color = Color(0x3344FF44.toInt()),
                                            topLeft = Offset(hl, ht),
                                            size = androidx.compose.ui.geometry.Size(hr - hl, hb - ht)
                                        )
                                        // 绿色边框
                                        drawRect(
                                            color = Color(0xFF44FF44.toInt()),
                                            topLeft = Offset(hl, ht),
                                            size = androidx.compose.ui.geometry.Size(hr - hl, hb - ht),
                                            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 3f / zoomScale)
                                        )
                                        // 标签文字
                                        drawContext.canvas.nativeCanvas.apply {
                                            val tagPaint = android.graphics.Paint().apply {
                                                color = 0xFF44FF44.toInt(); textSize = 28f / zoomScale
                                                isFakeBoldText = true
                                            }
                                            drawText("检测到标尺 ${loc.lengthPx}px", hl + 8f / zoomScale, ht - 10f / zoomScale, tagPaint)
                                        }
                                    }
                                }

                                if (isCalibrating) Canvas(modifier = Modifier.matchParentSize()) {
                                    val lineColor = Color(0xFFFF3D00); val dr = 8f / zoomScale
                                    calibPointA?.let { a ->
                                        drawCircle(lineColor, dr, a); drawCircle(Color.White, dr - 3f / zoomScale, a); drawCircle(lineColor, 3f / zoomScale, a)
                                        calibPointB?.let { b ->
                                            drawCircle(lineColor, dr, b); drawCircle(Color.White, dr - 3f / zoomScale, b); drawCircle(lineColor, 3f / zoomScale, b); drawLine(lineColor, a, b, 4f / zoomScale)
                                            val mx = (a.x + b.x) / 2f; val my = (a.y + b.y) / 2f - 20f / zoomScale
                                            val dist = sqrt(((b.x-a.x)*(b.x-a.x)+(b.y-a.y)*(b.y-a.y)).toDouble()); val ts = 48f / zoomScale
                                            drawContext.canvas.nativeCanvas.apply {
                                                val bg = android.graphics.Paint().apply { color = 0xBB000000.toInt(); textSize = ts; isFakeBoldText = true; textAlign = android.graphics.Paint.Align.CENTER }
                                                val fg = android.graphics.Paint().apply { color = 0xFFFF3D00.toInt(); textSize = ts; isFakeBoldText = true; textAlign = android.graphics.Paint.Align.CENTER }
                                                drawText("${dist.toInt()} px", mx+1, my+1, bg); drawText("${dist.toInt()} px", mx, my, fg)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // 缩放控制栏
                        Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 2.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
                            if (isCalibrating) {
                                val status = when { autoDetectedDist > 0 -> "检测到标尺"; autoDetectedDist == 0 && calibPointA == null -> "未检测到，请点击两端"; calibPointA == null -> "点击标尺起点"; calibPointB == null -> "点击标尺终点"; else -> "标定完成" }
                                Text(status, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
                                if (autoDetectedDist == 0 && calibPointA == null && segResult != null) {
                                    TextButton(onClick = {
                                        autoDetectedDist = -1; detectedBarLocation = null
                                        scope.launch(Dispatchers.IO) { val d = OnnxInference.detectScaleBar(segResult!!.original); if (d != null && d.lengthPx > 0) { detectedBarLocation = d; autoDetectedDist = d.lengthPx; adjustedPixelDist = d.lengthPx.toDouble(); showCalibDialog = true } else { autoDetectedDist = 0; detectedBarLocation = null } }
                                    }) { Text("重新检测", fontSize = 12.sp) }
                                }
                            } else Spacer(Modifier.size(1.dp))
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                IconButton(onClick = { zoomScale = (zoomScale / 1.5f).coerceAtLeast(1f); if (zoomScale == 1f) zoomOffset = Offset.Zero }, modifier = Modifier.size(28.dp)) { Icon(Icons.Default.Remove, "缩小", modifier = Modifier.size(16.dp)) }
                                Text("%.0f%%".format(zoomScale * 100), fontSize = 11.sp, fontWeight = FontWeight.Medium, modifier = Modifier.padding(horizontal = 2.dp))
                                IconButton(onClick = { zoomScale = (zoomScale * 1.5f).coerceAtMost(5f) }, modifier = Modifier.size(28.dp)) { Icon(Icons.Default.Add, "放大", modifier = Modifier.size(16.dp)) }
                                IconButton(onClick = { zoomScale = 1f; zoomOffset = Offset.Zero }, modifier = Modifier.size(28.dp)) { Icon(Icons.Default.Refresh, "重置", modifier = Modifier.size(14.dp)) }
                            }
                        }

                    // 比例尺状态
                    val curSet = settingsManager.getSettings()
                    if (curSet.enableScale && curSet.scaleBarPixels > 0) {
                        val ratio = curSet.scaleBarPixels / curSet.scaleBarLength
                        if (ratio > 0) Surface(modifier = Modifier.padding(horizontal = 12.dp, vertical = 2.dp), shape = RoundedCornerShape(4.dp), color = Color(0xBB000000)) {
                            Text("比例尺: %.0f px = %.0f μm  (%.2f px/μm)".format(curSet.scaleBarPixels, curSet.scaleBarLength, ratio),
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp), color = Color.White, fontSize = 12.sp)
                        }
                    }

                    // 操作按钮
                    if (currentView != "analysis") {
                        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { scope.launch { saveResult(context, seg.overlay); snackbarHostState.showSnackbar("结果已保存到相册") } }, modifier = Modifier.weight(1f)) {
                                Icon(Icons.Default.Save, null); Spacer(Modifier.width(4.dp)); Text("保存", fontSize = 13.sp)
                            }
                            Button(onClick = {
                                if (isCalibrating) { isCalibrating = false; calibPointA = null; calibPointB = null; currentView = "original" }
                                else { currentView = "calibrate"; isCalibrating = true; calibPointA = null; calibPointB = null }
                                zoomScale = 1f; zoomOffset = Offset.Zero
                            }, modifier = Modifier.weight(1f), colors = ButtonDefaults.buttonColors(containerColor = if (isCalibrating) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.secondary)) {
                                Icon(if (isCalibrating) Icons.Default.Close else Icons.Default.Straighten, null); Spacer(Modifier.width(4.dp))
                                Text(if (isCalibrating) "退出标定" else "标定比例尺", fontSize = 13.sp)
                            }
                        }
                    }

                    Spacer(Modifier.height(16.dp))
                }
            }
        }
    }
}

// ========== 结果信息卡片 ==========

@Composable
private fun ResultInfoCard(
    sampleId: String,
    analysis: AnalysisResult,
    settings: com.stemapp.model.AppSettings,
    inferenceTimeMs: Long,
    segResult: OnnxInference.SegmentationResult,
    resultRepo: ResultRepository,
    onSaveResult: () -> Unit,
    resultSaved: Boolean,
    snackbarHostState: SnackbarHostState
) {
    val hasScale = analysis.meanAreaUm2 != null
    val scope = rememberCoroutineScope()
    val context = LocalContext.current
    var exporting by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth().padding(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("📋 分析结果", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))

            if (sampleId.isNotEmpty()) {
                InfoRow("样本ID", sampleId)
                Spacer(Modifier.height(4.dp))
            }

            InfoRow("维管束总数", "${analysis.totalBundles}")
            InfoRow("推理耗时", "${inferenceTimeMs} ms")

            if (hasScale) {
                val ratio = settings.scaleBarPixels / settings.scaleBarLength
                InfoRow("比例尺", "%.0f px = %.0f μm".format(settings.scaleBarPixels, settings.scaleBarLength))
            }

            Spacer(Modifier.height(8.dp))
            Divider()
            Spacer(Modifier.height(8.dp))

            val clusterColors = listOf(0xFFFF4444, 0xFF44BB44, 0xFF4488FF, 0xFFFFBB44, 0xFFFF44FF, 0xFF44FFFF)

            analysis.clusters.forEachIndexed { idx, cluster ->
                val color = clusterColors[idx % clusterColors.size]
                val areaStr = if (hasScale && cluster.meanAreaUm2 != null) "${"%.1f".format(cluster.meanAreaUm2)} μm²"
                    else "${"%.1f".format(cluster.meanArea)} px"
                Surface(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    shape = RoundedCornerShape(8.dp),
                    color = Color(color).copy(alpha = 0.12f)
                ) {
                    Column(Modifier.padding(12.dp)) {
                        Text("类别 ${idx + 1}", fontWeight = FontWeight.Bold, color = Color(color), fontSize = 15.sp)
                        Spacer(Modifier.height(4.dp))
                        InfoRow("数量", "${cluster.count}")
                        InfoRow("平均面积", areaStr)
                        val rangeStr = if (hasScale && cluster.meanAreaUm2 != null) {
                            val scale = cluster.meanAreaUm2 / cluster.meanArea
                            "${"%.1f".format(cluster.minArea * scale)} - ${"%.1f".format(cluster.maxArea * scale)} μm²"
                        } else {
                            "${"%.1f".format(cluster.minArea)} - ${"%.1f".format(cluster.maxArea)} px"
                        }
                        InfoRow("范围", rangeStr)
                    }
                }
            }

            Spacer(Modifier.height(12.dp))

            // 操作按钮行
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                // 保存结果按钮
                Button(
                    onClick = onSaveResult,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (resultSaved) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.primary
                    ),
                    enabled = !resultSaved && sampleId.isNotEmpty()
                ) {
                    Icon(if (resultSaved) Icons.Default.Check else Icons.Default.Save, null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text(if (resultSaved) "已保存" else "保存结果", fontSize = 12.sp)
                }
            }

            Spacer(Modifier.height(8.dp))

            // 导出按钮
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedButton(
                    onClick = {
                        exporting = true
                        scope.launch(Dispatchers.IO) {
                            val text = ResultRepository(context).generateTextReport()
                            val file = ResultRepository.exportTextToFile(context, text)
                            withContext(Dispatchers.Main) {
                                exporting = false
                                snackbarHostState.showSnackbar(if (file != null) "文本已导出" else "导出失败")
                            }
                        }
                    },
                    modifier = Modifier.weight(1f),
                    enabled = !exporting
                ) { Icon(Icons.Default.Description, null, modifier = Modifier.size(14.dp)); Spacer(Modifier.width(2.dp)); Text("文本", fontSize = 11.sp) }

                OutlinedButton(
                    onClick = {
                        exporting = true
                        scope.launch(Dispatchers.IO) {
                            val json = ResultRepository(context).generateJsonReport()
                            val file = ResultRepository.exportJsonToFile(context, json)
                            withContext(Dispatchers.Main) {
                                exporting = false
                                snackbarHostState.showSnackbar(if (file != null) "JSON已导出" else "导出失败")
                            }
                        }
                    },
                    modifier = Modifier.weight(1f),
                    enabled = !exporting
                ) { Icon(Icons.Default.Code, null, modifier = Modifier.size(14.dp)); Spacer(Modifier.width(2.dp)); Text("JSON", fontSize = 11.sp) }

                // 导出全部：原图 + Mask + JSON + TXT
                Button(
                    onClick = {
                        exporting = true
                        scope.launch(Dispatchers.IO) {
                            try {
                                val exportDir = File(context.getExternalFilesDir(null), "stem_exports").also { it.mkdirs() }
                                val bName = "${sampleId.filter { it.isLetterOrDigit() }}_${java.text.SimpleDateFormat("HHmmss", java.util.Locale.getDefault()).format(java.util.Date())}"
                                // 原图
                                FileOutputStream(File(exportDir, "${bName}_original.jpg")).use { segResult.original.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                                // Mask
                                FileOutputStream(File(exportDir, "${bName}_mask.png")).use { segResult.mask.compress(Bitmap.CompressFormat.PNG, 100, it) }
                                // JSON
                                File(exportDir, "${bName}.json").writeText(ResultRepository(context).generateJsonReport())
                                // TXT
                                File(exportDir, "${bName}.txt").writeText(ResultRepository(context).generateTextReport())
                                withContext(Dispatchers.Main) { snackbarHostState.showSnackbar("导出完成") }
                            } catch (e: Exception) {
                                e.printStackTrace()
                                withContext(Dispatchers.Main) { snackbarHostState.showSnackbar("导出失败: ${e.message}") }
                            }
                            exporting = false
                        }
                    },
                    modifier = Modifier.weight(1.5f),
                    enabled = !exporting && sampleId.isNotEmpty()
                ) { Icon(Icons.Default.Archive, null, modifier = Modifier.size(16.dp)); Spacer(Modifier.width(2.dp)); Text("导出全部", fontSize = 11.sp) }
            }

            // 文件路径提示
            Text(
                "导出的文件位于: 内部存储/Android/data/com.stemapp/files/stem_exports/",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 8.dp)
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ViewSwitcher(
    currentView: String, onViewChange: (String) -> Unit,
    hasAnalysis: Boolean, isCalibrating: Boolean
) {
    if (currentView == "calibrate" || isCalibrating) return
    Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        val views = listOf("original" to "原图", "mask" to "分割 Mask", "overlay" to "叠加图", "analysis" to "结果")
        views.forEach { (key, label) ->
            if (key == "analysis" && !hasAnalysis) return@forEach
            FilterChip(selected = currentView == key, onClick = { onViewChange(key) }, label = { Text(label, fontSize = 11.sp) })
        }
    }
}

private fun fmtInt(value: Number) = String.format("%,d", value.toLong())
private fun fmtArea(area: Double, unit: String = "px") = when {
    area >= 10000 -> String.format("%,.0f %s", area, unit)
    area >= 100 -> String.format("%,.1f %s", area, unit)
    else -> String.format("%.1f %s", area, unit)
}
private fun fmtUm2(area: Double?) = if (area == null) "" else fmtArea(area, "μm²")

@Composable
private fun InfoRow(label: String, value: String, unit: String = "") {
    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
        Text(label, Modifier.weight(1f), fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(if (unit.isNotEmpty()) "$value $unit" else value, Modifier.weight(1f), fontSize = 14.sp, fontWeight = FontWeight.SemiBold, textAlign = TextAlign.End)
    }
}

private fun saveResult(context: android.content.Context, bitmap: Bitmap) {
    try {
        val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)?.also { it.mkdirs() }
        if (dir != null) {
            val file = File(dir, "stem_seg_${System.currentTimeMillis()}.jpg")
            FileOutputStream(file).use { out -> bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out) }
        }
    } catch (e: Exception) { e.printStackTrace() }
}
