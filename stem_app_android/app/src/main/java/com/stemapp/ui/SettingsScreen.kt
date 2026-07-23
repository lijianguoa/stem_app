package com.stemapp.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Straighten
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.stemapp.model.AppSettings
import com.stemapp.model.SettingsManager

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    settingsManager: SettingsManager,
    onNavigateBack: () -> Unit
) {
    val currentSettings = remember { settingsManager.getSettings() }
    var kmeansK by remember { mutableStateOf(currentSettings.kmeansK) }
    var enableFilter by remember { mutableStateOf(currentSettings.enableFilter) }
    var filterSigma by remember { mutableStateOf(currentSettings.filterSigma.toFloat()) }
    val snackbarHostState = remember { SnackbarHostState() }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("分析设置") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "返回")
                    }
                },
                actions = {
                    IconButton(onClick = {
                        val newSettings = currentSettings.copy(
                            kmeansK = kmeansK,
                            enableFilter = enableFilter,
                            filterSigma = filterSigma.toDouble()
                        )
                        settingsManager.saveSettings(newSettings)
                    }) {
                        Icon(Icons.Default.Save, contentDescription = "保存")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // ===== K-means 聚类 =====
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("K-means 聚类", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("设置维管束面积聚类的类别数 (k 值)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(2, 3, 4, 5).forEach { k ->
                            FilterChip(
                                selected = kmeansK == k,
                                onClick = { kmeansK = k },
                                label = { Text("$k") }
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ===== 3σ 滤波 =====
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("异常值滤波", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("剔除面积偏离均值超过 N 倍标准差的异常维管束",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("启用滤波")
                        Switch(checked = enableFilter, onCheckedChange = { enableFilter = it })
                    }

                    if (enableFilter) {
                        Spacer(modifier = Modifier.height(12.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            listOf(2.0f, 2.5f, 3.0f, 3.5f).forEach { s ->
                                FilterChip(
                                    selected = filterSigma == s,
                                    onClick = { filterSigma = s },
                                    label = { Text("%.1f".format(s)) }
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ===== 比例尺标定（只读显示） =====
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Straighten, contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("比例尺标定", style = MaterialTheme.typography.titleMedium)
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("在结果页打开原图 → 点击「标定比例尺」→ 沿照片上的标尺线点两点 → 输入标尺长度",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)

                    if (currentSettings.enableScale && currentSettings.scaleBarPixels > 0) {
                        val ratio = currentSettings.scaleBarPixels / currentSettings.scaleBarLength
                        Spacer(modifier = Modifier.height(12.dp))
                        Surface(
                            shape = MaterialTheme.shapes.small,
                            color = MaterialTheme.colorScheme.tertiaryContainer,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Text("当前标定", style = MaterialTheme.typography.titleSmall)
                                Spacer(modifier = Modifier.height(4.dp))
                                Text("%.0f px = %.0f μm".format(
                                    currentSettings.scaleBarPixels, currentSettings.scaleBarLength))
                                Text("→ %.2f px/μm".format(ratio))
                                Text("→ 1 px = %.4f μm".format(1.0 / ratio))
                            }
                        }
                    } else {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("（尚未标定）", color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // 说明
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("使用说明", style = MaterialTheme.typography.titleSmall)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "• 在结果页打开原图，点击「标定比例尺」\n" +
                                "• 沿着照片上的旧标尺线，依次点两点\n" +
                                "• 输入标尺上印的数字（如 500 μm）\n" +
                                "• 保存后面积自动换算为 μm²\n" +
                                "• 不需要手动量像素，也无需查倍数",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                }
            }
        }
    }
}
