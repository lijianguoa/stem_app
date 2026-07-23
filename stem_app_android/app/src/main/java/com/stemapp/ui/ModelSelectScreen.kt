package com.stemapp.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.stemapp.ml.OnnxInference
import com.stemapp.model.ModelManager
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelSelectScreen(
    modelManager: ModelManager,
    onnxInference: OnnxInference,
    onNavigateBack: () -> Unit
) {
    val context = LocalContext.current
    var isLoading by remember { mutableStateOf(false) }
    var loadError by remember { mutableStateOf<String?>(null) }
    var modelLoaded by remember { mutableStateOf(onnxInference.isModelLoaded()) }
    var modelInfo by remember { mutableStateOf(onnxInference.getModelInfo()) }
    var currentModel by remember { mutableStateOf(modelManager.getCurrentModel()) }
    val snackbarHostState = remember { SnackbarHostState() }

    // 文件选择器
    val filePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            isLoading = true
            loadError = null

            try {
                // 复制模型文件到内部存储
                val loadedModel = modelManager.loadModelFromUri(context, it)
                if (loadedModel != null) {
                    // 加载到推理引擎
                    val modelFile = File(loadedModel.filePath)
                    val success = onnxInference.loadModel(modelFile)

                    if (success) {
                        modelLoaded = true
                        modelInfo = onnxInference.getModelInfo()
                        currentModel = loadedModel
                        loadError = null
                    } else {
                        loadError = "模型加载失败：请检查模型文件格式"
                    }
                } else {
                    loadError = "文件复制失败"
                }
            } catch (e: Exception) {
                loadError = "加载出错：${e.message}"
            }

            isLoading = false
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("模型管理") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "返回")
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
            // 当前模型状态
            Card(
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "当前模型",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(12.dp))

                    if (currentModel != null) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                Icons.Default.CheckCircle,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(24.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text(
                                    text = currentModel!!.name,
                                    style = MaterialTheme.typography.bodyLarge
                                )
                                Text(
                                    text = "大小: ${"%.1f".format(currentModel!!.fileSizeMb)} MB",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    } else {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                Icons.Default.Info,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.error,
                                modifier = Modifier.size(24.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("未加载模型")
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // 模型信息
            if (modelInfo != null) {
                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "模型信息",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        val info = modelInfo!!
                        ModelInfoRow("输入名称", info.inputName)
                        ModelInfoRow("输出名称", info.outputName)
                        ModelInfoRow(
                            "输入尺寸",
                            "${info.inputWidth}×${info.inputHeight}×${info.inputChannels}"
                        )
                        ModelInfoRow("输出类别数", info.numClasses.toString())
                        ModelInfoRow(
                            "输出尺寸",
                            "${info.outputShape[2]}×${info.outputShape[3]}"
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // 加载模型按钮
            Button(
                onClick = {
                    filePickerLauncher.launch("application/octet-stream")
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isLoading
            ) {
                Icon(Icons.Default.FileOpen, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text(if (isLoading) "加载中..." else "从文件选择 ONNX 模型")
            }

            if (isLoading) {
                Spacer(modifier = Modifier.height(16.dp))
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }

            // 错误信息
            if (loadError != null) {
                Spacer(modifier = Modifier.height(16.dp))
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Text(
                        text = loadError ?: "",
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // 使用说明
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "使用说明",
                        style = MaterialTheme.typography.titleSmall
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "1. 点击上方按钮选择 .onnx 模型文件\n" +
                                "2. 模型会自动加载并显示信息\n" +
                                "3. 返回相机页面即可拍照识别\n" +
                                "4. 支持任意 [1,C,H,W] → [1,N,H,W] 格式的 ONNX 分割模型",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelInfoRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.width(120.dp),
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall
        )
    }
}