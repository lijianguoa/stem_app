package com.stemapp.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.stemapp.ml.OnnxInference
import com.stemapp.model.ModelManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executor

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    onnxInference: OnnxInference,
    modelManager: ModelManager,
    onNavigateToResult: (String, String) -> Unit,
    onNavigateToModelSelect: () -> Unit,
    onNavigateToSettings: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var isProcessing by remember { mutableStateOf(false) }
    var isCameraReady by remember { mutableStateOf(false) }
    val snackbarHostState = remember { SnackbarHostState() }

    val currentModel = modelManager.getCurrentModel()
    val isModelLoaded = currentModel != null

    // 样本ID
    var sampleId by remember { mutableStateOf("") }
    var showSampleIdDialog by remember { mutableStateOf(false) }
    var tempSampleId by remember { mutableStateOf("") }

    // 样本ID输入对话框
    if (showSampleIdDialog) {
        AlertDialog(
            onDismissRequest = { showSampleIdDialog = false },
            title = { Text("样本ID") },
            text = {
                OutlinedTextField(
                    value = tempSampleId,
                    onValueChange = { tempSampleId = it },
                    label = { Text("输入样本编号") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    sampleId = tempSampleId.trim()
                    showSampleIdDialog = false
                }) { Text("确定") }
            },
            dismissButton = {
                TextButton(onClick = { showSampleIdDialog = false }) { Text("取消") }
            }
        )
    }

    val mainExecutor: Executor = ContextCompat.getMainExecutor(context)

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                val filePath = copyUriToFile(context, it)
                if (filePath != null) {
                    onNavigateToResult(filePath, sampleId)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        isProcessing = false
    }

    // 当 PreviewView 创建完毕后，用其 surfaceProvider 初始化相机
    val onPreviewViewCreated: (PreviewView) -> Unit = remember {
        { pv ->
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build()
                    val imageCaptureObj = ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .build()
                    imageCapture = imageCaptureObj

                    // 将 Preview 连接到 PreviewView（关键修复！）
                    preview.setSurfaceProvider(pv.surfaceProvider)

                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageCaptureObj
                    )
                    isCameraReady = true
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }, mainExecutor)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = if (isModelLoaded) "模型: ${currentModel?.name ?: ""}" else "未加载模型",
                        style = MaterialTheme.typography.bodyMedium
                    )
                },
                actions = {
                    // 样本ID按钮
                    IconButton(onClick = {
                        tempSampleId = sampleId
                        showSampleIdDialog = true
                    }) {
                        if (sampleId.isNotEmpty()) {
                            Surface(
                                shape = RoundedCornerShape(12.dp),
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                            ) {
                                Text(sampleId, fontSize = 11.sp,
                                    color = MaterialTheme.colorScheme.onPrimary,
                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp))
                            }
                        } else {
                            Text("+ID", fontSize = 12.sp, color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
                        }
                    }
                    IconButton(onClick = onNavigateToSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "分析设置")
                    }
                    IconButton(onClick = onNavigateToModelSelect) {
                        Icon(Icons.Default.ModelTraining, contentDescription = "选择模型")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // CameraX PreviewView
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx ->
                    PreviewView(ctx).apply {
                        scaleType = PreviewView.ScaleType.FILL_CENTER
                        // 创建完成后回调初始化相机
                        post { onPreviewViewCreated(this) }
                    }
                }
            )

            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 48.dp)
                    .fillMaxWidth()
                    .padding(horizontal = 32.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // 导入图片按钮
                FloatingActionButton(
                    onClick = {
                        isProcessing = true
                        galleryLauncher.launch("image/*")
                    },
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                ) {
                    Icon(Icons.Default.Image, contentDescription = "导入图片")
                }

                // 拍照按钮
                FloatingActionButton(
                    onClick = {
                        if (!isModelLoaded) {
                            CoroutineScope(Dispatchers.Main).launch {
                                snackbarHostState.showSnackbar("请先在右上角加载 ONNX 模型文件")
                            }
                        } else if (!isCameraReady) {
                            CoroutineScope(Dispatchers.Main).launch {
                                snackbarHostState.showSnackbar("相机正在初始化，请稍候...")
                            }
                        } else if (!isProcessing) {
                            isProcessing = true
                            takePhoto(context, imageCapture) { filePath ->
                                isProcessing = false
                                if (filePath != null) {
                                    onNavigateToResult(filePath, sampleId)
                                } else {
                                    CoroutineScope(Dispatchers.Main).launch {
                                        snackbarHostState.showSnackbar("拍照失败，请重试")
                                    }
                                }
                            }
                        }
                    },
                    containerColor = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(72.dp)
                ) {
                    if (isProcessing) {
                        CircularProgressIndicator(
                            color = MaterialTheme.colorScheme.onPrimary,
                            modifier = Modifier.size(32.dp)
                        )
                    } else {
                        Icon(
                            Icons.Default.CameraAlt,
                            contentDescription = "拍照",
                            modifier = Modifier.size(32.dp),
                            tint = MaterialTheme.colorScheme.onPrimary
                        )
                    }
                }

                Spacer(modifier = Modifier.size(56.dp))
            }

            // 未加载模型提示
            if (!isModelLoaded) {
                Surface(
                    modifier = Modifier
                        .align(Alignment.Center)
                        .padding(32.dp),
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.9f)
                ) {
                    Text(
                        text = "请先加载 ONNX 模型文件\n点击右上角图标选择\n或重启 APP 自动加载默认模型",
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.bodyLarge
                    )
                }
            }
        }
    }
}

private fun takePhoto(
    context: android.content.Context,
    imageCapture: ImageCapture?,
    onComplete: (String?) -> Unit
) {
    val photoFile = createImageFile(context)
    val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

    imageCapture?.takePicture(
        outputOptions,
        ContextCompat.getMainExecutor(context),
        object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                onComplete(photoFile.absolutePath)
            }

            override fun onError(exception: ImageCaptureException) {
                exception.printStackTrace()
                onComplete(null)
            }
        }
    ) ?: run { onComplete(null) }
}

private fun createImageFile(context: android.content.Context): File {
    val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    val imageDir = File(context.cacheDir, "photos")
    if (!imageDir.exists()) imageDir.mkdirs()
    return File(imageDir, "IMG_$timeStamp.jpg")
}

private fun copyUriToFile(context: android.content.Context, uri: Uri): String? {
    return try {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val imageDir = File(context.cacheDir, "photos")
        if (!imageDir.exists()) imageDir.mkdirs()
        val file = File(imageDir, "IMG_$timeStamp.jpg")

        context.contentResolver.openInputStream(uri)?.use { input ->
            file.outputStream().use { output ->
                input.copyTo(output)
            }
        } ?: return null
        file.absolutePath
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}
