package com.stemapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.stemapp.ml.OnnxInference
import com.stemapp.model.ModelManager
import com.stemapp.model.SettingsManager
import com.stemapp.ui.CameraScreen
import com.stemapp.ui.ModelSelectScreen
import com.stemapp.ui.ResultScreen
import com.stemapp.ui.SettingsScreen
import com.stemapp.ui.theme.StemAppTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URLDecoder

class MainActivity : ComponentActivity() {

    private val modelManager = ModelManager()
    private val onnxInference = OnnxInference()
    private lateinit var settingsManager: SettingsManager

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* 权限结果处理 */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settingsManager = SettingsManager(this)
        requestCameraPermissions()

        setContent {
            var isModelLoading by remember { mutableStateOf(true) }

            // 在后台线程加载默认模型（避免阻塞 UI）
            LaunchedEffect(Unit) {
                withContext(Dispatchers.IO) {
                    try {
                        val loadedModel = modelManager.loadDefaultModel(
                            this@MainActivity, "model_fp16.onnx"
                        )
                        if (loadedModel != null) {
                            val modelFile = File(loadedModel.filePath)
                            val success = onnxInference.loadModel(modelFile)
                            if (success) {
                                withContext(Dispatchers.Main) {
                                    Toast.makeText(
                                        this@MainActivity,
                                        "已自动加载模型: model_fp16.onnx",
                                        Toast.LENGTH_SHORT
                                    ).show()
                                }
                            } else {
                                // 模型加载失败，CameraScreen 会显示"未加载模型"
                            }
                        } else {
                            // 模型加载失败
                        }
                    } catch (e: Exception) {
                        e.printStackTrace()
                        // 模型加载失败，用户可在相机页选择其他模型
                    }
                }
                isModelLoading = false
            }

            StemAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (isModelLoading) {
                        // 加载模型时的启动屏
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                CircularProgressIndicator()
                                Spacer(modifier = Modifier.height(16.dp))
                                Text("正在加载模型...")
                            }
                        }
                    } else {
                        val navController = rememberNavController()

                        NavHost(
                            navController = navController,
                            startDestination = "camera"
                        ) {
                            composable("camera") {
                                CameraScreen(
                                    onnxInference = onnxInference,
                                    modelManager = modelManager,
                                    onNavigateToResult = { imagePath, sampleId ->
                                        val encPath = java.util.Base64.getUrlEncoder()
                                            .encodeToString(imagePath.toByteArray())
                                        val sid = if (sampleId.isBlank()) "unknown" else sampleId
                                        val encSample = java.util.Base64.getUrlEncoder()
                                            .encodeToString(sid.toByteArray())
                                        navController.navigate("result/$encPath/$encSample")
                                    },
                                    onNavigateToModelSelect = {
                                        navController.navigate("model_select")
                                    },
                                    onNavigateToSettings = {
                                        navController.navigate("settings")
                                    }
                                )
                            }

                            composable(
                                route = "result/{imagePath}/{sampleId}",
                                arguments = listOf(
                                    navArgument("imagePath") { type = NavType.StringType },
                                    navArgument("sampleId") { type = NavType.StringType; defaultValue = "" }
                                )
                            ) { backStackEntry ->
                                val encodedPath = backStackEntry.arguments?.getString("imagePath") ?: ""
                                val imagePath = try {
                                    String(java.util.Base64.getUrlDecoder().decode(encodedPath), Charsets.UTF_8)
                                } catch (e: Exception) { java.net.URLDecoder.decode(encodedPath, "UTF-8") }
                                val encodedSample = backStackEntry.arguments?.getString("sampleId") ?: ""
                                val sampleId = try {
                                    String(java.util.Base64.getUrlDecoder().decode(encodedSample), Charsets.UTF_8)
                                } catch (e: Exception) { "" }
                                ResultScreen(
                                    imagePath = imagePath,
                                    sampleId = sampleId,
                                    onnxInference = onnxInference,
                                    settingsManager = settingsManager,
                                    onNavigateBack = { navController.popBackStack() }
                                )
                            }

                            composable("model_select") {
                                ModelSelectScreen(
                                    modelManager = modelManager,
                                    onnxInference = onnxInference,
                                    onNavigateBack = { navController.popBackStack() }
                                )
                            }

                            composable("settings") {
                                SettingsScreen(
                                    settingsManager = settingsManager,
                                    onNavigateBack = { navController.popBackStack() }
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    private fun requestCameraPermissions() {
        val permissionsToRequest = mutableListOf<String>()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            permissionsToRequest.add(Manifest.permission.CAMERA)
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.READ_MEDIA_IMAGES)
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }

        if (permissionsToRequest.isNotEmpty()) {
            requestPermissionLauncher.launch(permissionsToRequest.toTypedArray())
        }
    }
}
