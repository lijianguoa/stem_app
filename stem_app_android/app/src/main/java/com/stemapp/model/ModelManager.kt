package com.stemapp.model

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream

/**
 * 模型管理器
 * 负责从手机文件系统选择并复制 ONNX 模型到 App 内部存储
 */
class ModelManager {

    data class LoadedModel(
        val name: String,
        val filePath: String,
        val fileSizeMb: Double
    )

    private var currentModel: LoadedModel? = null

    /**
     * 获取当前加载的模型信息
     */
    fun getCurrentModel(): LoadedModel? = currentModel

    /**
     * 从 URI 加载 ONNX 模型文件
     * 将文件从外部存储复制到 App 内部缓存目录
     */
    fun loadModelFromUri(context: Context, uri: Uri): LoadedModel? {
        return try {
            // 获取文件名
            val fileName = getFileName(context, uri) ?: "model.onnx"
            
            // 创建内部缓存文件
            val modelDir = File(context.cacheDir, "models")
            if (!modelDir.exists()) modelDir.mkdirs()
            
            val modelFile = File(modelDir, fileName)
            
            // 复制文件
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(modelFile).use { output ->
                    input.copyTo(output)
                }
            }

            val fileSizeMb = modelFile.length() / (1024.0 * 1024.0)

            currentModel = LoadedModel(
                name = fileName,
                filePath = modelFile.absolutePath,
                fileSizeMb = fileSizeMb
            )

            currentModel
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * 从 Assets 目录加载默认模型
     */
    fun loadDefaultModel(context: Context, assetFileName: String): LoadedModel? {
        return try {
            val modelDir = File(context.cacheDir, "models")
            if (!modelDir.exists()) modelDir.mkdirs()

            val modelFile = File(modelDir, assetFileName)

            // 如果缓存中已存在则直接使用
            if (modelFile.exists()) {
                val fileSizeMb = modelFile.length() / (1024.0 * 1024.0)
                currentModel = LoadedModel(
                    name = assetFileName,
                    filePath = modelFile.absolutePath,
                    fileSizeMb = fileSizeMb
                )
                return currentModel
            }

            // 从 Assets 复制
            context.assets.open(assetFileName).use { input ->
                FileOutputStream(modelFile).use { output ->
                    input.copyTo(output)
                }
            }

            val fileSizeMb = modelFile.length() / (1024.0 * 1024.0)
            currentModel = LoadedModel(
                name = assetFileName,
                filePath = modelFile.absolutePath,
                fileSizeMb = fileSizeMb
            )
            currentModel
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * 获取模型文件路径
     */
    fun getModelFilePath(): String? = currentModel?.filePath

    /**
     * 获取用于文件选择的 MIME 类型过滤器
     */
    fun getModelMimeTypes(): Array<String> = arrayOf("application/octet-stream", "*/*")

    /**
     * 从 URI 提取文件名
     */
    private fun getFileName(context: Context, uri: Uri): String? {
        var name: String? = null
        if (uri.scheme == "content") {
            val cursor = context.contentResolver.query(uri, null, null, null, null)
            cursor?.use {
                if (it.moveToFirst()) {
                    val nameIndex = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (nameIndex >= 0) {
                        name = it.getString(nameIndex)
                    }
                }
            }
        }
        if (name == null) {
            name = uri.path
            val cut = name?.lastIndexOf('/')
            if (cut != null && cut != -1) {
                name = name?.substring(cut + 1)
            }
        }
        return name
    }
}