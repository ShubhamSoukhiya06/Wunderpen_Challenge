package com.example.backend_service.api.controller

import com.google.gson.Gson
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import org.springframework.http.HttpStatus
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.io.File
import java.io.IOException

@RestController
class UploadController2 {
    @PostMapping("/checkImageQuality")
    fun handleFileUpload(
            @RequestParam("file1") file1: MultipartFile,
            @RequestParam("file2") file2: MultipartFile,
            @RequestParam("file3") file3: MultipartFile
    ): ResponseEntity<String> {
        // Specify an absolute path for the uploads directory
        val uploadDir = File("/Users/soukh2/Desktop/backend/backend-service/uploads") // Adjust this path as necessary
        // Create the directory if it does not exist
        if (!uploadDir.exists()) {
            if (!uploadDir.mkdirs()) {
                println("Failed to create directory: ${uploadDir.absolutePath}")
                return ResponseEntity("Failed to create upload directory", HttpStatus.INTERNAL_SERVER_ERROR)
            }
        }

        // Save files locally
        return try {
            val file1Name = "${uploadDir.path}/${file1.originalFilename}"
            val file2Name = "${uploadDir.path}/${file2.originalFilename}"
            val file3Name = "${uploadDir.path}/${file3.originalFilename}"

            file1.transferTo(File(file1Name))
            file2.transferTo(File(file2Name))
            file3.transferTo(File(file3Name))

            // Check if files exist and are not empty
            if (!File(file1Name).exists() || File(file1Name).length() == 0L) {
                throw IOException("File1 not saved correctly or is empty.")
            }
            if (!File(file2Name).exists() || File(file2Name).length() == 0L) {
                throw IOException("File2 not saved correctly or is empty.")
            }
            if (!File(file3Name).exists() || File(file3Name).length() == 0L) {
                throw IOException("File3 not saved correctly or is empty.")
            }

            // Directly call the Python service and wait for its response
            val result = callPythonService(file1Name, file2Name, file3Name)

            // Process the result
            val gson = Gson()
            val response: CheckImageQualityResponse = gson.fromJson(result, CheckImageQualityResponse::class.java)
            val jsonResponse = gson.toJson(response)
            ResponseEntity.ok(jsonResponse)

        } catch (e: IOException) {
            e.printStackTrace()
            ResponseEntity("Error saving files: ${e.message}", HttpStatus.INTERNAL_SERVER_ERROR)
        } catch (e: Exception) {
            e.printStackTrace()
            ResponseEntity("Error processing images: ${e.message}", HttpStatus.INTERNAL_SERVER_ERROR)
        }
    }

    fun callPythonService(file1Path: String, file2Path: String, file3Path: String): String {
        val client = OkHttpClient()
        val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file1", File(file1Path).name, File(file1Path).asRequestBody("image/png".toMediaTypeOrNull()))
                .addFormDataPart("file2", File(file2Path).name, File(file2Path).asRequestBody("image/png".toMediaTypeOrNull()))
                .addFormDataPart("file3", File(file3Path).name, File(file3Path).asRequestBody("image/png".toMediaTypeOrNull()))
                .build()

        val request = Request.Builder()
                .url("http://localhost:5000/process_images")
                .post(requestBody)
                .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("Unexpected code $response")
            return response.body?.string() ?: throw IOException("Empty response body")
        }
    }
}
